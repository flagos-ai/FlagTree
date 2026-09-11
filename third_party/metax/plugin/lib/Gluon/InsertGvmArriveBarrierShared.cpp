#include "Gluon/GluonSharedAccessPath.h"
#include "Gluon/GluonC500LayoutHelpers.h"
#include "Gluon/Passes.h"
#include "Gluon/GluonC500AsyncCopyPlan.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Matchers.h"
#include "mlir/Interfaces/ControlFlowInterfaces.h"
#include "mlir/Interfaces/LoopLikeInterface.h"
#include "triton/Analysis/Allocation.h"
#include "triton/Analysis/Utility.h"
#include "triton/Dialect/Gluon/IR/Dialect.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/Triton/IR/Utility.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/Transforms/Utility.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/raw_ostream.h"
#include <algorithm>
#include <array>
#include <cstdint>
#include <limits>
#include <string>
#include <utility>

using namespace mlir;
namespace tt = mlir::triton;
namespace ttg = mlir::triton::gpu;
namespace gluon_dialect = mlir::triton::gluon;
namespace c500 = mlir::triton::gpu::metax::gluon::c500;
namespace gluon_analysis = mlir::triton::gpu::metax::gluon;

#define DEBUG_TYPE "metax-insert-gvm-arrive-barrier-shared"
#define DBGS() (llvm::dbgs() << "[" DEBUG_TYPE "]: ")
#define LDBG(X) LLVM_DEBUG(DBGS() << X << "\n")

static constexpr StringLiteral kInferredGvmAttr =
    "metax.gluon.inferred_gvm_arrive";
static constexpr StringLiteral kInferredBarrierAttr =
    "metax.gluon.inferred_barrier_shared";

#define GEN_PASS_CLASSES
#include "Gluon/Passes.h.inc"

namespace {

struct RegionKey {
  Value root;
  Value logicalAccess;
  /// Exact access whose complete initialization covers this view. This is set
  /// only for coverage-narrowing views and preserves the one-way relation from
  /// a whole source to its partial child.
  Value coveringAccess;
  Operation *scratchOwner = nullptr;
  Allocation::BufferId allocationId = Allocation::InvalidBufferId;
  std::optional<int64_t> slot;
  SmallVector<Interval<size_t>, 2> physicalIntervals;
  std::string label;
  unsigned slotCount = 1;
  bool valid = false;
  /// Whether the region addresses the root without memdesc_index. Definite
  /// initialization additionally requires `completeCoverage`.
  bool wholeRoot = false;
  /// Whether the view covers all elements of `wholeRoot` or of the selected
  /// constant slot. This is independent of physical alias precision: a
  /// partial subslice still has an exact Allocation interval for hazards, but
  /// cannot prove initialization of that interval.
  bool completeCoverage = false;
};

struct LiveCopyRegion {
  RegionKey region;
  uint64_t tokenEnd = 0;
};

struct SyncState {
  uint64_t issuedTokens = 0;
  uint64_t waitedTokens = 0;
  uint64_t visibleTokens = 0;
  SmallVector<LiveCopyRegion, 8> liveCopies;
  SmallVector<RegionKey, 8> activeReaders;
  SmallVector<RegionKey, 8> pendingSynchronousWrites;
  /// Regions initialized on at least one/all paths reaching this point.
  /// The distinction is required because synchronization cannot initialize a
  /// bypass path that did not execute a producer.
  SmallVector<RegionKey, 8> mayInitialized;
  SmallVector<RegionKey, 8> mustInitialized;
  Operation *lastBarrier = nullptr;
  uint64_t lastBarrierIssuedTokens = 0;
  uint64_t lastBarrierVisibleTokens = 0;
  ttg::GVMArriveOp lastBarrierManualGvm;
  ttg::GVMArriveOp pendingManualGvm;
  /// True when a predicated GVM producer issued an unknown number of tokens.
  /// A later wait cannot preserve an exact suffix and must drain the queue.
  bool hasInexactIssueCount = false;
  /// The exact relative visibility lag was widened at a loop header.
  /// `visibleTokens` remains a conservative lower bound. This domain is
  /// independent of issue-count exactness so nonzero GVM waits remain legal;
  /// a shared barrier re-establishes exact visibility.
  bool hasUnknownVisibilityLag = false;
  /// Whether every thread in the CTA is proven to execute the current region.
  /// Inferred CTA barriers are forbidden when this is false.
  bool collectiveSafe = true;
  /// Planning ignores synchronization owned by an earlier invocation. A
  /// concrete replay/verifier interprets those operations as real hardware
  /// events and must validate their interaction with user-authored waits.
  bool interpretInferredSynchronization = false;
};

enum class WaitReason {
  LoadUse,
  SlotReuse,
  SharedAccessHazard,
  BoundaryDrain,
  UnknownSlotDrain,
};

static const char *stringifyWaitReason(WaitReason reason) {
  switch (reason) {
  case WaitReason::LoadUse:
    return "load-use";
  case WaitReason::SlotReuse:
    return "slot-reuse";
  case WaitReason::SharedAccessHazard:
    return "shared-access-hazard";
  case WaitReason::BoundaryDrain:
    return "boundary-drain";
  case WaitReason::UnknownSlotDrain:
    return "unknown-slot-drain";
  }
  llvm_unreachable("unknown GVM wait reason");
}

struct PlannedSync {
  std::optional<uint32_t> gvmNum;
  bool barrierShared = false;
  WaitReason reason = WaitReason::BoundaryDrain;
};

class GvmInsertionPlan {
public:
  LogicalResult requestGvm(Operation *anchor, uint64_t num,
                           WaitReason reason) {
    if (num > static_cast<uint64_t>(std::numeric_limits<int32_t>::max()))
      return anchor->emitError("gvm_arrive token count exceeds i32 range");

    auto [it, inserted] = insertions.try_emplace(anchor, PlannedSync{});
    if (inserted)
      insertionOrder.push_back(anchor);
    // Fixed-point transfers may visit the same operation more than once.
    // Keep one insertion per anchor and select the smaller outstanding suffix,
    // i.e. the stronger wait that is safe for every abstract iteration.
    if (inserted || !it->second.gvmNum || num < *it->second.gvmNum) {
      it->second.gvmNum = static_cast<uint32_t>(num);
      it->second.reason = reason;
    }
    LDBG("[gvm-plan] anchor=" << anchor->getName() << " num=" << num
                               << " reason=" << stringifyWaitReason(reason));
    return success();
  }

  LogicalResult requestBarrierShared(Operation *anchor, WaitReason reason,
                                     bool collectiveSafe) {
    if (!collectiveSafe)
      return anchor->emitError()
             << "cannot place CTA-wide barrier_shared in control flow that "
                "is not proven CTA-uniform; reconverge before the shared "
                "access or make the branch condition CTA-uniform";
    auto [it, inserted] = insertions.try_emplace(anchor, PlannedSync{});
    if (inserted)
      insertionOrder.push_back(anchor);
    it->second.barrierShared = true;
    if (inserted)
      it->second.reason = reason;
    LDBG("[gvm-plan] anchor=" << anchor->getName()
                                << " barrier-shared=true reason="
                                << stringifyWaitReason(reason));
    return success();
  }

  void materialize() const {
    for (Operation *anchor : insertionOrder) {
      const PlannedSync &insertion = insertions.find(anchor)->second;
      OpBuilder builder(anchor);
      if (insertion.gvmNum) {
        ttg::GVMArriveOp gvm = builder.create<ttg::GVMArriveOp>(
            anchor->getLoc(), *insertion.gvmNum);
        gvm->setAttr(kInferredGvmAttr, builder.getUnitAttr());
      }
      if (insertion.barrierShared) {
        ttg::BarrierSharedOp barrier =
            builder.create<ttg::BarrierSharedOp>(anchor->getLoc());
        barrier->setAttr(kInferredBarrierAttr, builder.getUnitAttr());
      }
      LDBG("[gvm-materialize] anchor="
           << anchor->getName() << " num="
           << (insertion.gvmNum ? std::to_string(*insertion.gvmNum) : "none")
           << " barrier-shared=" << insertion.barrierShared << " reason="
           << stringifyWaitReason(insertion.reason));
    }
  }

  bool empty() const { return insertionOrder.empty(); }

  LogicalResult emitUnexpectedRequirement() const {
    if (empty())
      return success();
    Operation *anchor = insertionOrder.front();
    const PlannedSync &insertion = insertions.find(anchor)->second;
    return anchor->emitError()
           << "concrete C500 synchronization is incomplete before this "
              "operation; verifier would need to insert gvm_arrive="
           << (insertion.gvmNum ? std::to_string(*insertion.gvmNum) : "none")
           << ", barrier_shared=" << insertion.barrierShared;
  }

private:
  DenseMap<Operation *, PlannedSync> insertions;
  SmallVector<Operation *, 8> insertionOrder;
};

static bool haveSamePhysicalAllocationIdentity(const RegionKey &lhs,
                                               const RegionKey &rhs) {
  if (lhs.scratchOwner || rhs.scratchOwner)
    return lhs.scratchOwner && lhs.scratchOwner == rhs.scratchOwner;
  bool lhsHasId = lhs.allocationId != Allocation::InvalidBufferId;
  bool rhsHasId = rhs.allocationId != Allocation::InvalidBufferId;
  if (lhsHasId || rhsHasId)
    return lhsHasId && rhsHasId && lhs.allocationId == rhs.allocationId;
  return lhs.root == rhs.root;
}

static bool haveSameLogicalAccessIdentity(const RegionKey &lhs,
                                          const RegionKey &rhs) {
  if (lhs.scratchOwner || rhs.scratchOwner)
    return lhs.scratchOwner && lhs.scratchOwner == rhs.scratchOwner;
  if (lhs.logicalAccess || rhs.logicalAccess)
    return lhs.logicalAccess && lhs.logicalAccess == rhs.logicalAccess;
  return lhs.root && lhs.root == rhs.root;
}

static bool isSameRegion(const RegionKey &lhs, const RegionKey &rhs) {
  if (!lhs.valid || !rhs.valid ||
      !haveSamePhysicalAllocationIdentity(lhs, rhs) ||
      !haveSameLogicalAccessIdentity(lhs, rhs) ||
      lhs.slot != rhs.slot ||
      lhs.wholeRoot != rhs.wholeRoot ||
      lhs.completeCoverage != rhs.completeCoverage ||
      lhs.physicalIntervals.size() != rhs.physicalIntervals.size())
    return false;
  return llvm::equal(lhs.physicalIntervals, rhs.physicalIntervals);
}

static bool mayAlias(const RegionKey &lhs, const RegionKey &rhs) {
  // Multiple/unknown Allocation intervals can represent a RegionBranch alias
  // set. Keep them at may-alias-all rather than proving disjointness from an
  // incomplete physical identity.
  if (!lhs.valid || !rhs.valid || lhs.physicalIntervals.size() != 1 ||
      rhs.physicalIntervals.size() != 1)
    return true;
  if (!lhs.physicalIntervals.front().intersects(
          rhs.physicalIntervals.front()))
    return false;
  // Different SSA roots (or an explicit root and compiler scratch) that reuse
  // intersecting physical storage necessarily alias. Slot refinement is valid
  // only within one explicit allocation root.
  if (!haveSamePhysicalAllocationIdentity(lhs, rhs))
    return true;
  if (lhs.scratchOwner)
    return true;
  return !lhs.slot || !rhs.slot || lhs.slot == rhs.slot;
}

class RegionKeyBuilder {
public:
  RegionKeyBuilder(Allocation &allocation,
                   gluon_analysis::SharedAccessPathAnalysis &accessPaths)
      : allocation(allocation), accessPaths(accessPaths) {}

  RegionKey get(Value value) {
    Value original = value;
    const gluon_analysis::SharedAccessPathState &state =
        accessPaths.lookup(original);
    Value root = getUniqueLogicalRoot(state);
    Value logicalAccess = accessPaths.getCarrierIdentity(original);
    Value coveringAccess = accessPaths.getCoveringIdentity(original);
    std::optional<int64_t> slot;
    Value identity = root ? root : original;
    const bool pathKnown = root && !state.unknown && !state.alternatives.empty();
    const bool wholeRoot =
        pathKnown && llvm::all_of(state.alternatives, [](const auto &fact) {
          return fact.coverage ==
                 gluon_analysis::SharedAccessCoverage::Whole;
        });
    const bool completeCoverage = pathKnown && llvm::all_of(
        state.alternatives, [](const auto &fact) {
          return fact.coverage ==
                     gluon_analysis::SharedAccessCoverage::Whole ||
                 fact.coverage ==
                     gluon_analysis::SharedAccessCoverage::ExactSlot;
        });
    if (pathKnown && llvm::all_of(state.alternatives, [](const auto &fact) {
          return fact.coverage ==
                     gluon_analysis::SharedAccessCoverage::ExactSlot &&
                 fact.slot.has_value();
        })) {
      int64_t candidate = *state.alternatives.front().slot;
      if (llvm::all_of(state.alternatives, [&](const auto &fact) {
            return *fact.slot == candidate;
          }))
        slot = candidate;
    }
    unsigned slotCount = getSlotCount(root);

    SmallVector<Allocation::BufferId, 2> bufferIds = getBufferIds(original);
    if (bufferIds.empty() && root && root != original)
      bufferIds = getBufferIds(root);
    SmallVector<Interval<size_t>, 2> intervals = getIntervals(bufferIds);
    refineIntervalsWithAccessPath(state, bufferIds, intervals);
    Allocation::BufferId allocationId =
        bufferIds.size() == 1 ? bufferIds.front()
                              : Allocation::InvalidBufferId;

    std::string label = "root" + std::to_string(getRootId(identity));
    if (logicalAccess)
      label += "/carrier:" + std::to_string(getRootId(logicalAccess));
    if (wholeRoot)
      label += "/whole";
    else
      label += slot ? "/slot:" + std::to_string(*slot) : "/slot:*";
    if (allocationId != Allocation::InvalidBufferId)
      label += "/buffer:" + std::to_string(allocationId);
    if (!completeCoverage)
      label += "/partial";
    appendIntervalLabel(label, intervals);
    bool valid = pathKnown && intervals.size() == 1 &&
                 allocationId != Allocation::InvalidBufferId;
    return {identity, logicalAccess, coveringAccess, nullptr, allocationId, slot,
            std::move(intervals), std::move(label), slotCount, valid,
            wholeRoot, completeCoverage};
  }

  std::optional<RegionKey> getScratch(Operation *op) {
    Allocation::BufferId id = allocation.getBufferId(op);
    if (id == Allocation::InvalidBufferId)
      return std::nullopt;
    SmallVector<Interval<size_t>, 2> intervals{
        allocation.getAllocatedInterval(id)};
    std::string label = "scratch:" + op->getName().getStringRef().str();
    appendIntervalLabel(label, intervals);
    return RegionKey{Value(), Value(), Value(), op, id, std::nullopt,
                     std::move(intervals), std::move(label), 1, true, true,
                     true};
  }

  bool hasScratch(Operation *op) {
    return allocation.getBufferId(op) != Allocation::InvalidBufferId;
  }

  ArrayRef<gluon_analysis::RegionCarrierEdge> getCarrierEdges() const {
    return accessPaths.getCarrierEdges();
  }

private:
  static Value getUniqueLogicalRoot(
      const gluon_analysis::SharedAccessPathState &state) {
    if (state.unknown || state.alternatives.empty())
      return {};
    Value root = state.alternatives.front().logicalRoot;
    if (!root || llvm::any_of(state.alternatives, [&](const auto &fact) {
          return fact.logicalRoot != root;
        }))
      return {};
    return root;
  }

  static unsigned getSlotCount(Value root) {
    auto type = root ? dyn_cast<ttg::MemDescType>(root.getType())
                     : ttg::MemDescType();
    if (!type || type.getShape().empty())
      return 1;
    int64_t count = type.getShape().front();
    if (ShapedType::isDynamic(count) || count <= 0 ||
        count > std::numeric_limits<unsigned>::max())
      return 1;
    return static_cast<unsigned>(count);
  }

  SmallVector<Allocation::BufferId, 2> getBufferIds(Value value) {
    SmallVector<Allocation::BufferId, 2> ids;
    llvm::append_range(ids, allocation.getBufferIds(value));
    llvm::sort(ids);
    return ids;
  }

  SmallVector<Interval<size_t>, 2>
  getIntervals(ArrayRef<Allocation::BufferId> bufferIds) {
    SmallVector<Interval<size_t>, 2> intervals;
    for (Allocation::BufferId id : bufferIds)
      intervals.push_back(allocation.getAllocatedInterval(id));
    llvm::sort(intervals);
    intervals.erase(std::unique(intervals.begin(), intervals.end()),
                    intervals.end());
    return intervals;
  }

  void refineIntervalsWithAccessPath(
      const gluon_analysis::SharedAccessPathState &state,
      ArrayRef<Allocation::BufferId> bufferIds,
      SmallVectorImpl<Interval<size_t>> &intervals) {
    if (bufferIds.size() != 1 || state.unknown || state.alternatives.empty() ||
        llvm::any_of(state.alternatives, [](const auto &fact) {
          return !fact.byteInterval.has_value();
        }))
      return;

    Interval<size_t> allocationInterval =
        allocation.getAllocatedInterval(bufferIds.front());
    SmallVector<Interval<size_t>, 2> refined;
    for (const auto &fact : state.alternatives) {
      auto [relativeStart, relativeEnd] = *fact.byteInterval;
      if (relativeStart > std::numeric_limits<size_t>::max() -
                              allocationInterval.start() ||
          relativeEnd > std::numeric_limits<size_t>::max() -
                            allocationInterval.start())
        return;
      refined.emplace_back(allocationInterval.start() + relativeStart,
                           allocationInterval.start() + relativeEnd);
    }
    llvm::sort(refined);
    refined.erase(std::unique(refined.begin(), refined.end()), refined.end());
    intervals.assign(refined.begin(), refined.end());
  }

  static void appendIntervalLabel(
      std::string &label, ArrayRef<Interval<size_t>> intervals) {
    llvm::raw_string_ostream os(label);
    if (intervals.empty()) {
      os << "@physical:*";
      return;
    }
    os << "@physical:";
    for (auto [index, interval] : llvm::enumerate(intervals)) {
      if (index)
        os << ",";
      os << "[" << interval.start() << "," << interval.end() << ")";
    }
  }

  unsigned getRootId(Value root) {
    auto it = rootIds.find(root);
    if (it != rootIds.end())
      return it->second;
    unsigned id = nextRootId++;
    rootIds[root] = id;
    return id;
  }

  DenseMap<Value, unsigned> rootIds;
  unsigned nextRootId = 0;
  Allocation &allocation;
  gluon_analysis::SharedAccessPathAnalysis &accessPaths;
};

static bool isTargetOp(Operation *op, RegionKeyBuilder &keyBuilder) {
  return isa<ttg::AsyncCopyGlobalToLocalOp, ttg::LocalLoadOp,
             ttg::LocalStoreOp, ttg::LocalAllocOp, ttg::GVMArriveOp,
             ttg::BarrierSharedOp, ttg::BarrierOp, tt::AtomicRMWOp,
             tt::AtomicCASOp>(op) ||
         keyBuilder.hasScratch(op);
}

struct GvmIssuePlan {
  uint64_t minTokens = 0;
  uint64_t maxTokens = 0;
  StringRef producer;

  bool isExact() const { return minTokens == maxTokens; }
};

static std::optional<bool> getConstantMaskValue(Value mask) {
  if (!mask)
    return true;
  Attribute value;
  if (!matchPattern(mask, m_Constant(&value)))
    return std::nullopt;
  if (auto integer = dyn_cast<IntegerAttr>(value))
    return !integer.getValue().isZero();
  auto elements = dyn_cast<DenseElementsAttr>(value);
  if (!elements || !elements.isSplat() ||
      !elements.getElementType().isInteger(1))
    return std::nullopt;
  return !elements.getSplatValue<APInt>().isZero();
}

static bool isCTAUniformWhile(scf::WhileOp whileOp);

static bool
isCTAUniformImpl(Value value, DenseSet<Value> &visiting,
                 const DenseSet<Value> *assumedUniform = nullptr) {
  if (!value || isa<ShapedType>(value.getType()) ||
      isa<ttg::MemDescType>(value.getType()))
    return false;
  if (assumedUniform && assumedUniform->contains(value))
    return true;

  if (auto argument = dyn_cast<BlockArgument>(value)) {
    Operation *parent = argument.getOwner()->getParentOp();
    // Scalar kernel arguments are loaded uniformly for the CTA.
    if (isa_and_nonnull<tt::FuncOp>(parent))
      return true;
    if (auto forOp = dyn_cast_or_null<scf::ForOp>(parent)) {
      if (argument == forOp.getInductionVar())
        return isCTAUniformImpl(forOp.getLowerBound(), visiting,
                                assumedUniform) &&
               isCTAUniformImpl(forOp.getUpperBound(), visiting,
                                assumedUniform) &&
               isCTAUniformImpl(forOp.getStep(), visiting, assumedUniform);
    }
    return false;
  }

  Operation *def = value.getDefiningOp();
  if (!def || !visiting.insert(value).second)
    return false;

  bool uniform = false;
  StringRef name = def->getName().getStringRef();
  if (name == "tt.get_program_id" || name == "tt.get_num_programs") {
    uniform = true;
  } else if (auto whileOp = dyn_cast<scf::WhileOp>(def)) {
    // A structured while result is one of the values forwarded by its false
    // condition edge. Reuse the loop-carrier induction proof instead of
    // treating the RegionBranch result as an opaque scalar. This matters for
    // ordinary sequential loops, where one uniform loop's result seeds the
    // next loop, and remains conservative when any carrier is unproven.
    uniform = isCTAUniformWhile(whileOp);
  } else if ((name.starts_with("arith.") ||
              isa<tt::LoadOp, tt::AddPtrOp, tt::IntToPtrOp, tt::PtrToIntOp,
                  tt::BitcastOp>(def)) &&
             def->getNumRegions() == 0) {
    // Scalar Triton pointer arithmetic and loads preserve CTA uniformity when
    // their complete operand slice is uniform. Tensor results were rejected
    // above, so this cannot accidentally classify a per-lane load as uniform.
    uniform = llvm::all_of(def->getOperands(), [&](Value operand) {
      return isCTAUniformImpl(operand, visiting, assumedUniform);
    });
  }
  visiting.erase(value);
  return uniform;
}

static bool isCTAUniform(Value value) {
  DenseSet<Value> visiting;
  return isCTAUniformImpl(value, visiting);
}

static bool isCTAUniform(Value value,
                         const DenseSet<Value> &assumedUniform) {
  DenseSet<Value> visiting;
  return isCTAUniformImpl(value, visiting, &assumedUniform);
}

/// Prove CTA-uniform execution of both regions by induction over the values
/// carried around an `scf.while`. This deliberately requires every carried
/// scalar to remain uniform: it is conservative, but prevents a divergent
/// future condition from making an inferred CTA barrier illegal.
static bool isCTAUniformWhile(scf::WhileOp whileOp) {
  if (!llvm::all_of(whileOp.getInits(),
                    [](Value value) { return isCTAUniform(value); }))
    return false;

  DenseSet<Value> assumedUniform;
  assumedUniform.insert(whileOp.getBeforeArguments().begin(),
                        whileOp.getBeforeArguments().end());

  scf::ConditionOp condition = whileOp.getConditionOp();
  if (!isCTAUniform(condition.getCondition(), assumedUniform) ||
      !llvm::all_of(condition.getArgs(), [&](Value value) {
        return isCTAUniform(value, assumedUniform);
      }))
    return false;

  assumedUniform.insert(whileOp.getAfterArguments().begin(),
                        whileOp.getAfterArguments().end());
  return llvm::all_of(whileOp.getYieldOp().getResults(), [&](Value value) {
    return isCTAUniform(value, assumedUniform);
  });
}

static bool isCTAUniformLoop(LoopLikeOpInterface loop) {
  if (auto forOp = dyn_cast<scf::ForOp>(loop.getOperation()))
    return isCTAUniform(forOp.getLowerBound()) &&
           isCTAUniform(forOp.getUpperBound()) &&
           isCTAUniform(forOp.getStep());
  if (auto whileOp = dyn_cast<scf::WhileOp>(loop.getOperation()))
    return isCTAUniformWhile(whileOp);
  return false;
}

static bool matchPositiveConstantForBounds(scf::ForOp forOp,
                                           APInt &lowerBound,
                                           APInt &upperBound, APInt &step) {
  return matchPattern(forOp.getLowerBound(), m_ConstantInt(&lowerBound)) &&
         matchPattern(forOp.getUpperBound(), m_ConstantInt(&upperBound)) &&
         matchPattern(forOp.getStep(), m_ConstantInt(&step)) &&
         !step.isZero() && !step.isNegative();
}

static bool loopMayExecuteZeroTimes(LoopLikeOpInterface loop) {
  auto forOp = dyn_cast<scf::ForOp>(loop.getOperation());
  if (!forOp)
    return true;
  APInt lowerBound;
  APInt upperBound;
  APInt step;
  if (!matchPositiveConstantForBounds(forOp, lowerBound, upperBound, step))
    return true;
  return lowerBound.sge(upperBound);
}

static bool loopExecutesExactlyOnce(LoopLikeOpInterface loop) {
  auto forOp = dyn_cast<scf::ForOp>(loop.getOperation());
  if (!forOp)
    return false;
  APInt lowerBound;
  APInt upperBound;
  APInt step;
  if (!matchPositiveConstantForBounds(forOp, lowerBound, upperBound, step) ||
      lowerBound.sge(upperBound))
    return false;
  bool overflow = false;
  APInt next = lowerBound.sadd_ov(step, overflow);
  return !overflow && next.sge(upperBound);
}

static bool isCTAUniformBranch(RegionBranchOpInterface branch) {
  if (auto ifOp = dyn_cast<scf::IfOp>(branch.getOperation()))
    return isCTAUniform(ifOp.getCondition());
  return false;
}

/// The MetaX atomic lowering predicates threads for which
/// `tid * elemsPerThread >= numElements`. Prove that predicate true for every
/// participating thread before treating the issue count as exact.
static bool hasUniformAtomicThreadPredicate(tt::AtomicRMWOp atomic,
                                            uint64_t elemsPerThread) {
  auto tensorType = dyn_cast<RankedTensorType>(atomic.getVal().getType());
  if (!tensorType)
    return false;
  ModuleOp module = atomic->getParentOfType<ModuleOp>();
  std::optional<int> numWarps = ttg::maybeLookupNumWarps(atomic);
  if (!module || !numWarps || *numWarps <= 0)
    return false;
  int threadsPerWarp = ttg::TritonGPUDialect::getThreadsPerWarp(module);
  if (threadsPerWarp <= 0)
    return false;

  uint64_t threadCount = static_cast<uint64_t>(*numWarps);
  if (threadCount > std::numeric_limits<uint64_t>::max() /
                        static_cast<uint64_t>(threadsPerWarp))
    return false;
  threadCount *= static_cast<uint64_t>(threadsPerWarp);
  if (elemsPerThread != 0 &&
      threadCount > std::numeric_limits<uint64_t>::max() / elemsPerThread)
    return false;
  uint64_t distributedElements = threadCount * elemsPerThread;
  return distributedElements <=
         static_cast<uint64_t>(tensorType.getNumElements());
}

static FailureOr<GvmIssuePlan> getGvmIssuePlan(Operation *op) {
  return TypeSwitch<Operation *, FailureOr<GvmIssuePlan>>(op)
      .Case<ttg::AsyncCopyGlobalToLocalOp>([&](auto copy) {
        FailureOr<c500::AsyncCopyIssuePlan> plan =
            c500::getC500AsyncCopyIssuePlan(copy);
        if (failed(plan)) {
          copy.emitError()
              << "cannot infer GVM tokens without a valid C500 async-copy "
                 "issue plan";
          return FailureOr<GvmIssuePlan>(failure());
        }
        uint64_t tokens = std::max<uint64_t>(1, plan->instructionsPerThread);
        return FailureOr<GvmIssuePlan>(
            GvmIssuePlan{tokens, tokens, "async-copy"});
      })
      .Case<tt::AtomicRMWOp>([&](auto atomic) {
        uint64_t tokens = ttg::getTotalElemsPerThread(atomic.getVal().getType());
        std::optional<bool> mask = getConstantMaskValue(atomic.getMask());
        if (mask && !*mask)
          return FailureOr<GvmIssuePlan>(
              GvmIssuePlan{0, 0, "atomic-rmw"});
        bool uniformIssue = mask && *mask &&
                            hasUniformAtomicThreadPredicate(atomic, tokens);
        return FailureOr<GvmIssuePlan>(GvmIssuePlan{
            uniformIssue ? tokens : 0, tokens, "atomic-rmw"});
      })
      .Case<tt::AtomicCASOp>([&](auto atomic) {
        // Atomic CAS vectorization is selected during LLVM lowering. Keep an
        // interval here so any synchronization crossing CAS conservatively
        // drains the queue instead of assuming a target-dependent width.
        uint64_t tokens = ttg::getTotalElemsPerThread(atomic.getVal().getType());
        return FailureOr<GvmIssuePlan>(
            GvmIssuePlan{0, tokens, "atomic-cas"});
      })
      .Default([](Operation *) {
        return FailureOr<GvmIssuePlan>(failure());
      });
}

static FailureOr<uint64_t>
getGvmTokenCount(ttg::AsyncCopyGlobalToLocalOp copy) {
  FailureOr<GvmIssuePlan> plan = getGvmIssuePlan(copy);
  if (failed(plan) || !plan->isExact())
    return failure();
  return plan->maxTokens;
}

static Value getCopyDstMemDesc(ttg::AsyncCopyGlobalToLocalOp copy) {
  return copy->getOperand(1);
}

static bool hasNestedTargetOp(Operation *op, RegionKeyBuilder &keyBuilder) {
  return op
      ->walk([&](Operation *nested) {
        if (nested == op)
          return WalkResult::advance();
        return isTargetOp(nested, keyBuilder) ? WalkResult::interrupt()
                                              : WalkResult::advance();
      })
      .wasInterrupted();
}

static bool hasNestedTargetOp(Region &region,
                              RegionKeyBuilder &keyBuilder) {
  return region
      .walk([&](Operation *op) {
        return isTargetOp(op, keyBuilder) ? WalkResult::interrupt()
                                          : WalkResult::advance();
      })
      .wasInterrupted();
}

static void pruneVisibleCopies(SyncState &state) {
  llvm::erase_if(state.liveCopies, [&](const LiveCopyRegion &copy) {
    return copy.tokenEnd <= state.visibleTokens;
  });
}

static bool aliasesAny(const RegionKey &region,
                       ArrayRef<RegionKey> candidates) {
  return llvm::any_of(candidates, [&](const RegionKey &candidate) {
    return mayAlias(region, candidate);
  });
}

static void appendUniqueRegion(SmallVectorImpl<RegionKey> &regions,
                               const RegionKey &region) {
  if (llvm::none_of(regions, [&](const RegionKey &candidate) {
        return (!region.valid && !candidate.valid) ||
               isSameRegion(region, candidate);
      }))
    regions.push_back(region);
}

static bool isPreciseInitializationRegion(const RegionKey &region) {
  return region.valid &&
         (region.logicalAccess ||
          (region.completeCoverage &&
           (region.wholeRoot || region.slot.has_value())));
}

static bool initializationCovers(const RegionKey &initialized,
                                 const RegionKey &accessed) {
  if (!initialized.valid || !accessed.valid)
    return false;
  if (initialized.logicalAccess && accessed.logicalAccess &&
      initialized.logicalAccess == accessed.logicalAccess)
    return true;
  if (!initialized.completeCoverage)
    return false;
  if (initialized.logicalAccess && accessed.coveringAccess &&
      initialized.logicalAccess == accessed.coveringAccess)
    return true;
  if (!initialized.root || initialized.root != accessed.root)
    return false;
  return initialized.wholeRoot ||
         (initialized.slot && accessed.slot &&
          initialized.slot == accessed.slot);
}

static void markInitialized(SyncState &state, const RegionKey &region) {
  // A dynamic slot identifies some part of the root, but not which part. It is
  // a may-write only and cannot establish definite initialization.
  if (!region.valid)
    return;
  appendUniqueRegion(state.mayInitialized, region);
  if (isPreciseInitializationRegion(region))
    appendUniqueRegion(state.mustInitialized, region);
}

static void markUninitialized(SyncState &state, const RegionKey &region) {
  if (!region.valid)
    return;
  llvm::erase_if(state.mayInitialized, [&](const RegionKey &candidate) {
    return mayAlias(candidate, region);
  });
  llvm::erase_if(state.mustInitialized, [&](const RegionKey &candidate) {
    return mayAlias(candidate, region);
  });
}

static bool isDefinitelyInitialized(const SyncState &state,
                                    const RegionKey &region) {
  return llvm::any_of(state.mustInitialized, [&](const RegionKey &candidate) {
    return initializationCovers(candidate, region);
  });
}

static bool isMaybeInitialized(const SyncState &state,
                               const RegionKey &region) {
  return llvm::any_of(state.mayInitialized, [&](const RegionKey &candidate) {
    return initializationCovers(candidate, region);
  });
}

/// Transfer logical initialization facts along one RegionBranch control-flow
/// edge. Physical queue and hazard facts need no SSA renaming because they are
/// queried through final Allocation intervals. Initialization is different: a
/// region argument/result denotes the exact predecessor access only on the
/// traversed edge, so globally unioning mixed-slot carrier identities would be
/// unsound.
static void projectInitializationPairs(
    ArrayRef<std::pair<Value, Value>> edges, SyncState &state,
    RegionKeyBuilder &keyBuilder, StringRef debugScope) {
  struct Projection {
    RegionKey successor;
    bool may = false;
    bool must = true;
    bool seen = false;
  };

  const SyncState incoming = state;
  SmallVector<Projection, 4> projections;
  for (auto [predecessorValue, successorValue] : edges) {
    if (!isa<ttg::MemDescType>(predecessorValue.getType()) ||
        !isa<ttg::MemDescType>(successorValue.getType()))
      continue;

    RegionKey predecessor = keyBuilder.get(predecessorValue);
    RegionKey successor = keyBuilder.get(successorValue);
    auto found = llvm::find_if(projections, [&](const Projection &projection) {
      return isSameRegion(projection.successor, successor);
    });
    if (found == projections.end()) {
      projections.push_back({successor});
      found = std::prev(projections.end());
    }
    bool edgeMay = isMaybeInitialized(incoming, predecessor);
    bool edgeMust = isDefinitelyInitialized(incoming, predecessor);
    found->may |= edgeMay;
    found->must &= edgeMust;
    found->seen = true;
    LDBG("[gvm-carrier-project] scope="
         << debugScope << " predecessor=" << predecessor.label
         << " successor=" << successor.label << " may=" << edgeMay
         << " must=" << edgeMust);
  }

  for (const Projection &projection : projections) {
    if (!projection.seen || !projection.successor.valid)
      continue;
    llvm::erase_if(state.mayInitialized, [&](const RegionKey &candidate) {
      return isSameRegion(candidate, projection.successor);
    });
    llvm::erase_if(state.mustInitialized, [&](const RegionKey &candidate) {
      return isSameRegion(candidate, projection.successor);
    });
    if (projection.may || projection.must)
      appendUniqueRegion(state.mayInitialized, projection.successor);
    if (projection.must &&
        isPreciseInitializationRegion(projection.successor))
      appendUniqueRegion(state.mustInitialized, projection.successor);
  }
}

static void projectCarrierInitialization(Operation *owner,
                                         Region *predecessorRegion,
                                         Region *successorRegion,
                                         SyncState &state,
                                         RegionKeyBuilder &keyBuilder) {
  SmallVector<std::pair<Value, Value>, 4> edges;
  for (const gluon_analysis::RegionCarrierEdge &edge :
       keyBuilder.getCarrierEdges())
    if (edge.owner == owner &&
        edge.predecessorRegion == predecessorRegion &&
        edge.successorRegion == successorRegion)
      edges.emplace_back(edge.predecessor, edge.successorInput);
  projectInitializationPairs(edges, state, keyBuilder,
                             owner->getName().getStringRef());
}

static void projectLoopZeroTripInitialization(
    LoopLikeOpInterface loop, SyncState &state,
    RegionKeyBuilder &keyBuilder) {
  std::optional<ResultRange> results = loop.getLoopResults();
  if (!results)
    return;
  SmallVector<std::pair<Value, Value>, 4> edges;
  for (auto [init, result] : llvm::zip_equal(loop.getInits(), *results))
    edges.emplace_back(init, result);
  projectInitializationPairs(edges, state, keyBuilder, "loop-zero-trip");
}

static bool isLocalAllocation(const RegionKey &region) {
  return region.valid && !region.scratchOwner &&
         region.allocationId != Allocation::InvalidBufferId;
}

static void applySharedBarrier(SyncState &state) {
  state.visibleTokens = std::max(state.visibleTokens, state.waitedTokens);
  state.hasUnknownVisibilityLag = false;
  state.activeReaders.clear();
  state.pendingSynchronousWrites.clear();
  pruneVisibleCopies(state);
}

static uint64_t getAliasingCopyEnd(const RegionKey &region,
                                   const SyncState &state) {
  uint64_t requiredTokenEnd = state.visibleTokens;
  for (const LiveCopyRegion &copy : state.liveCopies)
    if (mayAlias(region, copy.region))
      requiredTokenEnd = std::max(requiredTokenEnd, copy.tokenEnd);
  return requiredTokenEnd;
}

static void resetBarrierAnchor(SyncState &state);

static LogicalResult requireVisibleBefore(Operation *anchor,
                                          uint64_t requiredTokenEnd,
                                          WaitReason reason, SyncState &state,
                                          GvmInsertionPlan &plan) {
  if (requiredTokenEnd <= state.visibleTokens)
    return success();
  if (requiredTokenEnd > state.issuedTokens)
    return anchor->emitError("visibility requires a future async token");
  if (state.hasInexactIssueCount) {
    if (failed(plan.requestGvm(anchor, /*num=*/0, reason)))
      return failure();
    if (failed(plan.requestBarrierShared(anchor, reason,
                                         state.collectiveSafe)))
      return failure();
    state.waitedTokens = state.issuedTokens;
    state.hasInexactIssueCount = false;
    applySharedBarrier(state);
    resetBarrierAnchor(state);
    LDBG("[gvm-state] conservative queue drain for inexact issue count "
         << "before shared visibility at " << anchor->getName());
    return success();
  }
  bool canCompleteAtExistingBarrier =
      state.lastBarrier &&
      state.lastBarrierIssuedTokens >= requiredTokenEnd &&
      !state.lastBarrierManualGvm && !state.hasUnknownVisibilityLag;
  if (canCompleteAtExistingBarrier) {
    uint64_t num = state.lastBarrierIssuedTokens - requiredTokenEnd;
    if (failed(plan.requestGvm(state.lastBarrier, num, reason)))
      return failure();
    state.waitedTokens = std::max(state.waitedTokens, requiredTokenEnd);
    state.visibleTokens = std::max(state.visibleTokens, requiredTokenEnd);
    state.lastBarrierVisibleTokens =
        std::max(state.lastBarrierVisibleTokens, requiredTokenEnd);
    pruneVisibleCopies(state);
  } else {
    // A prior manual wait may already have completed the required copy. In
    // that case only shared visibility is missing; emitting a larger GVM
    // suffix would move the queue backwards.
    if (requiredTokenEnd > state.waitedTokens) {
      uint64_t num = state.issuedTokens - requiredTokenEnd;
      if (failed(plan.requestGvm(anchor, num, reason)))
        return failure();
      state.waitedTokens = requiredTokenEnd;
    }
    if (failed(plan.requestBarrierShared(anchor, reason,
                                         state.collectiveSafe)))
      return failure();
    applySharedBarrier(state);
    state.lastBarrier = nullptr;
    state.lastBarrierIssuedTokens = 0;
    state.lastBarrierVisibleTokens = state.visibleTokens;
    state.lastBarrierManualGvm = {};
    state.pendingManualGvm = {};
  }
  LDBG("[gvm-state] inferred wait reason=" << stringifyWaitReason(reason)
                                            << " issued=" << state.issuedTokens
                                            << " waited=" << state.waitedTokens
                                            << " visible="
                                            << state.visibleTokens
                                            << " required=" << requiredTokenEnd);
  return success();
}

static LogicalResult requireQueueCompletionBefore(
    Operation *anchor, uint64_t requiredTokenEnd, WaitReason reason,
    SyncState &state, GvmInsertionPlan &plan) {
  if (requiredTokenEnd <= state.waitedTokens)
    return success();
  if (requiredTokenEnd > state.issuedTokens)
    return anchor->emitError("queue wait requires a future async token");

  if (state.hasInexactIssueCount) {
    if (failed(plan.requestGvm(anchor, /*num=*/0, reason)))
      return failure();
    state.waitedTokens = state.issuedTokens;
    state.hasInexactIssueCount = false;
    LDBG("[gvm-state] conservative queue drain for inexact issue count "
         << "reason=" << stringifyWaitReason(reason));
    return success();
  }

  uint64_t num = state.issuedTokens - requiredTokenEnd;
  if (failed(plan.requestGvm(anchor, num, reason)))
    return failure();
  state.waitedTokens = requiredTokenEnd;
  LDBG("[gvm-state] queue wait reason=" << stringifyWaitReason(reason)
                                        << " num=" << num
                                        << " issued=" << state.issuedTokens
                                        << " waited=" << state.waitedTokens);
  return success();
}

static LogicalResult insertDrainBefore(Operation *anchor, WaitReason reason,
                                       SyncState &state,
                                       GvmInsertionPlan &plan) {
  return requireQueueCompletionBefore(anchor, state.issuedTokens, reason, state,
                                      plan);
}

static LogicalResult handleGvm(ttg::GVMArriveOp gvm, SyncState &state) {
  const uint64_t threshold = gvm.getNum();
  const uint64_t outstandingUpper = state.issuedTokens - state.waitedTokens;

  if (state.hasInexactIssueCount && threshold != 0)
    return gvm.emitError()
           << "nonzero gvm_arrive cannot be proven safe after a dynamically "
              "predicated GVM producer; use gvm_arrive(0) or make the "
              "producer count uniform";

  // gvm_arrive(N) establishes "outstanding <= N"; it does not require the
  // queue to contain exactly N entries. Therefore a threshold at or above the
  // current upper bound is a no-op. This also preserves the monotonic waited
  // frontier when a weaker wait follows a stronger one.
  if (threshold >= outstandingUpper) {
    state.pendingManualGvm = gvm;
    LDBG("[gvm-state] manual wait no-op threshold="
         << threshold << " outstanding-upper=" << outstandingUpper
         << " issued=" << state.issuedTokens
         << " waited=" << state.waitedTokens
         << " inexact=" << state.hasInexactIssueCount);
    return success();
  }

  state.waitedTokens = state.issuedTokens - threshold;
  if (threshold == 0)
    state.hasInexactIssueCount = false;
  state.pendingManualGvm = gvm;
  LDBG("[gvm-state] manual wait threshold=" << threshold
                                      << " issued=" << state.issuedTokens
                                      << " waited=" << state.waitedTokens);
  return success();
}

static void handleUserBarrier(Operation *barrier, SyncState &state) {
  state.lastBarrier = barrier;
  state.lastBarrierIssuedTokens = state.issuedTokens;
  applySharedBarrier(state);
  state.lastBarrierVisibleTokens = state.visibleTokens;
  state.lastBarrierManualGvm = state.pendingManualGvm;
  state.pendingManualGvm = {};
  LDBG("[gvm-state] user barrier issued=" << state.issuedTokens
                                          << " waited=" << state.waitedTokens
                                          << " visible="
                                          << state.visibleTokens);
}

static void resetBarrierAnchor(SyncState &state) {
  state.lastBarrier = nullptr;
  state.lastBarrierIssuedTokens = 0;
  state.lastBarrierVisibleTokens = state.visibleTokens;
  state.lastBarrierManualGvm = {};
  state.pendingManualGvm = {};
}

static LogicalResult requireSharedAccessBarrierBefore(
    Operation *anchor, ArrayRef<RegionKey> regions, bool checkReaders,
    bool checkSynchronousWrites, SyncState &state, GvmInsertionPlan &plan) {
  bool hasHazard = llvm::any_of(regions, [&](const RegionKey &region) {
    return (checkReaders && aliasesAny(region, state.activeReaders)) ||
           (checkSynchronousWrites &&
            aliasesAny(region, state.pendingSynchronousWrites));
  });
  if (!hasHazard)
    return success();

  if (failed(plan.requestBarrierShared(anchor,
                                       WaitReason::SharedAccessHazard,
                                       state.collectiveSafe)))
    return failure();
  applySharedBarrier(state);
  resetBarrierAnchor(state);
  LDBG("[gvm-state] inferred shared barrier before "
       << anchor->getName() << " for shared access hazard");
  return success();
}

static LogicalResult handleSynchronousSharedWrite(
    Operation *anchor, Value destination, SyncState &state,
    RegionKeyBuilder &keyBuilder, GvmInsertionPlan &plan) {
  RegionKey key = keyBuilder.get(destination);
  uint64_t requiredTokens = getAliasingCopyEnd(key, state);
  if (failed(requireVisibleBefore(anchor, requiredTokens,
                                  WaitReason::SharedAccessHazard, state,
                                  plan)))
    return failure();

  SmallVector<RegionKey, 1> regions{key};
  if (failed(requireSharedAccessBarrierBefore(
          anchor, regions, /*checkReaders=*/true,
          /*checkSynchronousWrites=*/true, state, plan)))
    return failure();
  appendUniqueRegion(state.pendingSynchronousWrites, key);
  markInitialized(state, key);
  LDBG("[gvm-state] synchronous shared write region="
       << (key.valid ? key.label : "<unknown>"));
  return success();
}

static LogicalResult handleScratchBefore(Operation *op,
                                         const RegionKey &scratch,
                                         SyncState &state,
                                         GvmInsertionPlan &plan) {
  if (!state.collectiveSafe)
    return op->emitError()
           << "operation with shared-memory scratch " << scratch.label
           << " is nested in control flow that is not proven CTA-uniform";

  uint64_t requiredTokens = getAliasingCopyEnd(scratch, state);
  if (failed(requireVisibleBefore(op, requiredTokens,
                                  WaitReason::SharedAccessHazard, state,
                                  plan)))
    return failure();
  SmallVector<RegionKey, 1> regions{scratch};
  if (failed(requireSharedAccessBarrierBefore(
          op, regions, /*checkReaders=*/true,
          /*checkSynchronousWrites=*/true, state, plan)))
    return failure();
  LDBG("[gvm-state] scratch write begins region=" << scratch.label
                                                   << " op="
                                                   << op->getName());
  return success();
}

static void handleScratchAfter(Operation *op, const RegionKey &scratch,
                               SyncState &state) {
  // Allocation-backed scratch lowers as write -> internal synchronization ->
  // read. Keep both phases live after the op so physical reuse by any later
  // explicit buffer or scratch operation receives a CTA barrier.
  appendUniqueRegion(state.activeReaders, scratch);
  appendUniqueRegion(state.pendingSynchronousWrites, scratch);
  LDBG("[gvm-state] scratch read/write remains active region="
       << scratch.label << " op=" << op->getName());
}

static LogicalResult handleCopy(ttg::AsyncCopyGlobalToLocalOp copy,
                                SyncState &state,
                                RegionKeyBuilder &keyBuilder,
                                GvmInsertionPlan &plan) {
  if (!state.collectiveSafe)
    return copy.emitError()
           << "async global-to-shared copy is nested in control flow that is "
              "not proven CTA-uniform";
  RegionKey key = keyBuilder.get(getCopyDstMemDesc(copy));
  SmallVector<RegionKey, 1> regions{key};
  if (failed(requireSharedAccessBarrierBefore(
          copy, regions, /*checkReaders=*/true,
          /*checkSynchronousWrites=*/true, state, plan)))
    return failure();
  uint64_t overwrittenTokenEnd = getAliasingCopyEnd(key, state);
  bool reusesRegion = overwrittenTokenEnd > state.visibleTokens;
  WaitReason reason = key.valid ? WaitReason::SlotReuse
                                : WaitReason::UnknownSlotDrain;
  if (failed(requireQueueCompletionBefore(copy, overwrittenTokenEnd, reason,
                                          state, plan)))
    return failure();

  FailureOr<uint64_t> tokens = getGvmTokenCount(copy);
  if (failed(tokens))
    return failure();
  if (*tokens > std::numeric_limits<uint64_t>::max() - state.issuedTokens)
    return copy.emitError("GVM token epoch overflow");
  LDBG("async_copy tokens=" << *tokens << " issued-begin="
                            << state.issuedTokens << " reuse=" << reusesRegion
                            << " region="
                            << (key.valid ? key.label : "<unknown>")
                            << " slot-count="
                            << (key.valid ? key.slotCount : 0));
  state.issuedTokens += *tokens;
  llvm::erase_if(state.liveCopies, [&](const LiveCopyRegion &live) {
    return isSameRegion(key, live.region) ||
           (!key.valid && !live.region.valid);
  });
  state.liveCopies.push_back({key, state.issuedTokens});
  markInitialized(state, key);
  return success();
}

static LogicalResult handleAtomicIssue(Operation *atomic, SyncState &state) {
  FailureOr<GvmIssuePlan> issue = getGvmIssuePlan(atomic);
  if (failed(issue))
    return atomic->emitError("cannot derive a GVM issue plan for atomic op");
  if (issue->maxTokens >
      std::numeric_limits<uint64_t>::max() - state.issuedTokens)
    return atomic->emitError("GVM token epoch overflow");
  state.issuedTokens += issue->maxTokens;
  state.hasInexactIssueCount |= !issue->isExact();
  LDBG("[gvm-state] " << issue->producer << " issues ["
                       << issue->minTokens << ", " << issue->maxTokens
                       << "] token(s), issued-upper=" << state.issuedTokens
                       << ", inexact=" << state.hasInexactIssueCount);
  return success();
}

static LogicalResult
handleLoadCluster(Operation *anchor, ArrayRef<ttg::LocalLoadOp> loads,
                  SyncState &state, RegionKeyBuilder &keyBuilder,
                  GvmInsertionPlan &plan) {
  uint64_t loadRequiredTokens = state.visibleTokens;
  SmallVector<RegionKey, 4> loadRegions;

  for (ttg::LocalLoadOp load : loads) {
    RegionKey key = keyBuilder.get(load.getSrc());
    if (isLocalAllocation(key) && !isDefinitelyInitialized(state, key))
      return load.emitError()
             << "shared region " << key.label
             << " may be uninitialized on at least one control-flow path";
    loadRegions.push_back(key);
    loadRequiredTokens =
        std::max(loadRequiredTokens, getAliasingCopyEnd(key, state));
  }
  if (failed(requireSharedAccessBarrierBefore(
          anchor, loadRegions, /*checkReaders=*/false,
          /*checkSynchronousWrites=*/true, state, plan)))
    return failure();
  uint64_t requiredTokens = loadRequiredTokens;
  if (requiredTokens > state.visibleTokens) {
    if (requiredTokens > state.issuedTokens)
      return anchor->emitError("local_load requires a future gvm token");

    LDBG("local_load cluster size=" << loads.size()
                                    << " load-required=" << loadRequiredTokens
                                    << " required=" << requiredTokens
                                    << " issued=" << state.issuedTokens
                                    << " waited=" << state.waitedTokens
                                    << " visible=" << state.visibleTokens);
    if (failed(requireVisibleBefore(anchor, requiredTokens,
                                    WaitReason::LoadUse, state, plan)))
      return failure();
  }
  for (const RegionKey &region : loadRegions)
    appendUniqueRegion(state.activeReaders, region);
  return success();
}

static bool canSkipInLoadCluster(Operation *op) {
  return c500::getMemDescViewSource(op) && op->getNumResults() == 1;
}

static bool hasSeenAllocation(ArrayRef<RegionKey> regions,
                              const RegionKey &candidate) {
  return candidate.valid && llvm::any_of(regions, [&](const RegionKey &region) {
           return haveSamePhysicalAllocationIdentity(region, candidate);
         });
}

static bool haveSameRegionIdentity(const RegionKey &lhs,
                                   const RegionKey &rhs) {
  if (lhs.valid != rhs.valid)
    return false;
  return !lhs.valid || isSameRegion(lhs, rhs);
}

static bool haveSameRegionSet(ArrayRef<RegionKey> lhs,
                              ArrayRef<RegionKey> rhs) {
  if (lhs.size() != rhs.size())
    return false;
  return llvm::all_of(lhs, [&](const RegionKey &lhsRegion) {
    return llvm::any_of(rhs, [&](const RegionKey &rhsRegion) {
      return haveSameRegionIdentity(lhsRegion, rhsRegion);
    });
  });
}

static bool haveSameCopyTokenState(const SyncState &lhs, const SyncState &rhs,
                                   bool relativeToIssueEpoch) {
  if (lhs.liveCopies.size() != rhs.liveCopies.size())
    return false;
  for (const LiveCopyRegion &lhsCopy : lhs.liveCopies) {
    auto found = llvm::find_if(rhs.liveCopies, [&](const LiveCopyRegion &copy) {
      return haveSameRegionIdentity(lhsCopy.region, copy.region);
    });
    if (found == rhs.liveCopies.end())
      return false;
    uint64_t lhsToken = lhsCopy.tokenEnd;
    uint64_t rhsToken = found->tokenEnd;
    if (relativeToIssueEpoch) {
      if (lhsToken > lhs.issuedTokens || rhsToken > rhs.issuedTokens)
        return false;
      lhsToken = lhs.issuedTokens - lhsToken;
      rhsToken = rhs.issuedTokens - rhsToken;
    }
    if (lhsToken != rhsToken)
      return false;
  }
  return true;
}

/// Compare only the issue/outstanding-count domain. Visibility and
/// shared-region facts have their own finite fixed point and must not make a
/// numerically exact outstanding suffix inexact.
static bool haveSameQueueCountState(const SyncState &lhs,
                                    const SyncState &rhs,
                                    bool relativeToIssueEpoch) {
  auto compareToken = [&](uint64_t lhsToken, uint64_t rhsToken) {
    if (!relativeToIssueEpoch)
      return lhsToken == rhsToken;
    if (lhsToken > lhs.issuedTokens || rhsToken > rhs.issuedTokens)
      return false;
    return lhs.issuedTokens - lhsToken == rhs.issuedTokens - rhsToken;
  };
  if (!compareToken(lhs.waitedTokens, rhs.waitedTokens) ||
      lhs.hasInexactIssueCount != rhs.hasInexactIssueCount)
    return false;
  if (!relativeToIssueEpoch && lhs.issuedTokens != rhs.issuedTokens)
    return false;
  return true;
}

static bool haveSameVisibilityState(const SyncState &lhs,
                                    const SyncState &rhs,
                                    bool relativeToIssueEpoch) {
  if (lhs.hasUnknownVisibilityLag != rhs.hasUnknownVisibilityLag)
    return false;
  if (lhs.hasUnknownVisibilityLag)
    return true;
  if (!relativeToIssueEpoch)
    return lhs.visibleTokens == rhs.visibleTokens;
  if (lhs.visibleTokens > lhs.issuedTokens ||
      rhs.visibleTokens > rhs.issuedTokens)
    return false;
  return lhs.issuedTokens - lhs.visibleTokens ==
         rhs.issuedTokens - rhs.visibleTokens;
}

static bool haveSameLoopState(const SyncState &lhs, const SyncState &rhs,
                              bool relativeToIssueEpoch) {
  return haveSameQueueCountState(lhs, rhs, relativeToIssueEpoch) &&
         haveSameVisibilityState(lhs, rhs, relativeToIssueEpoch) &&
         haveSameCopyTokenState(lhs, rhs, relativeToIssueEpoch) &&
         haveSameRegionSet(lhs.activeReaders, rhs.activeReaders) &&
         haveSameRegionSet(lhs.pendingSynchronousWrites,
                           rhs.pendingSynchronousWrites) &&
         haveSameRegionSet(lhs.mayInitialized, rhs.mayInitialized) &&
         haveSameRegionSet(lhs.mustInitialized, rhs.mustInitialized);
}

static std::string summarizeQueueState(const SyncState &state) {
  std::string summary;
  llvm::raw_string_ostream os(summary);
  os << "issued=" << state.issuedTokens
     << ", outstanding=" << state.issuedTokens - state.waitedTokens
     << ", visibility-lag=" << state.issuedTokens - state.visibleTokens
     << ", live-regions=" << state.liveCopies.size()
     << ", active-readers=" << state.activeReaders.size()
     << ", pending-sync-writes="
     << state.pendingSynchronousWrites.size()
     << ", may-initialized=" << state.mayInitialized.size()
     << ", must-initialized=" << state.mustInitialized.size()
     << ", inexact-issue-count=" << state.hasInexactIssueCount
     << ", unknown-visibility-lag=" << state.hasUnknownVisibilityLag;
  return os.str();
}

static LogicalResult processBlock(Block &block, SyncState &state,
                                  RegionKeyBuilder &keyBuilder,
                                  GvmInsertionPlan &plan);

static LogicalResult processRegion(Region &region, SyncState &state,
                                   RegionKeyBuilder &keyBuilder,
                                   GvmInsertionPlan &plan) {
  if (!region.hasOneBlock())
    return region.getParentOp()->emitError()
           << "C500 async queue analysis currently requires single-block "
              "structured regions";
  for (Block &block : region) {
    if (failed(processBlock(block, state, keyBuilder, plan)))
      return failure();
  }
  return success();
}

static SyncState joinBranchExitStates(ArrayRef<SyncState> exits);

/// Collapse an unbounded loop-carried queue suffix to a single abstract token.
/// `hasInexactIssueCount` makes every later completion obligation drain the
/// real queue, so retaining concrete epoch growth is unnecessary. Region and
/// initialization sets remain finite and continue to a normal fixed point.
static void widenLoopQueueState(SyncState &state) {
  bool hasPendingQueue = state.hasInexactIssueCount ||
                         state.issuedTokens != state.waitedTokens ||
                         state.issuedTokens != state.visibleTokens ||
                         !state.liveCopies.empty();
  state.issuedTokens = hasPendingQueue ? 1 : 0;
  state.waitedTokens = 0;
  state.visibleTokens = 0;
  for (LiveCopyRegion &copy : state.liveCopies)
    copy.tokenEnd = 1;
  state.hasInexactIssueCount = hasPendingQueue;
  state.hasUnknownVisibilityLag |= hasPendingQueue;
  resetBarrierAnchor(state);
}

/// Widen an independently growing relative visibility lag without losing an
/// exact outstanding-token suffix. The zero epoch is a safe lower bound; any
/// later shared visibility obligation inserts a barrier and clears this top
/// element through `applySharedBarrier`.
static void widenLoopVisibilityState(SyncState &state) {
  state.visibleTokens = 0;
  state.hasUnknownVisibilityLag = true;
  resetBarrierAnchor(state);
}

/// Analyze the two-region control-flow graph of `scf.while`:
///
///   parent -> before --false--> parent
///                       |
///                      true
///                       v
///                     after -> before
///
/// The header state joins the parent entry with the `after` backedge. The
/// parent exit is the converged `before` transfer, which preserves both the
/// zero-body path and exits after any number of iterations.
static LogicalResult processWhileOp(scf::WhileOp whileOp, SyncState &state,
                                    RegionKeyBuilder &keyBuilder,
                                    GvmInsertionPlan &plan) {
  if (!whileOp.getBefore().hasOneBlock() ||
      !whileOp.getAfter().hasOneBlock())
    return whileOp.emitError()
           << "C500 async queue analysis currently requires single-block "
              "scf.while regions";

  const bool outerCollectiveSafe = state.collectiveSafe;
  const std::optional<bool> constantCondition =
      getConstantMaskValue(whileOp.getConditionOp().getCondition());
  const bool loopCollectiveSafe =
      outerCollectiveSafe &&
      (constantCondition.has_value() || isCTAUniformWhile(whileOp));

  SyncState preheader = state;
  resetBarrierAnchor(preheader);
  SyncState preheaderHeader = preheader;
  projectCarrierInitialization(whileOp.getOperation(),
                               /*predecessorRegion=*/nullptr,
                               &whileOp.getBefore(), preheaderHeader,
                               keyBuilder);

  // The `before` region always executes once. A statically false condition
  // has neither an `after` edge nor a recurrence.
  if (constantCondition && !*constantCondition) {
    SyncState exit = preheaderHeader;
    // No thread can reach `after` or the backedge, so the single execution of
    // `before` only needs the enclosing region's collective guarantee.
    exit.collectiveSafe = outerCollectiveSafe;
    if (failed(processRegion(whileOp.getBefore(), exit, keyBuilder, plan)))
      return failure();
    projectCarrierInitialization(whileOp.getOperation(), &whileOp.getBefore(),
                                 /*successorRegion=*/nullptr, exit,
                                 keyBuilder);
    exit.collectiveSafe = outerCollectiveSafe;
    resetBarrierAnchor(exit);
    state = std::move(exit);
    LDBG("[gvm-region-join] constant-false scf.while transfer, state={"
         << summarizeQueueState(state) << "}");
    return success();
  }

  SyncState header = preheaderHeader;
  unsigned iteration = 0;
  SmallVector<SyncState, 8> visitedHeaders;

  while (true) {
    if (llvm::any_of(visitedHeaders, [&](const SyncState &visited) {
          return haveSameLoopState(visited, header,
                                   /*relativeToIssueEpoch=*/false);
        }))
      return whileOp.emitError()
             << "C500 async queue scf.while transfer is non-monotone after "
             << iteration << " iteration(s); insert an explicit uniform "
                             "queue drain/barrier at the loop boundary";
    visitedHeaders.push_back(header);

    SyncState beforeExit = header;
    beforeExit.collectiveSafe = loopCollectiveSafe;
    if (failed(
            processRegion(whileOp.getBefore(), beforeExit, keyBuilder, plan)))
      return failure();

    SyncState parentExit = beforeExit;
    projectCarrierInitialization(whileOp.getOperation(), &whileOp.getBefore(),
                                 /*successorRegion=*/nullptr, parentExit,
                                 keyBuilder);

    SyncState backedge = beforeExit;
    projectCarrierInitialization(whileOp.getOperation(), &whileOp.getBefore(),
                                 &whileOp.getAfter(), backedge, keyBuilder);
    if (failed(
            processRegion(whileOp.getAfter(), backedge, keyBuilder, plan)))
      return failure();
    projectCarrierInitialization(whileOp.getOperation(), &whileOp.getAfter(),
                                 &whileOp.getBefore(), backedge, keyBuilder);

    SmallVector<SyncState, 2> incoming{preheaderHeader, backedge};
    SyncState nextHeader = joinBranchExitStates(incoming);
    nextHeader.collectiveSafe = loopCollectiveSafe;
    bool queueCountFixedPoint = haveSameQueueCountState(
        header, nextHeader, /*relativeToIssueEpoch=*/true);
    bool visibilityFixedPoint = haveSameVisibilityState(
        header, nextHeader, /*relativeToIssueEpoch=*/true);
    if (!queueCountFixedPoint)
      widenLoopQueueState(nextHeader);
    else if (!visibilityFixedPoint)
      widenLoopVisibilityState(nextHeader);

    ++iteration;
    bool fullFixedPoint = haveSameLoopState(
        header, nextHeader, /*relativeToIssueEpoch=*/true);
    if (fullFixedPoint ||
        haveSameLoopState(header, nextHeader,
                          /*relativeToIssueEpoch=*/false)) {
      // Unknown conditions may exit from any converged header visit. A
      // constant-true loop has no semantic parent edge, but retaining the
      // converged transfer is conservative for any structurally following IR.
      state = std::move(parentExit);
      state.collectiveSafe = outerCollectiveSafe;
      resetBarrierAnchor(state);
      LDBG("[gvm-region-join] scf.while fixed point after "
           << iteration << " iteration(s), state={"
           << summarizeQueueState(state) << "}");
      return success();
    }
    header = std::move(nextHeader);
  }
}

static LogicalResult processLoopLikeOp(LoopLikeOpInterface loop,
                                       SyncState &state,
                                       RegionKeyBuilder &keyBuilder,
                                       GvmInsertionPlan &plan) {
  if (auto whileOp = dyn_cast<scf::WhileOp>(loop.getOperation()))
    return processWhileOp(whileOp, state, keyBuilder, plan);

  SmallVector<Region *> loopRegions = loop.getLoopRegions();
  if (loopRegions.size() != 1)
    return loop.emitError()
           << "C500 async queue analysis requires a loop with exactly one "
              "loop region";

  const bool outerCollectiveSafe = state.collectiveSafe;
  const bool bodyCollectiveSafe =
      outerCollectiveSafe && isCTAUniformLoop(loop);
  const bool hasZeroTripExit = loopMayExecuteZeroTimes(loop);
  if (!loopRegions.front()->hasOneBlock())
    return loop.emitError()
           << "C500 async queue analysis currently requires a single-block "
              "loop region";
  SyncState preheader = state;
  resetBarrierAnchor(preheader);
  Region *body = loopRegions.front();
  SyncState preheaderHeader = preheader;
  projectCarrierInitialization(loop.getOperation(),
                               /*predecessorRegion=*/nullptr, body,
                               preheaderHeader, keyBuilder);

  // A proven single-trip loop has no backedge recurrence. Applying an
  // unbounded-loop widening here would manufacture an inexact queue state
  // from the (unreachable) second iteration and can reject a legal nonzero
  // manual GVM suffix.
  if (loopExecutesExactlyOnce(loop)) {
    SyncState exit = preheaderHeader;
    exit.collectiveSafe = bodyCollectiveSafe;
    if (failed(processRegion(*body, exit, keyBuilder, plan)))
      return failure();
    projectCarrierInitialization(loop.getOperation(), body,
                                 /*successorRegion=*/nullptr, exit, keyBuilder);
    state = std::move(exit);
    state.collectiveSafe = outerCollectiveSafe;
    resetBarrierAnchor(state);
    LDBG("[gvm-region-join] exact-one loop transfer, state={"
         << summarizeQueueState(state) << "}");
    return success();
  }

  SyncState header = preheaderHeader;
  unsigned iteration = 0;
  SmallVector<SyncState, 8> visitedHeaders;

  // This is a finite monotone iteration. Only a change in the relative queue
  // count domain triggers widening; live-copy identities and shared-region
  // facts continue independently to their union/intersection fixed point.
  while (true) {
    if (llvm::any_of(visitedHeaders, [&](const SyncState &visited) {
          return haveSameLoopState(visited, header,
                                   /*relativeToIssueEpoch=*/false);
        }))
      return loop.emitError()
             << "C500 async queue loop transfer is non-monotone after "
             << iteration << " iteration(s); insert an explicit uniform "
                             "queue drain/barrier at the loop boundary";
    visitedHeaders.push_back(header);

    SyncState backedge = header;
    backedge.collectiveSafe = bodyCollectiveSafe;
    if (failed(processRegion(*body, backedge, keyBuilder, plan)))
      return failure();

    SyncState backedgeHeader = backedge;
    projectCarrierInitialization(loop.getOperation(), body, body,
                                 backedgeHeader, keyBuilder);
    SmallVector<SyncState, 2> incoming{preheaderHeader, backedgeHeader};
    SyncState nextHeader = joinBranchExitStates(incoming);
    nextHeader.collectiveSafe = bodyCollectiveSafe;
    bool queueCountFixedPoint = haveSameQueueCountState(
        header, nextHeader, /*relativeToIssueEpoch=*/true);
    bool visibilityFixedPoint = haveSameVisibilityState(
        header, nextHeader, /*relativeToIssueEpoch=*/true);
    if (!queueCountFixedPoint)
      widenLoopQueueState(nextHeader);
    else if (!visibilityFixedPoint)
      widenLoopVisibilityState(nextHeader);

    ++iteration;
    bool fullFixedPoint = haveSameLoopState(
        header, nextHeader, /*relativeToIssueEpoch=*/true);
    if (fullFixedPoint ||
        haveSameLoopState(header, nextHeader,
                          /*relativeToIssueEpoch=*/false)) {
      // The parent successor receives the join of the zero-trip preheader edge
      // and every loop-backedge exit. If constant bounds prove at least one
      // iteration, only the backedge transfer reaches the parent successor.
      SyncState bodyExit = backedge;
      projectCarrierInitialization(loop.getOperation(), body,
                                   /*successorRegion=*/nullptr, bodyExit,
                                   keyBuilder);
      if (hasZeroTripExit) {
        SyncState zeroTripExit = preheader;
        projectLoopZeroTripInitialization(loop, zeroTripExit, keyBuilder);
        SmallVector<SyncState, 2> parentExits{zeroTripExit, bodyExit};
        state = joinBranchExitStates(parentExits);
      } else {
        state = std::move(bodyExit);
      }
      state.collectiveSafe = outerCollectiveSafe;
      resetBarrierAnchor(state);
      LDBG("[gvm-region-join] loop fixed point after "
           << iteration << " iteration(s), state={"
           << summarizeQueueState(state) << "}");
      return success();
    }
    header = std::move(nextHeader);
  }
}

static uint64_t getOutstandingTokens(const SyncState &state) {
  return state.issuedTokens - state.waitedTokens;
}

static uint64_t getVisibilityLag(const SyncState &state) {
  return state.issuedTokens - state.visibleTokens;
}

static void appendStateRegions(SmallVectorImpl<RegionKey> &destination,
                               ArrayRef<RegionKey> source) {
  for (const RegionKey &region : source)
    appendUniqueRegion(destination, region);
}

static SmallVector<RegionKey, 8>
intersectStateRegions(ArrayRef<SyncState> states,
                      SmallVector<RegionKey, 8> SyncState::*member) {
  assert(!states.empty());
  SmallVector<RegionKey, 8> intersection;
  for (const RegionKey &candidate : states.front().*member) {
    if (llvm::all_of(states.drop_front(), [&](const SyncState &state) {
          return llvm::any_of(state.*member, [&](const RegionKey &region) {
            return haveSameRegionIdentity(candidate, region);
          });
        }))
      appendUniqueRegion(intersection, candidate);
  }
  return intersection;
}

/// Join RegionBranch exits in a relative-to-queue-tail domain. Absolute issue
/// epochs are path-local and are therefore never compared directly. A
/// non-uniform outstanding suffix is represented as an inexact issue count;
/// the first synchronization obligation after reconvergence drains it.
static SyncState joinBranchExitStates(ArrayRef<SyncState> exits) {
  assert(!exits.empty());
  uint64_t joinedIssueEpoch = 0;
  for (const SyncState &exit : exits)
    joinedIssueEpoch = std::max(joinedIssueEpoch, exit.issuedTokens);

  const uint64_t referenceOutstanding = getOutstandingTokens(exits.front());
  bool uniformOutstanding = true;
  bool hadInexactIssueCount = false;
  bool hadUnknownVisibilityLag = false;
  bool hadInvalidCopyEpoch = false;
  uint64_t maxOutstanding = 0;
  uint64_t maxVisibilityLag = 0;
  SyncState joined;
  joined.interpretInferredSynchronization =
      exits.front().interpretInferredSynchronization;
  joined.issuedTokens = joinedIssueEpoch;
  for (const SyncState &exit : exits) {
    uint64_t outstanding = getOutstandingTokens(exit);
    uint64_t visibilityLag = getVisibilityLag(exit);
    uniformOutstanding &= outstanding == referenceOutstanding;
    maxOutstanding = std::max(maxOutstanding, outstanding);
    maxVisibilityLag = std::max(maxVisibilityLag, visibilityLag);
    hadInexactIssueCount |= exit.hasInexactIssueCount;
    hadUnknownVisibilityLag |= exit.hasUnknownVisibilityLag;
    appendStateRegions(joined.activeReaders, exit.activeReaders);
    appendStateRegions(joined.pendingSynchronousWrites,
                       exit.pendingSynchronousWrites);
    appendStateRegions(joined.mayInitialized, exit.mayInitialized);
    for (const LiveCopyRegion &copy : exit.liveCopies) {
      uint64_t translatedTokenEnd = joinedIssueEpoch;
      if (copy.tokenEnd <= exit.issuedTokens) {
        uint64_t relativeAge = exit.issuedTokens - copy.tokenEnd;
        translatedTokenEnd = joinedIssueEpoch - relativeAge;
      } else {
        // An invalid epoch cannot be represented exactly. Keep the region at
        // the joined tail and force a later drain rather than dropping it.
        hadInvalidCopyEpoch = true;
      }
      auto found = llvm::find_if(joined.liveCopies, [&](const auto &candidate) {
        return haveSameRegionIdentity(copy.region, candidate.region);
      });
      if (found == joined.liveCopies.end())
        joined.liveCopies.push_back({copy.region, translatedTokenEnd});
      else
        found->tokenEnd = std::max(found->tokenEnd, translatedTokenEnd);
    }
  }
  joined.mustInitialized =
      intersectStateRegions(exits, &SyncState::mustInitialized);
  joined.waitedTokens = joinedIssueEpoch - maxOutstanding;
  joined.visibleTokens = hadUnknownVisibilityLag
                             ? 0
                             : joinedIssueEpoch - maxVisibilityLag;
  joined.hasInexactIssueCount =
      hadInexactIssueCount || !uniformOutstanding || hadInvalidCopyEpoch;
  joined.hasUnknownVisibilityLag = hadUnknownVisibilityLag;
  resetBarrierAnchor(joined);
  pruneVisibleCopies(joined);
  return joined;
}

static LogicalResult processBranchRegions(RegionBranchOpInterface branch,
                                          SyncState &state,
                                          RegionKeyBuilder &keyBuilder,
                                          GvmInsertionPlan &plan) {
  Operation *op = branch.getOperation();

  SmallVector<RegionSuccessor> entrySuccessors;
  branch.getSuccessorRegions(RegionBranchPoint::parent(), entrySuccessors);
  SmallVector<Region *> entryRegions;
  bool hasParentBypass = false;
  for (const RegionSuccessor &successor : entrySuccessors) {
    if (successor.isParent()) {
      hasParentBypass = true;
      continue;
    }
    Region *region = successor.getSuccessor();
    if (region && !llvm::is_contained(entryRegions, region))
      entryRegions.push_back(region);
  }

  bool hasKnownExecutableSuccessor = false;
  if (auto ifOp = dyn_cast<scf::IfOp>(op)) {
    if (std::optional<bool> condition =
            getConstantMaskValue(ifOp.getCondition())) {
      hasKnownExecutableSuccessor = true;
      entryRegions.clear();
      hasParentBypass = false;
      if (*condition) {
        entryRegions.push_back(&ifOp.getThenRegion());
      } else if (!ifOp.getElseRegion().empty()) {
        entryRegions.push_back(&ifOp.getElseRegion());
      } else {
        hasParentBypass = true;
      }
    }
  }

  for (Region &region : op->getRegions())
    if (!hasKnownExecutableSuccessor &&
        !llvm::is_contained(entryRegions, &region) &&
        hasNestedTargetOp(region, keyBuilder))
      return op->emitError()
             << "C500 async queue analysis found target operations in a "
                "RegionBranch region that is not a direct entry successor";

  for (Region *region : entryRegions) {
    SmallVector<RegionSuccessor> successors;
    branch.getSuccessorRegions(*region, successors);
    if (llvm::any_of(successors, [](const RegionSuccessor &successor) {
          return !successor.isParent();
        }))
      return op->emitError()
             << "C500 async queue analysis supports non-loop RegionBranch "
                "operations only when each entry region returns directly to "
                "the parent";
  }
  if (entryRegions.empty() && !hasParentBypass)
    return op->emitError()
           << "C500 async queue analysis found no executable RegionBranch "
              "successor";

  SmallVector<SyncState, 4> exitStates;
  exitStates.reserve(entryRegions.size() + hasParentBypass);
  if (hasParentBypass) {
    SyncState bypassState = state;
    projectCarrierInitialization(op, /*predecessorRegion=*/nullptr,
                                 /*successorRegion=*/nullptr, bypassState,
                                 keyBuilder);
    bypassState.lastBarrier = nullptr;
    bypassState.lastBarrierIssuedTokens = 0;
    bypassState.lastBarrierVisibleTokens = bypassState.visibleTokens;
    bypassState.lastBarrierManualGvm = {};
    bypassState.pendingManualGvm = {};
    exitStates.push_back(std::move(bypassState));
  }
  const bool branchCollectiveSafe =
      state.collectiveSafe && isCTAUniformBranch(branch);
  for (Region *region : entryRegions) {
    SyncState branchState = state;
    projectCarrierInitialization(op, /*predecessorRegion=*/nullptr, region,
                                 branchState, keyBuilder);
    branchState.collectiveSafe = branchCollectiveSafe;
    if (failed(processRegion(*region, branchState, keyBuilder, plan)))
      return failure();
    projectCarrierInitialization(op, region, /*successorRegion=*/nullptr,
                                 branchState, keyBuilder);
    // Branch-local barrier anchors do not dominate operations after the join.
    branchState.lastBarrier = nullptr;
    branchState.lastBarrierIssuedTokens = 0;
    branchState.lastBarrierVisibleTokens = branchState.visibleTokens;
    branchState.lastBarrierManualGvm = {};
    branchState.pendingManualGvm = {};
    exitStates.push_back(std::move(branchState));
  }
  if (exitStates.empty())
    return success();

  bool outerCollectiveSafe = state.collectiveSafe;
  state = joinBranchExitStates(exitStates);
  state.collectiveSafe = outerCollectiveSafe;
  LDBG("[gvm-region-join] relative branch state={"
       << summarizeQueueState(state) << "}");
  return success();
}

static LogicalResult processRegionBranchOp(RegionBranchOpInterface branch,
                                           SyncState &state,
                                           RegionKeyBuilder &keyBuilder,
                                           GvmInsertionPlan &plan) {
  Operation *op = branch.getOperation();
  if (!hasNestedTargetOp(op, keyBuilder) && !keyBuilder.hasScratch(op))
    return success();

  if (auto loop = dyn_cast<LoopLikeOpInterface>(op))
    return processLoopLikeOp(loop, state, keyBuilder, plan);
  return processBranchRegions(branch, state, keyBuilder, plan);
}

static LogicalResult processBlock(Block &block, SyncState &state,
                                  RegionKeyBuilder &keyBuilder,
                                  GvmInsertionPlan &plan) {
  for (auto it = block.begin(); it != block.end();) {
    Operation *op = &*it;

    if (auto gvm = dyn_cast<ttg::GVMArriveOp>(op)) {
      if (gvm->hasAttr(kInferredGvmAttr) &&
          !state.interpretInferredSynchronization) {
        ++it;
        continue;
      }
      if (failed(handleGvm(gvm, state)))
        return failure();
      ++it;
      continue;
    }

    if (isa<ttg::BarrierSharedOp, ttg::BarrierOp>(op)) {
      if (op->hasAttr(kInferredBarrierAttr) &&
          !state.interpretInferredSynchronization) {
        ++it;
        continue;
      }
      if (!state.collectiveSafe)
        return op->emitError()
               << "CTA-wide shared barrier is nested in control flow that "
                  "is not proven CTA-uniform";
      handleUserBarrier(op, state);
      ++it;
      continue;
    }

    if (auto regionBranch = dyn_cast<RegionBranchOpInterface>(op)) {
      std::optional<RegionKey> scratch = keyBuilder.getScratch(op);
      if (scratch &&
          failed(handleScratchBefore(op, *scratch, state, plan)))
        return failure();
      if (failed(processRegionBranchOp(regionBranch, state, keyBuilder, plan)))
        return failure();
      if (scratch)
        handleScratchAfter(op, *scratch, state);
      ++it;
      continue;
    }

    if (std::optional<RegionKey> scratch = keyBuilder.getScratch(op)) {
      if (failed(handleScratchBefore(op, *scratch, state, plan)))
        return failure();
      // Atomics contribute to the GVM FIFO before their optional result
      // broadcast scratch completes. Keep that order explicit so a following
      // scratch reuse sees both the queue and shared-memory obligations.
      if (isa<tt::AtomicRMWOp, tt::AtomicCASOp>(op) &&
          failed(handleAtomicIssue(op, state)))
        return failure();
      handleScratchAfter(op, *scratch, state);
      ++it;
      continue;
    }

    if (op->getNumRegions() != 0 && hasNestedTargetOp(op, keyBuilder)) {
      return op->emitError()
             << "C500 async operations nested in control flow require "
                "RegionBranchOpInterface";
    }

    if (auto copy = dyn_cast<ttg::AsyncCopyGlobalToLocalOp>(op)) {
      if (failed(handleCopy(copy, state, keyBuilder, plan)))
        return failure();
      ++it;
      continue;
    }

    if (isa<tt::AtomicRMWOp, tt::AtomicCASOp>(op)) {
      if (failed(handleAtomicIssue(op, state)))
        return failure();
      ++it;
      continue;
    }

    if (auto store = dyn_cast<ttg::LocalStoreOp>(op)) {
      if (failed(handleSynchronousSharedWrite(
              store, store.getDst(), state, keyBuilder, plan)))
        return failure();
      ++it;
      continue;
    }

    if (auto alloc = dyn_cast<ttg::LocalAllocOp>(op)) {
      RegionKey key = keyBuilder.get(alloc.getResult());
      if (alloc.getSrc()) {
        if (failed(handleSynchronousSharedWrite(
                alloc, alloc.getResult(), state, keyBuilder, plan)))
          return failure();
      } else {
        markUninitialized(state, key);
      }
      ++it;
      continue;
    }

    if (auto load = dyn_cast<ttg::LocalLoadOp>(op)) {
      SmallVector<ttg::LocalLoadOp> loads;
      SmallVector<RegionKey> seenRegions;
      auto clusterEnd = it;
      while (clusterEnd != block.end()) {
        Operation *clusterOp = &*clusterEnd;
        if (auto clusterLoad = dyn_cast<ttg::LocalLoadOp>(clusterOp)) {
          RegionKey key = keyBuilder.get(clusterLoad.getSrc());
          if (hasSeenAllocation(seenRegions, key))
            break;
          loads.push_back(clusterLoad);
          if (key.valid)
            seenRegions.push_back(std::move(key));
          ++clusterEnd;
          continue;
        }
        if (!canSkipInLoadCluster(clusterOp))
          break;
        ++clusterEnd;
      }

      if (failed(handleLoadCluster(load, loads, state, keyBuilder, plan)))
        return failure();
      it = clusterEnd;
      continue;
    }

    if (isa<tt::StoreOp, tt::ReturnOp>(op)) {
      if (failed(
              insertDrainBefore(op, WaitReason::BoundaryDrain, state, plan)))
        return failure();
    }
    ++it;
  }
  return success();
}

class TritonMETAXGPUGluonInsertGvmArriveBarrierSharedPass
    : public TritonMETAXGPUGluonInsertGvmArriveBarrierSharedBase<
          TritonMETAXGPUGluonInsertGvmArriveBarrierSharedPass> {
public:
  void runOnOperation() override {
    if (failed(insertTritonMETAXGPUGluonGvmArriveBarrierShared(
            getOperation())))
      signalPassFailure();
  }
};

} // namespace

static LogicalResult analyzeTritonMETAXGPUGluonSynchronization(
    ModuleOp module, GvmInsertionPlan &plan,
    bool interpretInferredSynchronization);

LogicalResult mlir::insertTritonMETAXGPUGluonGvmArriveBarrierShared(
    ModuleOp module) {
  auto plannedModule = cast<ModuleOp>(module->clone());
  SmallVector<Operation *> oldInferredSynchronization;
  plannedModule.walk([&](Operation *op) {
    if ((isa<ttg::GVMArriveOp>(op) && op->hasAttr(kInferredGvmAttr)) ||
        (isa<ttg::BarrierSharedOp>(op) &&
         op->hasAttr(kInferredBarrierAttr)))
      oldInferredSynchronization.push_back(op);
  });
  for (Operation *op : llvm::reverse(oldInferredSynchronization))
    op->erase();

  GvmInsertionPlan plan;
  if (failed(analyzeTritonMETAXGPUGluonSynchronization(
          plannedModule, plan,
          /*interpretInferredSynchronization=*/false))) {
    plannedModule.erase();
    return failure();
  }
  plan.materialize();

  // Replay the immutable materialized plan as concrete hardware events before
  // committing it. This catches interactions that a fixed-point requirement
  // merge cannot see locally, such as a stronger backpatched inferred wait
  // followed by a user-authored suffix that would move the FIFO backwards.
  if (failed(verifyTritonMETAXGPUGluonSynchronization(plannedModule))) {
    plannedModule.erase();
    return failure();
  }

  module->setAttrs(plannedModule->getAttrs());
  module.getBodyRegion().takeBody(plannedModule.getBodyRegion());
  plannedModule.erase();
  return success();
}

static LogicalResult verifySynchronizationMarkerOwnership(ModuleOp module) {
  WalkResult result = module.walk([&](Operation *op) {
    if (op->hasAttr(kInferredGvmAttr) && !isa<ttg::GVMArriveOp>(op)) {
      op->emitError() << kInferredGvmAttr
                      << " is reserved for compiler-owned ttg.gvm_arrive";
      return WalkResult::interrupt();
    }
    if (op->hasAttr(kInferredBarrierAttr) &&
        !isa<ttg::BarrierSharedOp>(op)) {
      op->emitError() << kInferredBarrierAttr
                      << " is reserved for compiler-owned "
                         "ttg.barrier_shared";
      return WalkResult::interrupt();
    }
    return WalkResult::advance();
  });
  return result.wasInterrupted() ? failure() : success();
}

/// A manual GVM wait makes synchronization ownership explicit for the whole
/// function. Mixing user scheduling with automatic repair is ambiguous: an
/// inserted wait can shorten the intended overlap window, while interpreting
/// a manual threshold requires target-specific queue invariants that are not
/// represented in generic SSA dependencies. Keep automatic and explicit modes
/// disjoint and deterministic.
static bool usesExplicitGvmScheduling(tt::FuncOp func) {
  WalkResult result = func.walk([&](ttg::GVMArriveOp gvm) {
    return gvm->hasAttr(kInferredGvmAttr) ? WalkResult::advance()
                                           : WalkResult::interrupt();
  });
  return result.wasInterrupted();
}

/// Verify the hard C500 shared-memory capacity and report the residency bound
/// from Triton's final per-kernel Allocation. This covers compiler-managed
/// explicit buffers and layout-conversion scratch; it must not be described
/// as a general static-plus-dynamic ELF accounting API. Residency is an upper
/// bound from these shared bytes only, not an occupancy claim: registers,
/// architectural block limits, and post-RA resource usage may reduce it
/// further. C500 exposes 64 KiB per execution partition, so the canonical
/// 1/2/4-CTA cliffs are 64/32/16 KiB.
static LogicalResult
verifyAndLogFinalSharedMemory(tt::FuncOp func,
                              const Allocation &allocation) {
  constexpr size_t kSharedCapacityBytes = 64 * 1024;
  constexpr unsigned kMaxBlocksPerPartition = 16;
  constexpr std::array<size_t, 3> kCanonicalCliffs{
      kSharedCapacityBytes, kSharedCapacityBytes / 2,
      kSharedCapacityBytes / 4};

  size_t sharedBytes = allocation.getSharedMemorySize();
  unsigned sharedLimitedResidentBlocks =
      sharedBytes == 0
          ? kMaxBlocksPerPartition
          : std::min<unsigned>(kMaxBlocksPerPartition,
                               kSharedCapacityBytes / sharedBytes);
  std::optional<size_t> containingCanonicalCliff;
  for (size_t cliff : llvm::reverse(kCanonicalCliffs))
    if (sharedBytes <= cliff) {
      containingCanonicalCliff = cliff;
      break;
    }

  size_t nextResidencyThreshold =
      sharedLimitedResidentBlocks >= kMaxBlocksPerPartition
          ? 0
          : kSharedCapacityBytes / (sharedLimitedResidentBlocks + 1);
  size_t bytesToNextResidency =
      sharedBytes > nextResidencyThreshold
          ? sharedBytes - nextResidencyThreshold
          : 0;
  LDBG("[shared-resource] function=@"
       << func.getSymName()
       << ", final-compiler-managed-shared-bytes=" << sharedBytes
       << ", capacity-bytes=" << kSharedCapacityBytes
       << ", shared-limited-resident-cta-upper-bound="
       << sharedLimitedResidentBlocks
       << ", containing-canonical-cliff-bytes="
       << (containingCanonicalCliff
               ? std::to_string(*containingCanonicalCliff)
               : "over-capacity")
       << ", bytes-to-next-shared-residency=" << bytesToNextResidency
       << ", canonical-cliffs-bytes=[65536,32768,16384]"
       << ", actual-occupancy-requires-post-ra-register-data=true");
  if (sharedBytes > kSharedCapacityBytes) {
    InFlightDiagnostic diagnostic =
        func.emitError()
        << "final compiler-managed shared-memory allocation exceeds the "
           "C500 hard capacity: required "
        << sharedBytes << " bytes, capacity " << kSharedCapacityBytes
        << " bytes; reduce live shared buffers or select a lower-scratch "
           "layout transfer";
    llvm::DenseSet<Allocation::BufferId> reported;
    auto report = [&](Operation *owner, Allocation::BufferId id,
                      StringRef kind) {
      if (id == Allocation::InvalidBufferId || !reported.insert(id).second)
        return;
      diagnostic.attachNote(owner->getLoc())
          << "final shared-memory " << kind << " buffer: id=" << id
          << ", offset=" << allocation.getOffset(id)
          << ", bytes=" << allocation.getAllocatedSize(id)
          << ", owner=" << owner->getName();
      llvm::errs() << "[metax-gluon-shared-capacity] kind=" << kind
                   << ", id=" << id
                   << ", offset=" << allocation.getOffset(id)
                   << ", bytes=" << allocation.getAllocatedSize(id)
                   << ", owner=" << owner->getName()
                   << ", location=" << owner->getLoc() << "\n";
    };
    func.walk([&](Operation *op) {
      report(op, allocation.getBufferId(op), "scratch");
      for (Value result : op->getResults())
        report(op, allocation.getBufferId(result), "explicit");
    });
    return failure();
  }
  return success();
}

static LogicalResult analyzeTritonMETAXGPUGluonSynchronization(
    ModuleOp module, GvmInsertionPlan &plan,
    bool interpretInferredSynchronization) {
  // This pass runs after staging, RDD, instruction scheduling, and final
  // control-flow canonicalization. Build Allocation on exactly that IR so
  // explicit buffers and every backend scratch owner share one physical
  // interval source of truth.
  ModuleAllocation moduleAllocation(module);
  for (tt::FuncOp func : module.getOps<tt::FuncOp>()) {
    auto function = cast<FunctionOpInterface>(func.getOperation());
    Allocation *allocation = moduleAllocation.getFuncData(function);
    if (!allocation)
      return func.emitError(
          "missing final shared-memory Allocation analysis");
    if (failed(verifyAndLogFinalSharedMemory(func, *allocation)))
      return failure();

    if (usesExplicitGvmScheduling(func)) {
      LDBG("[gvm-ownership] skip automatic synchronization planning for @"
           << func.getSymName() << ": function contains a manual gvm_arrive");
      continue;
    }

    gluon_analysis::SharedAccessPathAnalysis accessPaths(func);
    if (failed(accessPaths.initialize()))
      return failure();
    RegionKeyBuilder keyBuilder(*allocation, accessPaths);
    if (!func.getBody().hasOneBlock() &&
        hasNestedTargetOp(func, keyBuilder))
      return func.emitError()
             << "C500 async queue analysis requires structured control flow "
                "before SCF-to-CF lowering";
    SyncState state;
    state.interpretInferredSynchronization =
        interpretInferredSynchronization;
    for (Block &block : func.getBody()) {
      if (failed(processBlock(block, state, keyBuilder, plan)))
        return failure();
    }
  }
  return success();
}

LogicalResult mlir::verifyTritonMETAXGPUGluonSynchronization(ModuleOp module) {
  if (failed(verifySynchronizationMarkerOwnership(module)))
    return failure();
  GvmInsertionPlan repairPlan;
  if (failed(analyzeTritonMETAXGPUGluonSynchronization(
          module, repairPlan,
          /*interpretInferredSynchronization=*/true)))
    return failure();
  return repairPlan.emitUnexpectedRequirement();
}

class TritonMETAXGPUGluonVerifySynchronizationPass
    : public TritonMETAXGPUGluonVerifySynchronizationBase<
          TritonMETAXGPUGluonVerifySynchronizationPass> {
public:
  void runOnOperation() override {
    if (failed(verifyTritonMETAXGPUGluonSynchronization(getOperation())))
      signalPassFailure();
  }
};

std::unique_ptr<Pass>
mlir::createTritonMETAXGPUGluonInsertGvmArriveBarrierSharedPass() {
  return std::make_unique<TritonMETAXGPUGluonInsertGvmArriveBarrierSharedPass>();
}

std::unique_ptr<Pass>
mlir::createTritonMETAXGPUGluonVerifySynchronizationPass() {
  return std::make_unique<TritonMETAXGPUGluonVerifySynchronizationPass>();
}
