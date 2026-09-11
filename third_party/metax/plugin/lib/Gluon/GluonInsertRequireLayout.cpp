#include "Gluon/GluonLayoutCandidate.h"

#include "Gluon/Analysis/GluonDotAnalysis.h"
#include "Gluon/Analysis/GluonMemDescAliasAnalysis.h"
#include "Gluon/Analysis/GluonRegionBranchAnalysis.h"
#include "Gluon/GluonC500AsyncCopyPlan.h"
#include "Gluon/GluonC500LayoutHelpers.h"
#include "Gluon/GluonLayoutPlaceholders.h"
#include "Gluon/Targets/GluonC500Layout.h"
#include "Gluon/Targets/GluonC500LayoutRules.h"
#include "Gluon/Passes.h"

#include "triton/Analysis/AxisInfo.h"
#include "triton/Dialect/Gluon/IR/Dialect.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "mlir/Interfaces/ControlFlowInterfaces.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/MathExtras.h"
#include "llvm/Support/Debug.h"

#include <numeric>

#define DEBUG_TYPE "metax-gluon-insert-require-layout"
#define DBGS() (llvm::dbgs() << "[" DEBUG_TYPE "]: ")
#define LDBG(X) LLVM_DEBUG(DBGS() << X << "\n")

namespace tt = ::mlir::triton;
namespace ttg = ::mlir::triton::gpu;
namespace gd = ::mlir::triton::gluon;
namespace layout = ::mlir::triton::gpu::metax::gluon;

namespace mlir {

#define GEN_PASS_DEF_TRITONMETAXGPUGLUONINSERTREQUIRELAYOUT
#include "Gluon/Passes.h.inc"

namespace {

struct PlannedSeed {
  Value value;
  Attribute encoding;
  Operation *anchor;
  bool replaceUses;
};

struct PlannedRequirement {
  Operation *owner;
  unsigned operandIndex;
  Attribute encoding;
};

struct PlannedBoundarySeed {
  Operation *owner;
  unsigned operandIndex;
  Attribute encoding;
};

struct RequirementPlan {
  SmallVector<PlannedSeed, 8> accumulatorSeeds;
  SmallVector<PlannedBoundarySeed, 8> boundarySeeds;
  SmallVector<PlannedRequirement, 8> tensorRequirements;
  SmallVector<PlannedRequirement, 8> tensorReleases;
  SmallVector<PlannedRequirement, 8> memDescRequirements;
};

static bool hasAutoEncoding(Value value) {
  auto type = dyn_cast<RankedTensorType>(value.getType());
  return type &&
         isa_and_nonnull<gd::AutoEncodingAttr>(
             layout::unwrapNoVerifyEncoding(type.getEncoding()));
}

static LogicalResult recordSeed(RequirementPlan &plan, Value value,
                                Attribute encoding, Operation *anchor,
                                bool replaceUses = false) {
  if (!hasAutoEncoding(value))
    return anchor->emitError()
           << "selected candidate attempted to retag a fixed tensor";
  auto existing = llvm::find_if(
      plan.accumulatorSeeds,
      [&](const PlannedSeed &seed) { return seed.value == value; });
  if (existing == plan.accumulatorSeeds.end()) {
    plan.accumulatorSeeds.push_back(
        {value, encoding, anchor, replaceUses});
    return success();
  }
  if (existing->encoding == encoding) {
    existing->replaceUses |= replaceUses;
    return success();
  }
  return anchor->emitError()
         << "one Auto tensor received conflicting accumulator requirements";
}

static LogicalResult
recordRequirement(SmallVectorImpl<PlannedRequirement> &requirements,
                  Operation *owner, unsigned operandIndex,
                  Attribute encoding) {
  if (!owner || operandIndex >= owner->getNumOperands() || !encoding)
    return failure();
  auto existing = llvm::find_if(
      requirements, [&](const PlannedRequirement &requirement) {
        return requirement.owner == owner &&
               requirement.operandIndex == operandIndex;
      });
  if (existing == requirements.end()) {
    requirements.push_back({owner, operandIndex, encoding});
    return success();
  }
  return success(existing->encoding == encoding);
}

static LogicalResult recordBoundarySeed(RequirementPlan &plan,
                                        Operation *owner,
                                        unsigned operandIndex,
                                        Attribute encoding) {
  if (!owner || operandIndex >= owner->getNumOperands() ||
      !hasAutoEncoding(owner->getOperand(operandIndex)) || !encoding)
    return failure();
  auto existing = llvm::find_if(
      plan.boundarySeeds, [&](const PlannedBoundarySeed &seed) {
        return seed.owner == owner && seed.operandIndex == operandIndex;
      });
  if (existing == plan.boundarySeeds.end()) {
    plan.boundarySeeds.push_back({owner, operandIndex, encoding});
    return success();
  }
  return success(existing->encoding == encoding);
}

static LogicalResult recordRootRequirement(DenseMap<Value, Attribute> &roots,
                                           Value root, Attribute encoding,
                                           Operation *anchor) {
  auto [it, inserted] = roots.try_emplace(root, encoding);
  if (inserted || it->second == encoding)
    return success();
  return anchor->emitError()
         << "one shared allocation root received incompatible dot "
            "requirements: "
         << it->second << " versus " << encoding;
}

static FailureOr<SmallVector<layout::DotPipelineFacts, 4>>
collectAllDotFacts(ModuleOp module,
                   layout::GluonMemDescAliasAnalysis &aliases) {
  FailureOr<SmallVector<layout::DotPipelineFacts, 4>> allFacts =
      layout::collectDotPipelineFacts(module, aliases);
  if (failed(allFacts))
    return failure();
  return allFacts;
}

static LogicalResult
planDotRequirements(ModuleOp module,
                    ArrayRef<layout::DotPipelineFacts> facts,
                    const layout::LayoutCandidate &candidate,
                    RequirementPlan &plan,
                    DenseMap<Value, Attribute> &sharedRoots) {
  if (facts.size() != candidate.dots.size())
    return module.emitError()
           << "selected candidate contains " << candidate.dots.size()
           << " dot plans, but the module contains " << facts.size()
           << " tunable dots";

  for (auto [dotIndex, pair] :
       llvm::enumerate(llvm::zip(facts, candidate.dots))) {
    const layout::DotPipelineFacts &dotFacts = std::get<0>(pair);
    const layout::DotLayoutPlan &dotPlan = std::get<1>(pair);
    tt::DotOp dot = dotFacts.dot;
    if (!dotPlan.mma || !dotPlan.operandA || !dotPlan.operandB)
      return dot.emitError() << "selected an incomplete C500 dot plan";

    for (unsigned operandIndex : {0u, 1u}) {
      Attribute operandEncoding = dotPlan.getOperand(operandIndex);
      const layout::DotPipelineOperandFacts &operandFacts =
          dotFacts.operands[operandIndex];
      auto &requirements = operandFacts.hasDotProducer
                               ? plan.tensorReleases
                               : plan.tensorRequirements;
      if (failed(recordRequirement(requirements, dot, operandIndex,
                                   operandEncoding)))
        return dot.emitError()
               << "one dot operand received conflicting requirements";

      if (operandFacts.requiresBsmPermutation()) {
        for (Operation *operation : operandFacts.bsmPermutations) {
          auto bsm = dyn_cast<ttg::BsmPermOp>(operation);
          if (!bsm ||
              failed(recordRequirement(plan.tensorRequirements, bsm,
                                       /*src1=*/0, operandEncoding)))
            return dot.emitError()
                   << "cannot materialize the fixed C500 BSM carrier "
                      "contract";
        }
      }
      for (const layout::DotPipelineLocalLoadFacts &loadFacts :
           operandFacts.localLoads) {
        ttg::LocalLoadOp localLoad = loadFacts.localLoad;
        if (!layout::isCompilerManagedSharedFamilyRoot(
                loadFacts.sharedView.root)) {
          auto resultType =
              dyn_cast<RankedTensorType>(localLoad.getType());
          auto sharedType = dyn_cast<ttg::MemDescType>(
              localLoad.getSrc().getType());
          if (!resultType || !sharedType)
            return localLoad.emitError()
                   << "fixed dot local_load requires ranked tensor and "
                      "memdesc types";
          RankedTensorType candidateType =
              resultType.cloneWithEncoding(operandEncoding);
          FailureOr<layout::RegisterToSharedContractInfo> contract =
              layout::checkRegisterToSharedContract(candidateType, sharedType);
          if (failed(contract) || !contract->lowerable)
            return localLoad.emitError()
                   << "selected dot operand is not linearly lowerable from "
                      "the fixed shared view";
          LDBG("[fixed-shared-view] local-load="
               << localLoad.getLoc()
               << ", shared=" << sharedType.getEncoding()
               << ", register=" << operandEncoding);
          continue;
        }

        FailureOr<Attribute> viewEncoding =
            layout::inferC500DotSharedEncoding(
                operandEncoding, localLoad, operandFacts.path);
        if (failed(viewEncoding))
          return failure();
        FailureOr<Attribute> rootEncoding = layout::projectEncodingToRoot(
            loadFacts.sharedView, *viewEncoding, localLoad);
        if (failed(rootEncoding) ||
            failed(recordRootRequirement(sharedRoots,
                                         loadFacts.sharedView.root,
                                         *rootEncoding,
                                         localLoad)) ||
            failed(recordRequirement(plan.memDescRequirements,
                                     localLoad, /*src=*/0,
                                     *viewEncoding)))
          return failure();
      }
    }

    if (failed(recordSeed(plan, dot.getC(), dotPlan.mma, dot)) ||
        failed(recordSeed(plan, dot.getD(), dotPlan.mma, dot,
                          /*replaceUses=*/true)))
      return failure();

    LDBG("[dot] index=" << dotIndex << ", location=" << dot.getLoc()
                         << ", mma=" << dotPlan.mma);
  }
  return success();
}

static LogicalResult planAsyncCopyRequirements(
    ModuleOp module, layout::GluonMemDescAliasAnalysis &aliases,
    tt::ModuleAxisInfoAnalysis &axisInfo,
    const DenseMap<Value, Attribute> &sharedRoots, RequirementPlan &plan) {
  WalkResult result =
      module.walk([&](ttg::AsyncCopyGlobalToLocalOp copy) {
        FailureOr<layout::MemDescAliasPath> destination =
            aliases.getPath(copy->getOperand(1));
        if (failed(destination)) {
          copy.emitError()
              << "async-copy destination has no proven layout view path";
          return WalkResult::interrupt();
        }
        auto root = sharedRoots.find(destination->root);
        if (root == sharedRoots.end())
          return WalkResult::advance();

        FailureOr<Attribute> sharedEncoding =
            layout::projectEncodingFromRoot(*destination, root->second, copy);
        FailureOr<layout::c500::Rank2ContiguousOrderInfo> access =
            layout::c500::inferRank2ContiguousOrder(copy.getSrc(), &axisInfo,
                                                    copy);
        if (failed(sharedEncoding) || failed(access) ||
            access->order.size() != 2 ||
            access->maxContiguousElements == 0) {
          copy.emitError()
              << "dot-staging async copy lacks a proven local bridge";
          return WalkResult::interrupt();
        }

        layout::c500::AsyncCopyAddressContiguity contiguity{
            access->order.front(), access->maxContiguousElements};
        FailureOr<layout::c500::AsyncCopyIssuePlan> issue =
            layout::c500::inferC500AsyncCopyIssuePlan(
                copy, access->order, contiguity, *sharedEncoding);
        if (failed(issue)) {
          copy.emitError()
              << "cannot construct the selected C500 async-copy bridge";
          return WalkResult::interrupt();
        }

        auto seedOperand = [&](unsigned operandIndex) -> LogicalResult {
          Value operand = copy->getOperand(operandIndex);
          if (!isa<RankedTensorType>(operand.getType()) ||
              !hasAutoEncoding(operand))
            return success();
          return recordBoundarySeed(plan, copy, operandIndex,
                                    issue->sourceEncoding);
        };
        if (failed(seedOperand(/*src=*/0)) ||
            (copy.getMask() && failed(seedOperand(/*mask=*/2))) ||
            (copy.getOther() && failed(seedOperand(/*other=*/3))) ||
            failed(recordRequirement(plan.memDescRequirements, copy,
                                     /*destination=*/1,
                                     issue->sharedEncoding))) {
          copy.emitError()
              << "one async-copy bridge received conflicting requirements";
          return WalkResult::interrupt();
        }
        LDBG("[async-copy-requirement] loc="
             << copy.getLoc() << ", source=" << issue->sourceEncoding
             << ", shared=" << issue->sharedEncoding);
        return WalkResult::advance();
      });
  return result.wasInterrupted() ? failure() : success();
}

static FailureOr<Attribute>
inferGlobalAccessEncoding(Value pointer, Operation *anchor,
                          tt::ModuleAxisInfoAnalysis &axisInfo) {
  auto type = dyn_cast<RankedTensorType>(pointer.getType());
  if (!type)
    return failure();

  Attribute current =
      layout::unwrapNoVerifyEncoding(type.getEncoding());
  if (current && !isa<gd::AutoEncodingAttr>(current))
    return current;

  SmallVector<unsigned> order;
  unsigned transactionWidth = 1;
  if (type.getRank() == 2) {
    FailureOr<layout::c500::Rank2ContiguousOrderInfo> access =
        layout::c500::inferRank2ContiguousOrder(pointer, &axisInfo);
    unsigned elementBits =
        layout::c500::getTensorElementOrPointeeBitWidth(type);
    if (succeeded(access) && access->order.size() == 2 &&
        access->maxContiguousElements > 0 && elementBits > 0 &&
        elementBits <= 128) {
      order.assign(access->order.begin(), access->order.end());
      transactionWidth =
          std::min(128u / elementBits, access->maxContiguousElements);
      transactionWidth =
          static_cast<unsigned>(llvm::bit_floor(transactionWidth));
    }
  }
  if (order.empty()) {
    order.resize(type.getRank());
    std::iota(order.rbegin(), order.rend(), 0);
  }

  std::optional<ttg::BlockedEncodingAttr> encoding =
      layout::c500::getC500GlobalBlockedEncoding(
          pointer, anchor, order, std::max(1u, transactionWidth));
  if (!encoding)
    return anchor->emitError()
           << "cannot construct the deterministic C500 global-memory "
              "layout at this bridge";
  return Attribute(*encoding);
}

/// Return the selected DotOperand ownership when one global load is a direct
/// terminal of the selected dot dataflow. Multiple consumers must agree on the
/// complete encoding; otherwise the global boundary remains independent and
/// propagation materializes the necessary conversions at the consumers.
static std::optional<Attribute> getDirectDotLoadEncoding(
    tt::LoadOp load, ArrayRef<layout::DotPipelineFacts> facts,
    const layout::LayoutCandidate &candidate) {
  if (facts.size() != candidate.dots.size())
    return std::nullopt;

  Attribute selected;
  DenseSet<Operation *> directConsumers;
  for (auto [dotFacts, dotPlan] : llvm::zip(facts, candidate.dots)) {
    tt::DotOp dot = dotFacts.dot;
    for (unsigned operandIndex : {0u, 1u}) {
      Value operand = operandIndex == 0 ? dot.getA() : dot.getB();
      if (operand != load.getResult())
        continue;
      Attribute required = dotPlan.getOperand(operandIndex);
      if (selected && selected != required)
        return std::nullopt;
      selected = required;
      directConsumers.insert(dot);
    }
  }
  if (selected &&
      llvm::any_of(load.getResult().getUsers(), [&](Operation *user) {
        return !directConsumers.contains(user);
      }))
    return std::nullopt;
  return selected ? std::optional<Attribute>(selected) : std::nullopt;
}

/// Intersect one selected register ownership with the target-proven global
/// address domain. This is a legality query, not a new candidate axis: the
/// layout must preserve the proven contiguous dimension, and its per-thread
/// vector may not exceed either the address proof or C500's 128-bit
/// transaction width.
static bool isCompatibleDirectGlobalLoadEncoding(
    tt::LoadOp load, Attribute encoding,
    tt::ModuleAxisInfoAnalysis &axisInfo) {
  auto pointerType = dyn_cast<RankedTensorType>(load.getPtr().getType());
  auto resultType = dyn_cast<RankedTensorType>(load.getResult().getType());
  if (!pointerType || !resultType || pointerType.getRank() != 2 ||
      resultType.getRank() != 2 ||
      pointerType.getShape() != resultType.getShape())
    return false;

  FailureOr<layout::c500::Rank2ContiguousOrderInfo> access =
      layout::c500::inferRank2ContiguousOrder(load.getPtr(), &axisInfo, load);
  if (failed(access) || access->order.size() != 2 ||
      access->maxContiguousElements == 0)
    return false;

  RankedTensorType candidateType = resultType.cloneWithEncoding(encoding);
  SmallVector<unsigned> order = ttg::getOrder(candidateType);
  SmallVector<unsigned> contiguous = ttg::getContigPerThread(candidateType);
  unsigned dimension = access->order.front();
  unsigned elementBits =
      layout::c500::getTensorElementOrPointeeBitWidth(pointerType);
  if (order != access->order || dimension >= contiguous.size() ||
      elementBits == 0 || elementBits > 128)
    return false;
  unsigned transactionWidth =
      std::min(128u / elementBits, access->maxContiguousElements);
  return contiguous[dimension] > 0 &&
         contiguous[dimension] <= transactionWidth;
}

/// A load is one exact distributed-layout component: pointer, mask, other,
/// and result must agree. A direct selected-dot consumer may own this complete
/// component when that DotOperand encoding lies in the target-proven global
/// access domain. Every other load uses the same deterministic coalesced
/// blocked layout as the global-store boundary. Backward propagation closes the
/// chosen contract over every input while the pre-propagation IR remains
/// well-typed (all load operands and the result are still Auto). This
/// introduces neither a conversion nor a candidate axis.
static LogicalResult
planGlobalLoadRequirements(ModuleOp module,
                           tt::ModuleAxisInfoAnalysis &axisInfo,
                           ArrayRef<layout::DotPipelineFacts> facts,
                           const layout::LayoutCandidate &candidate,
                           RequirementPlan &plan) {
  WalkResult result = module.walk([&](tt::LoadOp load) {
    auto pointerType = dyn_cast<RankedTensorType>(load.getPtr().getType());
    auto resultType = dyn_cast<RankedTensorType>(load.getResult().getType());
    if (!pointerType || !resultType)
      return WalkResult::advance();
    // Only rank-2 currently carries a proven C500 coalescing choice. For
    // other ranks this planner would merely inject the same generic fallback
    // that placeholder resolution already provides, while prematurely
    // overriding a useful producer constraint.
    if (pointerType.getRank() != 2)
      return WalkResult::advance();

    FailureOr<Attribute> fallback =
        inferGlobalAccessEncoding(load.getPtr(), load, axisInfo);
    if (failed(fallback))
      return WalkResult::interrupt();
    Attribute encoding = *fallback;
    if (std::optional<Attribute> dotEncoding =
            getDirectDotLoadEncoding(load, facts, candidate);
        dotEncoding &&
        isCompatibleDirectGlobalLoadEncoding(load, *dotEncoding, axisInfo))
      encoding = *dotEncoding;

    if (hasAutoEncoding(load.getResult()) &&
        failed(recordSeed(plan, load.getResult(), encoding, load)))
      return WalkResult::interrupt();

    LDBG("[global-load-requirement] loc="
         << load.getLoc() << ", encoding=" << encoding
         << ", source="
         << (encoding == *fallback ? "coalesced-fallback"
                                   : "direct-dot-intersection"));
    return WalkResult::advance();
  });
  return result.wasInterrupted() ? failure() : success();
}

/// A rank-1 global result derived from a selected dot is naturally owned by
/// one accumulator row. Preserve that ownership across its exact store
/// component instead of resolving the pointer side independently to a generic
/// rank-1 blocked layout. The backward slice is the proof that this store
/// belongs to the selected dot pipeline; unrelated vector stores keep the
/// generic fallback.
static FailureOr<std::optional<Attribute>> inferRank1DotRowStoreEncoding(
    tt::StoreOp store, ArrayRef<layout::DotPipelineFacts> facts,
    const layout::LayoutCandidate &candidate,
    const layout::GluonRegionBranchAnalysis &carriers,
    bool &hasIncompatibleRows) {
  hasIncompatibleRows = false;
  if (facts.size() != candidate.dots.size())
    return failure();

  DenseSet<Value> visited;
  DenseSet<Operation *> backwardSlice;
  SmallVector<Value, 32> worklist{store.getValue()};
  while (!worklist.empty()) {
    Value value = worklist.pop_back_val();
    if (!value || !visited.insert(value).second)
      continue;
    llvm::append_range(worklist, carriers.getPredecessors(value));
    if (Operation *producer = value.getDefiningOp()) {
      backwardSlice.insert(producer);
      llvm::append_range(worklist, producer->getOperands());
    }
  }

  auto storeType = dyn_cast<RankedTensorType>(store.getValue().getType());
  if (!storeType || storeType.getRank() != 1)
    return std::optional<Attribute>();

  Attribute selected;
  for (auto [dotFacts, dotPlan] : llvm::zip(facts, candidate.dots)) {
    tt::DotOp dot = dotFacts.dot;
    if (!backwardSlice.contains(dot.getOperation()))
      continue;
    auto accumulatorType = dyn_cast<RankedTensorType>(dot.getD().getType());
    if (!accumulatorType || accumulatorType.getRank() != 2 ||
        accumulatorType.getShape()[0] != storeType.getShape()[0])
      continue;
    auto parent =
        dyn_cast<ttg::DistributedEncodingTrait>(dotPlan.mma);
    if (!parent)
      return store.emitError()
             << "selected dot accumulator is not a distributed encoding";
    Attribute row = ttg::SliceEncodingAttr::get(
        store.getContext(), /*dim=*/1, parent);
    if (selected && selected != row) {
      hasIncompatibleRows = true;
      return std::optional<Attribute>();
    }
    selected = row;
  }
  if (!selected)
    return std::optional<Attribute>();
  return std::optional<Attribute>(selected);
}

/// A store is an explicit global-memory boundary. Its value may be owned by an
/// MMA accumulator while its address/mask component must retain a coalesced
/// distributed layout. Requiring the value at the store edge produces exactly
/// one local conversion and prevents that physical store choice from flowing
/// backward into the accumulator component.
static LogicalResult
planGlobalStoreRequirements(ModuleOp module,
                            tt::ModuleAxisInfoAnalysis &axisInfo,
                            ArrayRef<layout::DotPipelineFacts> facts,
                            const layout::LayoutCandidate &candidate,
                            RequirementPlan &plan) {
  layout::GluonRegionBranchAnalysis carriers(module);
  if (failed(carriers.initialize()))
    return failure();
  WalkResult result = module.walk([&](tt::StoreOp store) {
    auto pointerType = dyn_cast<RankedTensorType>(store.getPtr().getType());
    auto valueType = dyn_cast<RankedTensorType>(store.getValue().getType());
    if (!pointerType || !valueType)
      return WalkResult::advance();
    // Preserve one coherent accumulator row when possible. If several selected
    // dots reach the same store with incompatible row ownership, the global
    // boundary receives one deterministic blocked layout and the value edge
    // becomes the necessary conversion. This keeps the mandatory per-dot
    // fallback materializable without changing any selected dot plan.
    if (pointerType.getRank() == 1) {
      bool incompatibleRows = false;
      FailureOr<std::optional<Attribute>> row =
          inferRank1DotRowStoreEncoding(store, facts, candidate, carriers,
                                        incompatibleRows);
      if (failed(row))
        return WalkResult::interrupt();
      if (!*row && !incompatibleRows)
        return WalkResult::advance();

      Attribute encoding;
      if (*row) {
        encoding = **row;
      } else {
        FailureOr<Attribute> fallback =
            inferGlobalAccessEncoding(store.getPtr(), store, axisInfo);
        if (failed(fallback))
          return WalkResult::interrupt();
        encoding = *fallback;
      }

      for (unsigned operandIndex : {0u, 2u}) {
        if (operandIndex == 2 && !store.getMask())
          continue;
        if (hasAutoEncoding(store->getOperand(operandIndex)) &&
            failed(recordBoundarySeed(plan, store, operandIndex, encoding))) {
          store.emitError()
              << "rank-1 dot-derived store received conflicting row seeds";
          return WalkResult::interrupt();
        }
      }

      if (*row) {
        if (hasAutoEncoding(store.getValue()) &&
            failed(recordBoundarySeed(plan, store, /*value=*/1, encoding))) {
          store.emitError()
              << "rank-1 dot-derived store received a conflicting value seed";
          return WalkResult::interrupt();
        }
      } else if (failed(recordRequirement(plan.tensorRequirements, store,
                                          /*value=*/1, encoding))) {
        store.emitError()
            << "rank-1 dot-derived store could not materialize its ownership "
               "conversion";
        return WalkResult::interrupt();
      }

      LDBG("[global-rank1-dot-store] loc="
           << store.getLoc() << ", encoding=" << encoding << ", source="
           << (*row ? "coherent-accumulator-row"
                    : "blocked-conflict-boundary"));
      return WalkResult::advance();
    }
    if (pointerType.getRank() != 2)
      return WalkResult::advance();

    FailureOr<Attribute> encoding =
        inferGlobalAccessEncoding(store.getPtr(), store, axisInfo);
    if (failed(encoding))
      return WalkResult::interrupt();
    Attribute valueEncoding =
        layout::unwrapNoVerifyEncoding(valueType.getEncoding());
    if (hasAutoEncoding(store.getPtr()) &&
        failed(recordBoundarySeed(plan, store, /*ptr=*/0, *encoding))) {
      store.emitError()
          << "one global store pointer received conflicting bridge seeds";
      return WalkResult::interrupt();
    }
    if (store.getMask() && hasAutoEncoding(store.getMask()) &&
        failed(recordBoundarySeed(plan, store, /*mask=*/2, *encoding))) {
      store.emitError()
          << "one global store mask received conflicting bridge seeds";
      return WalkResult::interrupt();
    }
    if (valueEncoding != *encoding &&
        failed(recordRequirement(plan.tensorRequirements, store,
                                 /*value=*/1, *encoding))) {
      store.emitError()
          << "one global store received conflicting bridge requirements";
      return WalkResult::interrupt();
    }
    LDBG("[global-store-requirement] loc="
         << store.getLoc() << ", encoding=" << *encoding);
    return WalkResult::advance();
  });
  return result.wasInterrupted() ? failure() : success();
}

static void materializeSeed(const PlannedSeed &seed, OpBuilder &builder) {
  if (auto result = dyn_cast<OpResult>(seed.value))
    builder.setInsertionPointAfter(result.getOwner());
  else
    builder.setInsertionPointToStart(
        cast<BlockArgument>(seed.value).getOwner());
  auto owner = builder.create<gd::SetAutoLayoutOp>(
      seed.value.getLoc(), seed.encoding, seed.value);
  if (!seed.replaceUses)
    return;
  Value value = seed.value;
  for (OpOperand &use :
       llvm::make_early_inc_range(value.getUses())) {
    Operation *user = use.getOwner();
    if (user == owner.getOperation() ||
        user->hasTrait<OpTrait::IsTerminator>() ||
        isa<RegionBranchOpInterface>(user))
      continue;
    use.set(owner.getResult());
  }
}

static void materializeBoundarySeed(const PlannedBoundarySeed &seed,
                                    OpBuilder &builder) {
  Value source = seed.owner->getOperand(seed.operandIndex);
  builder.setInsertionPoint(seed.owner);
  Value anchored = builder.create<gd::SetAutoLayoutOp>(
      seed.owner->getLoc(), seed.encoding, source);
  seed.owner->setOperand(seed.operandIndex, anchored);
}

static void materializeRequirement(const PlannedRequirement &requirement,
                                   OpBuilder &builder) {
  Value source = requirement.owner->getOperand(requirement.operandIndex);
  Type requiredType;
  if (auto tensor = dyn_cast<RankedTensorType>(source.getType())) {
    requiredType = tensor.cloneWithEncoding(requirement.encoding);
  } else {
    auto memdesc = cast<ttg::MemDescType>(source.getType());
    requiredType = ttg::MemDescType::get(
        memdesc.getShape(), memdesc.getElementType(), requirement.encoding,
        memdesc.getMemorySpace(), memdesc.getMutableMemory(),
        memdesc.getAllocShape());
  }

  builder.setInsertionPoint(requirement.owner);
  auto require = builder.create<gd::RequireLayoutOp>(
      requirement.owner->getLoc(), requiredType, source);
  requirement.owner->setOperand(requirement.operandIndex,
                                require.getResult());
}

static void materializeRelease(const PlannedRequirement &release,
                               OpBuilder &builder) {
  Value source = release.owner->getOperand(release.operandIndex);
  auto tensor = cast<RankedTensorType>(source.getType());
  Type releasedType = tensor.cloneWithEncoding(release.encoding);
  builder.setInsertionPoint(release.owner);
  auto boundary = builder.create<gd::ReleaseLayoutOp>(
      release.owner->getLoc(), releasedType, source);
  release.owner->setOperand(release.operandIndex,
                            boundary.getResult());
}

class TritonMETAXGPUGluonInsertRequireLayoutPass
    : public impl::TritonMETAXGPUGluonInsertRequireLayoutBase<
          TritonMETAXGPUGluonInsertRequireLayoutPass> {
public:
  TritonMETAXGPUGluonInsertRequireLayoutPass() = default;

  explicit TritonMETAXGPUGluonInsertRequireLayoutPass(int capability) {
    computeCapability = capability;
  }

  void runOnOperation() override {
    ModuleOp module = getOperation();
    FailureOr<layout::CandidateDomain> domain =
        layout::discoverC500LayoutCandidateDomain(module, computeCapability);
    if (failed(domain) ||
        failed(layout::insertLayoutRequirements(
            module, domain->fallback, computeCapability)))
      signalPassFailure();
  }
};

} // namespace

namespace triton::gpu::metax::gluon {

LogicalResult insertLayoutRequirements(ModuleOp module,
                                       const LayoutCandidate &candidate,
                                       int computeCapability) {
  GluonMemDescAliasAnalysis aliases(module);
  if (failed(aliases.initialize()))
    return failure();
  FailureOr<SmallVector<DotPipelineFacts, 4>> allFacts =
      collectAllDotFacts(module, aliases);
  if (failed(allFacts))
    return failure();

  SmallVector<DotPipelineFacts, 4> plannedFacts;
  llvm::copy_if(*allFacts, std::back_inserter(plannedFacts),
                isTunableC500Dot);
  if (plannedFacts.size() != candidate.dots.size())
    return module.emitError()
           << "selected candidate does not match the tunable dot set";

  tt::ModuleAxisInfoAnalysis axisInfo(module);
  SmallVector<DotPipelineFacts, 4> fixedBsmFacts;
  llvm::copy_if(
      *allFacts, std::back_inserter(fixedBsmFacts),
      [](const DotPipelineFacts &facts) {
        tt::DotOp dot = facts.dot;
        bool usesBsm = llvm::any_of(
            facts.operands, [](const DotPipelineOperandFacts &operand) {
              return operand.requiresBsmPermutation();
            });
        return usesBsm && hasAutoEncoding(dot.getC()) &&
               hasAutoEncoding(dot.getD());
      });

  // BSM owns one target-fixed physical ABI. It is finalized in the same thin
  // target-construction step, but never enters the candidate domain or digest.
  LayoutCandidate effectiveCandidate = candidate;
  if (!fixedBsmFacts.empty()) {
    if (failed(inferC500DotMemoryFacts(module, axisInfo, fixedBsmFacts)))
      return failure();
    for (const DotPipelineFacts &facts : fixedBsmFacts) {
      tt::DotOp dot = facts.dot;
      FailureOr<SmallVector<DotLayoutPlan, 8>> domain =
          inferC500DotLayoutDomain(facts, computeCapability);
      if (failed(domain) || domain->size() != 1)
        return dot.emitError()
               << "C500 BSM must have exactly one fixed layout contract";
      plannedFacts.push_back(facts);
      effectiveCandidate.dots.push_back(domain->front());
    }
  }

  RequirementPlan plan;
  DenseMap<Value, Attribute> sharedRoots;
  if (failed(planDotRequirements(module, plannedFacts, effectiveCandidate,
                                 plan,
                                 sharedRoots)) ||
      failed(planAsyncCopyRequirements(module, aliases, axisInfo,
                                       sharedRoots, plan)) ||
      failed(planGlobalLoadRequirements(module, axisInfo, plannedFacts,
                                        effectiveCandidate, plan)) ||
      failed(planGlobalStoreRequirements(module, axisInfo, plannedFacts,
                                         effectiveCandidate, plan)))
    return failure();

  OpBuilder builder(module.getContext());
  for (const PlannedSeed &seed : plan.accumulatorSeeds)
    materializeSeed(seed, builder);
  for (const PlannedBoundarySeed &seed : plan.boundarySeeds)
    materializeBoundarySeed(seed, builder);
  for (const PlannedRequirement &release : plan.tensorReleases)
    materializeRelease(release, builder);
  for (const PlannedRequirement &requirement : plan.tensorRequirements)
    materializeRequirement(requirement, builder);
  for (const PlannedRequirement &requirement : plan.memDescRequirements)
    materializeRequirement(requirement, builder);

  LDBG("[summary] tunable-dots="
       << candidate.dots.size() << ", fixed-bsm-dots="
       << fixedBsmFacts.size() << ", accumulator-seeds="
       << plan.accumulatorSeeds.size() << ", boundary-seeds="
       << plan.boundarySeeds.size() << ", tensor-requirements="
       << plan.tensorRequirements.size() << ", tensor-releases="
       << plan.tensorReleases.size() << ", memdesc-requirements="
       << plan.memDescRequirements.size());
  return success();
}

} // namespace triton::gpu::metax::gluon

std::unique_ptr<Pass> createTritonMETAXGPUGluonInsertRequireLayoutPass(
    int computeCapability) {
  return std::make_unique<
      TritonMETAXGPUGluonInsertRequireLayoutPass>(computeCapability);
}

} // namespace mlir

#undef LDBG
#undef DBGS
#undef DEBUG_TYPE
