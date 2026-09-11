#include "Gluon/GluonSharedAccessPath.h"
#include "Gluon/Passes.h"

#include "mlir/Analysis/AliasAnalysis.h"
#include "mlir/Analysis/Liveness.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Dominance.h"
#include "mlir/Interfaces/ControlFlowInterfaces.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Support/LLVM.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/Debug.h"

#define DEBUG_TYPE "metax-gluon-reorder-instructions"
#define DBGS() (llvm::dbgs() << "[" DEBUG_TYPE "]: ")
#define LDBG(X) LLVM_DEBUG(DBGS() << X << "\n")

namespace tt = ::mlir::triton;
namespace ttg = ::mlir::triton::gpu;
namespace gluon_analysis = ::mlir::triton::gpu::metax::gluon;

namespace mlir {

#define GEN_PASS_DEF_TRITONMETAXGPUGLUONREORDERINSTRUCTIONS
#include "Gluon/Passes.h.inc"

namespace {

enum class Placement { Before, After };

struct MotionPlan {
  Operation *operation = nullptr;
  Operation *anchor = nullptr;
  Placement placement = Placement::Before;
  StringRef reason;
};

static bool isSchedulingBoundary(Operation *op) {
  return isa<ttg::AsyncCopyGlobalToLocalOp, ttg::AsyncCommitGroupOp,
             ttg::AsyncWaitOp, ttg::GVMArriveOp, ttg::LocalBarrierOp,
             ttg::BarrierSharedOp, ttg::BarrierOp>(op);
}

static bool containsSchedulingBoundary(Operation *root) {
  WalkResult result = root->walk([&](Operation *nested) {
    if (nested != root && isSchedulingBoundary(nested))
      return WalkResult::interrupt();
    return WalkResult::advance();
  });
  return result.wasInterrupted();
}

/// Proves dependence-safe instruction motion for one function. The analysis
/// is deliberately conservative: an unknown effect or alias blocks motion,
/// while target view paths can refine generic MayAlias results only when all
/// alternatives are rooted in disjoint allocations or byte intervals.
class GluonMemoryMotionAnalysis {
public:
  explicit GluonMemoryMotionAnalysis(tt::FuncOp func)
      : aliases(func), accessPaths(func), dominance(func), liveness(func) {}

  LogicalResult initialize() { return accessPaths.initialize(); }

  bool canSinkPureBefore(Operation *candidate, Operation *user) {
    Operation *ancestor =
        candidate->getBlock()->findAncestorOpInBlock(*user);
    if (!ancestor) {
      reject(candidate, user,
             "the use has no ancestor in the candidate block");
      return false;
    }
    if (!isa<RegionBranchOpInterface>(ancestor)) {
      reject(candidate, user,
             "the crossed region does not implement RegionBranchOpInterface");
      return false;
    }
    if (!dominance.dominates(candidate, user)) {
      reject(candidate, user,
             "the candidate does not dominate the target use");
      return false;
    }
    logCrossRegionLiveness(candidate, user);
    return true;
  }

  bool canSinkLoadBefore(ttg::LocalLoadOp load, Operation *user) {
    if (!canSinkPureBefore(load, user))
      return false;
    Operation *ancestor = load->getBlock()->findAncestorOpInBlock(*user);

    for (Operation *crossed = load->getNextNode(); crossed;
         crossed = crossed->getNextNode()) {
      if (hasConflictingEffect(load.getSrc(), crossed)) {
        reject(load, crossed,
               "a may-alias write/free or scheduling boundary is crossed");
        return false;
      }
      if (crossed == ancestor)
        return true;
    }
    reject(load, user, "the region ancestor is not reachable in block order");
    return false;
  }

  bool canMoveLoadAfter(ttg::LocalLoadOp load, Operation *anchor) {
    if (load->getBlock() != anchor->getBlock() ||
        !dominance.dominates(load.getOperation(), anchor)) {
      reject(load, anchor,
             "the load and anchor do not form a dominated block range");
      return false;
    }
    for (Operation *crossed = load->getNextNode(); crossed;
         crossed = crossed->getNextNode()) {
      if (hasConflictingEffect(load.getSrc(), crossed)) {
        reject(load, crossed,
               "a may-alias write/free or scheduling boundary is crossed");
        return false;
      }
      if (crossed == anchor)
        return true;
    }
    reject(load, anchor, "the anchor is not reachable in block order");
    return false;
  }

private:
  void logCrossRegionLiveness(Operation *candidate, Operation *user) const {
    Block *sourceBlock = candidate->getBlock();
    Block *targetBlock = user->getBlock();
    if (sourceBlock == targetBlock)
      return;

    for (Value result : candidate->getResults()) {
      if (result.use_empty())
        continue;
      bool liveIn = liveness.getLiveIn(targetBlock).contains(result);
      bool liveOut = liveness.getLiveOut(sourceBlock).contains(result);
      LDBG("[liveness] operation="
           << candidate->getName() << " result=" << result
           << " source-block=" << sourceBlock << " target-block="
           << targetBlock << " live-in=" << liveIn
           << " live-out=" << liveOut);
    }
  }

  static bool intervalsAreDisjoint(
      const gluon_analysis::SharedAccessPathFact &lhs,
      const gluon_analysis::SharedAccessPathFact &rhs) {
    if (!lhs.byteInterval || !rhs.byteInterval)
      return false;
    return lhs.byteInterval->second <= rhs.byteInterval->first ||
           rhs.byteInterval->second <= lhs.byteInterval->first;
  }

  bool targetPathsProveNoAlias(Value lhs, Value rhs) const {
    if (!isa<ttg::MemDescType>(lhs.getType()) ||
        !isa<ttg::MemDescType>(rhs.getType()))
      return false;
    const auto &lhsState = accessPaths.lookup(lhs);
    const auto &rhsState = accessPaths.lookup(rhs);
    if (lhsState.unknown || rhsState.unknown || lhsState.alternatives.empty() ||
        rhsState.alternatives.empty())
      return false;
    return llvm::all_of(lhsState.alternatives, [&](const auto &lhsFact) {
      return llvm::all_of(rhsState.alternatives, [&](const auto &rhsFact) {
        if (rootsAreDistinctAllocations(lhsFact.logicalRoot,
                                        rhsFact.logicalRoot))
          return true;
        return lhsFact.logicalRoot == rhsFact.logicalRoot &&
               intervalsAreDisjoint(lhsFact, rhsFact);
      });
    });
  }

  static bool rootsAreDistinctAllocations(Value lhs, Value rhs) {
    if (!lhs || !rhs || lhs == rhs)
      return false;
    return lhs.getDefiningOp<ttg::LocalAllocOp>() &&
           rhs.getDefiningOp<ttg::LocalAllocOp>();
  }

  bool mayAlias(Value lhs, Value rhs) {
    if (!lhs || !rhs)
      return true;
    AliasResult result = aliases.alias(lhs, rhs);
    LDBG("[candidate] alias lhs=" << lhs << " rhs=" << rhs
                                  << " result=" << result);
    return !result.isNo() && !targetPathsProveNoAlias(lhs, rhs);
  }

  bool hasConflictingEffect(Value loadedMemDesc, Operation *crossed) {
    if (isSchedulingBoundary(crossed) || containsSchedulingBoundary(crossed))
      return true;
    if (isMemoryEffectFree(crossed))
      return false;

    auto effects = getEffectsRecursively(crossed);
    if (!effects)
      return true;
    return llvm::any_of(*effects, [&](MemoryEffects::EffectInstance effect) {
      if (isa<MemoryEffects::Read, MemoryEffects::Allocate>(effect.getEffect()))
        return false;
      if (!isa<MemoryEffects::Write, MemoryEffects::Free>(effect.getEffect()))
        return true;
      Value affected = effect.getValue();
      return !affected || mayAlias(loadedMemDesc, affected);
    });
  }

  static void reject(Operation *candidate, Operation *boundary,
                     StringRef reason) {
    LDBG("[reject] candidate=" << candidate->getName()
                                << " location=" << candidate->getLoc()
                                << " boundary=" << boundary->getName()
                                << " boundary-location=" << boundary->getLoc()
                                << " reason=" << reason);
  }

  AliasAnalysis aliases;
  gluon_analysis::SharedAccessPathAnalysis accessPaths;
  DominanceInfo dominance;
  Liveness liveness;
};

static bool shouldSinkForRegisterPressure(Operation *op) {
  if (isa<ttg::LocalLoadOp>(op))
    return true;
  auto conversion = dyn_cast<ttg::ConvertLayoutOp>(op);
  return conversion && isa<ttg::DotOperandEncodingAttr>(
                           conversion.getType().getEncoding());
}

static Operation *getFirstUseAncestor(Operation *op) {
  SmallVector<Operation *> ancestors;
  for (Operation *user : op->getUsers())
    if (Operation *ancestor = op->getBlock()->findAncestorOpInBlock(*user))
      ancestors.push_back(ancestor);
  auto first = llvm::min_element(ancestors, [](Operation *lhs, Operation *rhs) {
    return lhs->isBeforeInBlock(rhs);
  });
  return first == ancestors.end() ? nullptr : *first;
}

static void appendPlan(SmallVectorImpl<MotionPlan> &plans,
                       DenseSet<Operation *> &planned, Operation *operation,
                       Operation *anchor, Placement placement,
                       StringRef reason) {
  if (!operation || !anchor || operation == anchor ||
      !planned.insert(operation).second)
    return;
  plans.push_back({operation, anchor, placement, reason});
  LDBG("[plan] operation=" << operation->getName()
                             << " location=" << operation->getLoc()
                             << " anchor=" << anchor->getName()
                             << " anchor-location=" << anchor->getLoc()
                             << " placement="
                             << (placement == Placement::Before ? "before"
                                                                : "after")
                             << " reason=" << reason);
}

static LogicalResult reorderFunction(tt::FuncOp func) {
  GluonMemoryMotionAnalysis motion(func);
  if (failed(motion.initialize()))
    return func.emitError(
        "failed to initialize Gluon shared access-path analysis for safe "
        "instruction reordering");

  SmallVector<MotionPlan> plans;
  DenseSet<Operation *> planned;

  func.walk([&](Operation *op) {
    if (!shouldSinkForRegisterPressure(op) || !op->hasOneUse())
      return;
    Operation *user = *op->user_begin();
    if (op->getBlock() == user->getBlock())
      return;
    if (auto load = dyn_cast<ttg::LocalLoadOp>(op)) {
      if (!motion.canSinkLoadBefore(load, user))
        return;
    } else if (!isMemoryEffectFree(op) ||
               !motion.canSinkPureBefore(op, user)) {
      return;
    }
    appendPlan(plans, planned, op, user, Placement::Before,
               "shorten a proven-safe loop-crossing register live range");
  });

  func.walk([&](ttg::ConvertLayoutOp conversion) {
    if (planned.contains(conversion))
      return;
    Operation *firstUse = getFirstUseAncestor(conversion);
    if (!firstUse)
      return;
    Operation *lastDealloc = nullptr;
    for (Operation *next = conversion->getNextNode(); next && next != firstUse;
         next = next->getNextNode())
      if (isa<ttg::LocalDeallocOp>(next))
        lastDealloc = next;
    appendPlan(plans, planned, conversion, lastDealloc, Placement::After,
               "delay a pure layout conversion beyond an ended shared lifetime");
  });

  func.walk([&](ttg::LocalAllocOp alloc) {
    if (!alloc.getSrc() || planned.contains(alloc))
      return;
    Operation *source = alloc.getSrc().getDefiningOp();
    if (!source || source->getBlock() != alloc->getBlock() ||
        isa<arith::ConstantOp, tt::SplatOp>(source))
      return;
    appendPlan(plans, planned, alloc, source, Placement::After,
               "start a fresh allocation lifetime after its initializer is "
               "available");
  });

  func.walk([&](tt::TransposeOpInterface transpose) {
    Operation *op = transpose.getOperation();
    Operation *source = transpose.getSrc().getDefiningOp();
    if (!source || source->getBlock() != op->getBlock() ||
        !isMemoryEffectFree(op))
      return;
    appendPlan(plans, planned, op, source, Placement::After,
               "place a pure coordinate view next to its definition");
  });

  func.walk([&](ttg::LocalLoadOp loadB) {
    if (planned.contains(loadB) || !loadB->hasOneUse())
      return;
    auto dot = dyn_cast<tt::DotOp>(*loadB->user_begin());
    auto dotEncoding = dyn_cast_or_null<ttg::DotOperandEncodingAttr>(
        loadB.getType().getEncoding());
    if (!dot || !dotEncoding || dotEncoding.getOpIdx() != 1)
      return;
    auto loadA = dot.getA().getDefiningOp<ttg::LocalLoadOp>();
    if (!loadA || loadA->getBlock() != loadB->getBlock() ||
        !loadB->isBeforeInBlock(loadA) ||
        !motion.canMoveLoadAfter(loadB, loadA))
      return;
    appendPlan(plans, planned, loadB, loadA, Placement::After,
               "keep dot operand-B loading after operand-A when dependence-safe");
  });

  for (const MotionPlan &plan : plans) {
    if (plan.placement == Placement::Before)
      plan.operation->moveBefore(plan.anchor);
    else
      plan.operation->moveAfter(plan.anchor);
    LDBG("[materialize] operation=" << plan.operation->getName()
                                     << " location=" << plan.operation->getLoc()
                                     << " reason=" << plan.reason);
  }
  return success();
}

class TritonMETAXGPUGluonReorderInstructionsPass
    : public impl::TritonMETAXGPUGluonReorderInstructionsBase<
          TritonMETAXGPUGluonReorderInstructionsPass> {
public:
  void runOnOperation() override {
    for (tt::FuncOp func : getOperation().getOps<tt::FuncOp>()) {
      if (failed(reorderFunction(func))) {
        signalPassFailure();
        return;
      }
    }
  }
};

} // namespace

std::unique_ptr<Pass>
createTritonMETAXGPUGluonReorderInstructionsPass() {
  return std::make_unique<TritonMETAXGPUGluonReorderInstructionsPass>();
}

} // namespace mlir
