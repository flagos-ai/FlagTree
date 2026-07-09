#include "mlir/IR/DialectRegistry.h"
#include "triton/Analysis/Utility.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/Transforms/Utility.h"
#include "triton/Dialect/TritonNvidiaGPU/IR/Dialect.h"
#include "triton/Dialect/TritonNvidiaGPU/Transforms/Passes.h"
#ifdef __TLE__
#include "tle/dialect/include/IR/Dialect.h"
#endif
#include "triton/Tools/Sys/GetEnv.hpp"
#include "llvm/Support/Debug.h"

//===----------------------------------------------------------------------===//
//
// This pass works after all other passes, inserting fences to ensure that
// memory operations are properly ordered across generic and async proxy.
//
//===----------------------------------------------------------------------===//

namespace ttg = mlir::triton::gpu;

namespace mlir {
namespace triton {
namespace nvidia_gpu {

#ifdef __TLE__
static constexpr llvm::StringLiteral
    kTleWgmmaAccumulatorChainCAttr("tle.wgmma_accumulator_chain_c");

static Value skipNoopDefs(Value value) {
  while (Operation *def = value.getDefiningOp()) {
    if (!isNoop(def) || def->getNumOperands() != 1 || def->getNumResults() != 1)
      break;
    value = def->getOperand(0);
  }
  return value;
}

static WarpGroupDotOp getAccumulatorChainSourceDot(WarpGroupDotOp dot) {
  return skipNoopDefs(dot.getC()).getDefiningOp<WarpGroupDotOp>();
}
#endif

#define GEN_PASS_DEF_TRITONGPUFENCEINSERTION
#include "triton/Dialect/TritonNvidiaGPU/Transforms/Passes.h.inc"

struct FenceInsertionPass
    : public impl::TritonGPUFenceInsertionBase<FenceInsertionPass> {

public:
  using impl::TritonGPUFenceInsertionBase<
      FenceInsertionPass>::TritonGPUFenceInsertionBase;
#ifdef __TLE__
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<triton::tle::TleDialect>();
  }
#endif
  // TODO: support more general patterns to insert fences. eg. any op(generic)
  // to shared in use-def chain which refers by async proxy. We have generic(
  // convertlayout with sts/stmatix) + fence + async(wgmma) up to now
  void runOnOperation() override {
    // Only insert fences for compute capability 9.0
    if (computeCapability < 90)
      return;
    ModuleOp mod = getOperation();
    mod.walk([&](DotOpInterface dotOp) {
      Value a = dotOp.getA();
      Value b = dotOp.getB();
      SmallVector<Operation *> copyRegToSharedOpsA = findCopyRegToSharedOps(a);
      SmallVector<Operation *> copyRegToSharedOpsB = findCopyRegToSharedOps(b);
      if (copyRegToSharedOpsA.empty() && copyRegToSharedOpsB.empty())
        return WalkResult::advance();

      OpBuilder builder(dotOp);
#ifdef __TLE__
      SmallVector<Value> deps;
      if (!copyRegToSharedOpsA.empty() && isa<ttg::MemDescType>(a.getType()))
        deps.push_back(a);
      if (!copyRegToSharedOpsB.empty() && isa<ttg::MemDescType>(b.getType()))
        deps.push_back(b);
      Operation *fence =
          deps.empty()
              ? FenceAsyncSharedOp::create(builder, dotOp.getLoc(),
                                           /*bCluster=*/false)
                    .getOperation()
              : ([&]() -> Operation * {
                  builder.getContext()
                      ->getOrLoadDialect<triton::tle::TleDialect>();
                  return triton::tle::WGMMASharedOperandFenceOp::create(
                             builder, dotOp.getLoc(), deps,
                             /*bCluster=*/false)
                      .getOperation();
                })();
#else
      Operation *fence = FenceAsyncSharedOp::create(builder, dotOp.getLoc(),
                                                    /*bCluster=*/false)
                             .getOperation();
#endif
      // If there is all the dependencies are outside of the loop try to hoist
      // the fence.
      while (auto loopOp = fence->getParentOfType<LoopLikeOpInterface>()) {
        if (!copyRegToSharedOpsA.empty() &&
            llvm::any_of(copyRegToSharedOpsA,
                         [&](Operation *op) { return loopOp->isAncestor(op); }))
          break;
        if (!copyRegToSharedOpsB.empty() &&
            llvm::any_of(copyRegToSharedOpsB,
                         [&](Operation *op) { return loopOp->isAncestor(op); }))
          break;
#ifdef __TLE__
        if (!deps.empty() && llvm::any_of(deps, [&](Value dep) {
              return valueDefinedInside(dep, loopOp);
            }))
          break;
#endif
        loopOp.moveOutOfLoop(fence);
      }

      // If the previous op is already a fence, this one isn't needed.
#ifdef __TLE__
      if (hasEquivalentPreviousFence(fence)) {
        Operation *previousFence = fence->getPrevNode();
        closeOpenWgmmaCommitGroupBefore(previousFence);
        materializeAccumulatorChainBeforeLateFence(dotOp, previousFence);
        fence->erase();
      } else {
        closeOpenWgmmaCommitGroupBefore(fence);
        materializeAccumulatorChainBeforeLateFence(dotOp, fence);
      }
#else
      if (auto lastFence =
              dyn_cast_or_null<FenceAsyncSharedOp>(fence->getPrevNode())) {
        if (lastFence.getBCluster() ==
            cast<FenceAsyncSharedOp>(fence).getBCluster())
          fence->erase();
      }
#endif

      return WalkResult::advance();
    });
  }

private:
#ifdef __TLE__
  void materializeAccumulatorChainBeforeLateFence(DotOpInterface dotOp,
                                                  Operation *fence) const {
    auto targetDot = dyn_cast<WarpGroupDotOp>(dotOp.getOperation());
    if (!targetDot || !targetDot->hasAttr(kTleWgmmaAccumulatorChainCAttr))
      return;
    if (!fence || fence->getBlock() != targetDot->getBlock() ||
        !fence->isBeforeInBlock(targetDot))
      return;

    WarpGroupDotOp sourceDot = getAccumulatorChainSourceDot(targetDot);
    if (!sourceDot || sourceDot->getBlock() != targetDot->getBlock() ||
        !sourceDot->isBeforeInBlock(fence))
      return;

    unsigned commitsThroughSourceGroup = 0;
    for (Operation *op = sourceDot->getNextNode(); op && op != fence;
         op = op->getNextNode()) {
      if (isa<WarpGroupDotWaitOp>(op))
        return;
      if (isa<WarpGroupDotCommitOp>(op))
        ++commitsThroughSourceGroup;
    }

    // FenceInsertion runs after the WGMMA wait scheduler. If it inserts a
    // shared-operand fence in front of a dot that was previously allowed to use
    // a pending accumulator as C, the pass has created a new ptxas-visible
    // accumulator pipeline boundary. Materialize the source accumulator before
    // that boundary and rewrite the later dot to consume the waited value. This
    // keeps the invariant used by the scheduler: non-WGMMA sync/fence ops may
    // overlap pending WGMMA work, but a later WGMMA cannot reuse the same
    // accumulator across that boundary until a matching wait has closed the
    // hardware stage.
    OpBuilder builder(fence);
    SmallVector<Value, 4> waitOperands;
    waitOperands.push_back(sourceDot.getResult());
    for (Value operand : sourceDot->getOperands()) {
      if (!isa<ttg::MemDescType>(operand.getType()))
        continue;
      if (!llvm::is_contained(waitOperands, operand))
        waitOperands.push_back(operand);
    }

    unsigned pendings =
        commitsThroughSourceGroup > 0 ? commitsThroughSourceGroup - 1 : 0;
    auto wait = WarpGroupDotWaitOp::create(builder, fence->getLoc(),
                                           waitOperands, pendings);
    targetDot.getCMutable().assign(wait.getResult(0));
    targetDot->removeAttr(kTleWgmmaAccumulatorChainCAttr);
  }

  void closeOpenWgmmaCommitGroupBefore(Operation *boundary) const {
    if (!boundary)
      return;

    // FenceInsertion runs after the WGMMA scheduler in the NVIDIA pipeline.
    // A fence inserted at this point must still obey the same invariant as
    // scheduler-visible fences: a WGMMA commit group is a contiguous launch
    // region. If the fence lands after async WGMMA launches but before their
    // commit, close the group before the fence. This is only a commit boundary;
    // materializeAccumulatorChainBeforeLateFence handles the separate case
    // where the late fence splits an already-scheduled accumulator chain.
    for (Operation *prev = boundary->getPrevNode(); prev;
         prev = prev->getPrevNode()) {
      if (isa<WarpGroupDotCommitOp, WarpGroupDotWaitOp>(prev))
        return;
      if (auto dot = dyn_cast<WarpGroupDotOp>(prev)) {
        if (!dot.getIsAsync())
          return;
        OpBuilder builder(boundary);
        WarpGroupDotCommitOp::create(builder, boundary->getLoc());
        return;
      }
    }
  }

  void findAsyncCopyToSharedUsers(Value value, DenseSet<Value> &visitedValues,
                                  llvm::SetVector<Operation *> &result) {
    if (!visitedValues.insert(value).second)
      return;
    for (Operation *user : value.getUsers()) {
      if (auto asyncCopy = dyn_cast<ttg::AsyncCopyGlobalToLocalOp>(user)) {
        if (asyncCopy.getResult() == value)
          result.insert(user);
        continue;
      }
      if (user->hasTrait<OpTrait::MemDescViewTrait>()) {
        for (Value viewResult : user->getResults())
          findAsyncCopyToSharedUsers(viewResult, visitedValues, result);
      }
    }
  }

#ifdef __TLE__
  void
  findLocalStoresThroughMemDescViews(Value value,
                                     DenseSet<Value> &visitedValues,
                                     llvm::SetVector<Operation *> &result) {
    if (!visitedValues.insert(value).second)
      return;
    for (Operation *user : value.getUsers()) {
      if (isa<ttg::LocalStoreOp>(user)) {
        result.insert(user);
        continue;
      }
      if (user->hasTrait<OpTrait::MemDescViewTrait>()) {
        for (Value viewResult : user->getResults())
          findLocalStoresThroughMemDescViews(viewResult, visitedValues, result);
      }
    }
  }

  void findLocalStoresInRegionThroughMemDescViews(
      Value value, Region *region, llvm::SetVector<Operation *> &result) {
    llvm::SetVector<Value> aliases;
    aliases.insert(value);

    bool changed = true;
    while (changed) {
      changed = false;
      region->walk([&](Operation *op) {
        if (!op->hasTrait<OpTrait::MemDescViewTrait>())
          return;
        if (!llvm::any_of(op->getOperands(), [&](Value operand) {
              return aliases.count(operand);
            }))
          return;
        for (Value viewResult : op->getResults())
          changed |= aliases.insert(viewResult);
      });
    }

    region->walk([&](ttg::LocalStoreOp store) {
      if (aliases.count(store.getDst()))
        result.insert(store);
    });
  }
#endif

  bool valueDefinedInside(Value value, Operation *ancestor) const {
    if (Operation *def = value.getDefiningOp())
      return ancestor == def || ancestor->isAncestor(def);
    auto arg = dyn_cast<BlockArgument>(value);
    if (!arg)
      return false;
    Operation *owner = arg.getOwner()->getParentOp();
    return owner && (owner == ancestor || ancestor->isAncestor(owner));
  }

  bool hasEquivalentPreviousFence(Operation *fence) const {
    bool bCluster = false;
    if (!getFenceCluster(fence, bCluster))
      return false;
    Operation *prev = fence->getPrevNode();
    if (auto asyncFence = dyn_cast_or_null<FenceAsyncSharedOp>(prev))
      return asyncFence.getBCluster() == bCluster;
    if (auto operandFence =
            dyn_cast_or_null<triton::tle::WGMMASharedOperandFenceOp>(prev))
      return operandFence.getBCluster() == bCluster;
    return false;
  }

  bool getFenceCluster(Operation *fence, bool &bCluster) const {
    if (auto asyncFence = dyn_cast_or_null<FenceAsyncSharedOp>(fence)) {
      bCluster = asyncFence.getBCluster();
      return true;
    }
    if (auto operandFence =
            dyn_cast_or_null<triton::tle::WGMMASharedOperandFenceOp>(fence)) {
      bCluster = operandFence.getBCluster();
      return true;
    }
    return false;
  }
#endif

  // Return true if the operand depends on a copy from register to shared.
  SmallVector<Operation *> findCopyRegToSharedOps(Value operand) {
    DenseSet<Value> visited;
    llvm::SetVector<Operation *> result;
    findCopyRegToSharedOps(operand, visited, result);
    return result.takeVector();
  }

  void findCopyRegToSharedOps(Value operand, DenseSet<Value> &visited,
                              llvm::SetVector<Operation *> &result) {
    // If the value has already been visited we can safely return false as we
    // would early return when true.
    if (visited.count(operand))
      return;
    visited.insert(operand);
    if (!isa<triton::gpu::MemDescType>(operand.getType()))
      return;

    auto op = operand.getDefiningOp();
    if (op) {
      // reach an alloc copying from register, we need a fence.
      if (auto localAlloc = dyn_cast<ttg::LocalAllocOp>(op)) {
        if (localAlloc.getSrc()) {
          result.insert(op);
        }
#ifdef __TLE__
        DenseSet<Value> visitedStoreValues;
        findLocalStoresThroughMemDescViews(localAlloc.getResult(),
                                           visitedStoreValues, result);
        if (!result.empty())
          return;
#endif
        // Check if there are local_store ops that write to that buffer.
        for (auto user : localAlloc.getResult().getUsers()) {
          while (user->hasOneUse() &&
                 user->hasTrait<OpTrait::MemDescViewTrait>()) {
            user = *user->getUsers().begin();
          }
          if (isa<ttg::LocalStoreOp>(user)) {
            result.insert(user);
            return;
          }
        }
#ifdef __TLE__
        DenseSet<Value> visitedValues;
        findAsyncCopyToSharedUsers(localAlloc.getResult(), visitedValues,
                                   result);
        if (!result.empty())
          return;
#endif
      }
      // if it is not an alloc, iterate over the operands.
      for (auto v : op->getOperands()) {
        findCopyRegToSharedOps(v, visited, result);
      }
      return;
    }

    // reach BlockArgument
    BlockArgument arg = cast<BlockArgument>(operand);
#ifdef __TLE__
    DenseSet<Value> visitedStoreValues;
    findLocalStoresThroughMemDescViews(arg, visitedStoreValues, result);
    findLocalStoresInRegionThroughMemDescViews(arg, arg.getOwner()->getParent(),
                                               result);
    if (!result.empty())
      return;
#endif
    unsigned argNum = arg.getArgNumber();
    Operation *argOwner = arg.getOwner()->getParentOp();
    // look through ForOp iter argument
    if (auto forOp = dyn_cast<scf::ForOp>(argOwner)) {
      assert(argNum != 0 && "induction var cannot be memdesc type");
      --argNum;
      // prologue
      findCopyRegToSharedOps(forOp.getInitArgs()[argNum], visited, result);
      // yield
      auto yieldOp = forOp.getBody()->getTerminator();
      Value v = yieldOp->getOperand(argNum);
      findCopyRegToSharedOps(v, visited, result);
      return;
    }

    // look through `ttg.warp_specialize`.
    if (auto wsOp = dyn_cast<ttg::WarpSpecializePartitionsOp>(argOwner)) {
      findCopyRegToSharedOps(wsOp.getParentOp().getExplicitCaptures()[argNum],
                             visited, result);
      return;
    }

    // Conservatively return true for other ops
    result.insert(argOwner);
  }
};

} // namespace nvidia_gpu
} // namespace triton
} // namespace mlir
