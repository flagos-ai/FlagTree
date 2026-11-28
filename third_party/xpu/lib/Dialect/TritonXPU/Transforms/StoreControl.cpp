//===----------------------------------------------------------------------===//
// TODO[dyq]: Pass Description
//===----------------------------------------------------------------------===//

#include "triton/Dialect/TritonXPU/IR/Dialect.h"
#include "triton/Dialect/TritonXPU/Transforms/Passes.h"

#define DEBUG_TYPE "tritonxpu-store-control"

namespace mlir {
namespace triton {
namespace xpu {

#define GEN_PASS_DEF_TRITONXPUSTORECONTROL
#include "triton/Dialect/TritonXPU/Transforms/Passes.h.inc"

struct TritonXPUStoreControl
    : public impl::TritonXPUStoreControlBase<TritonXPUStoreControl> {

  using impl::TritonXPUStoreControlBase<
      TritonXPUStoreControl>::TritonXPUStoreControlBase;

  void getGroupInfo(triton::xpu::ReduceOp &reduceOp, int64_t &groupSize,
                    int64_t &groupNum) {
    auto types = reduceOp.getOperandTypes();
    assert(types.size() > 1);
    for (int i = 0; i < types.size() - 1; ++i) {
      if (auto tensorType = dyn_cast<RankedTensorType>(types[i])) {
        auto clusterEncoding =
            cast<triton::xpu::ClusterLayoutAttr>(tensorType.getEncoding());
        if (i == 0) {
          groupSize = product(clusterEncoding.getCoresPerGroup());
          groupNum = product(clusterEncoding.getGroupsPerCluster());
        } else {
          assert(groupSize == product(clusterEncoding.getCoresPerGroup()));
          assert(groupNum == product(clusterEncoding.getGroupsPerCluster()));
        }
      }
    }
  }

  bool findDefChain(Operation *startOp, Operation *endOp,
                    SetVector<Operation *> &chain,
                    SetVector<Operation *> &visitedOps) {
    if (!startOp) {
      return false;
    }
    // `visitedOps` was passed in but never filled: a DAG with shared
    // subexpressions was walked once per path, and vander's inlined fp64
    // pow/sin/cos chains made this pass take 85 s.  Marking the node keeps
    // the first path found and makes the walk linear.
    if (!visitedOps.insert(startOp)) {
      return false;
    }
    chain.insert(startOp);
    if (startOp == endOp) {
      return true;
    }
    for (auto operand : startOp->getOperands()) {
      auto defOp = operand.getDefiningOp();
      if (!visitedOps.count(defOp)) {
        if (findDefChain(defOp, endOp, chain, visitedOps)) {
          return true;
        }
      }
    }
    chain.pop_back();
    return false;
  }

  bool hasBroadcast(Operation *startOp, Operation *endOp) {
    SetVector<Operation *> chain;
    SetVector<Operation *> visitedOps;
    findDefChain(startOp, endOp, chain, visitedOps);
    for (auto op : chain) {
      if (isa<triton::xpu::BroadcastOp, triton::SplatOp>(op)) {
        return true;
      }
    }
    return false;
  }

  bool isSameSize(triton::xpu::ReduceOp &reduceOp,
                  triton::xpu::StoreOp storeOp) {
    SmallVector<int64_t> redResShape = {1};
    auto redRes = reduceOp.getResult()[0];
    if (auto redResTy = dyn_cast<RankedTensorType>(redRes.getType())) {
      auto sliceEncoding =
          cast<triton::gpu::SliceEncodingAttr>(redResTy.getEncoding());
      redResShape = SmallVector<int64_t>(redResTy.getShape().begin(),
                                         redResTy.getShape().end());
    }
    SmallVector<int64_t> storeValShape = {1};
    auto storeVal = storeOp.getValue();
    if (auto storeValTy = dyn_cast<RankedTensorType>(storeVal.getType())) {
      storeValShape = SmallVector<int64_t>(storeValTy.getShape().begin(),
                                           storeValTy.getShape().end());
    }
    if (product(redResShape) == product(storeValShape) && !storeOp.getIndex()) {
      return true;
    }
    return false;
  }

  // Does the stored value reach the reduce result without passing through a
  // loop-carried value?
  //
  // Why: the running-total rule only holds while the value written out is the
  // reduce cached result. A value that comes out of a loop iter_arg has an
  // accumulator of its own, and a masked-out slice leaves that accumulator at
  // its init value -- storing it every slice then writes the init value over
  // the total an earlier slice already produced (conv_tbc_backward bias: the
  // accumulator is rebuilt from zeros per slice and the tail slices are
  // masked out).
  // Whether `value` is a per-iteration accumulator rather than a running total.
  //
  // Why: both a loop RESULT and a loop iter_arg carry an accumulator of their
  // own, so either one disqualifies the value the same way.
  static bool isLoopCarried(Value value) {
    if (Operation *def = value.getDefiningOp())
      return isa<scf::ForOp, scf::WhileOp>(def);
    auto blockArg = dyn_cast<BlockArgument>(value);
    if (!blockArg)
      return false;
    return isa_and_nonnull<scf::ForOp, scf::WhileOp>(
        blockArg.getOwner()->getParentOp());
  }

  static bool isReduceResultValue(Value value, triton::xpu::ReduceOp reduceOp) {
    SmallPtrSet<Operation *, 16> seen;
    SmallVector<Value> worklist{value};
    // EVERY path has to end at the reduce. Returning true on the first path
    // that reaches it made the answer depend on the worklist pop order: for
    // `add(reduceResult, loopCarried)` the reduce operand could be popped
    // first and the loop-carried one never looked at, so the last-iteration
    // guard was dropped for a value that still holds an unfinished
    // accumulator -- exactly what the comment above is there to prevent.
    bool reachedReduce = false;
    while (!worklist.empty()) {
      Value current = worklist.pop_back_val();
      if (isLoopCarried(current))
        return false;
      Operation *def = current.getDefiningOp();
      if (!def)
        continue;
      if (def == reduceOp) {
        // Stop here: what feeds the reduce cannot change this answer.
        reachedReduce = true;
        continue;
      }
      if (!seen.insert(def).second)
        continue;
      worklist.append(def->getOperands().begin(), def->getOperands().end());
    }
    return reachedReduce;
  }

  void runOnOperation() override {
    MLIRContext *context = &getContext();
    ModuleOp m = getOperation();
    DenseMap<Operation *, SetVector<Operation *>> ifBodyMap;
    m.walk([&](triton::xpu::StoreOp storeOp) {
      OpBuilder builder(storeOp);
      auto loc = storeOp->getLoc();
      SetVector<Operation *> ifBodyOps;
      if (auto op = findDefOpBwd<triton::xpu::ReduceOp>(storeOp.getValue())) {
        auto reduceOp = cast<triton::xpu::ReduceOp>(op);
        ReduceOpHelper helper(reduceOp);
        auto srcShape = helper.getSrcShape();
        // A last-axis reduce writes a complete result every iteration; an
        // axis-0 reduce accumulates across the slicing iterations through the
        // SM loop result cache, so its store must wait for the last iteration.
        // A rank > 2 middle axis has no guard rule yet.
        if (srcShape.size() > 1 && reduceOp.getAxis() != srcShape.size() - 1 &&
            reduceOp.getAxis() != 0) {
          return;
        }
        if (hasBroadcast(storeOp, op) || !isSameSize(reduceOp, storeOp)) {
          return;
        }
        auto allocaOp = storeOp.getPtr().getDefiningOp();
        for (Operation *user : allocaOp->getUsers()) {
          if (auto lm2gmOp = dyn_cast<triton::xpu::LM2GMOp>(user)) {
            ifBodyOps.insert(storeOp);
            ifBodyOps.insert(lm2gmOp);
          } else if (auto lm2gmOp = dyn_cast<triton::xpu::LM2GMMaskOp>(user)) {
            ifBodyOps.insert(storeOp);
            ifBodyOps.insert(lm2gmOp);
          }
        }
        if (ifBodyOps.empty()) {
          return;
        }
        // Decide before building, and build the core-id chain lazily.
        // TTX_GetCoreIdOp is a hardware read not marked Pure, so DCE cannot
        // remove it; any path that does not consume the chain would leave dead
        // hardware-read IR. Two such paths exist: the axis-0 no-guard case
        // returns below before building anything, and the axis-0 last-iteration
        // guard uses only lastLoopCond (buildCoreCond is never called).
        bool isAxisZero = srcShape.size() > 1 && reduceOp.getAxis() == 0;
        // The running result only lives in the SM loop result cache for a
        // rank-1 (or scalar) result -- the rank the backend gates on. A
        // rank > 1 result keeps one partial per iteration, so it must wait
        // for the last one.
        auto redResTy =
            dyn_cast<RankedTensorType>(reduceOp.getResult()[0].getType());
        bool cacheAccumulates = !redResTy || redResTy.getRank() == 1;
        // A masked load puts the whole slice body -- the reduce included --
        // inside an scf.if.
        bool bodyCanBeMasked =
            reduceOp->getParentOfType<scf::IfOp>() != nullptr;
        // Only the reduce result itself carries the running total across
        // slices; anything that goes through a loop accumulator does not.
        bool storeValueIsReduceResult =
            isReduceResultValue(storeOp.getValue(), reduceOp);
        // An axis-0 reduce returns one value per column, distributed over the
        // cores, so every core writes its own columns and needs no core
        // guard; with an accumulating cache it needs no last-iteration guard
        // either, so there is nothing left to build.
        if (isAxisZero && cacheAccumulates && bodyCanBeMasked &&
            storeValueIsReduceResult)
          return;

        int64_t _groupSize = 64, _groupNum = 1;
        getGroupInfo(reduceOp, _groupSize, _groupNum);
        int64_t _usedCoreNum = _groupSize * _groupNum;
        // Materialise the core-ownership predicate only where it is consumed,
        // so a path that does not use it leaves no dangling GetCoreIdOp chain.
        auto buildCoreCond = [&]() -> Value {
          auto coreId = builder.create<triton::xpu::GetCoreIdOp>(
              loc, builder.getI32Type());
          auto groupSize =
              builder.create<arith::ConstantIntOp>(loc, _groupSize, 32);
          auto coreIdInsideGroup = builder.create<arith::RemSIOp>(
              loc, builder.getI32Type(), coreId, groupSize);
          auto zero = builder.create<arith::ConstantIntOp>(loc, 0, 32);
          auto isCoreId0InsideGroup = builder.create<arith::CmpIOp>(
              loc, builder.getI1Type(), arith::CmpIPredicate::eq,
              coreIdInsideGroup, zero);
          auto usedCoreNum =
              builder.create<arith::ConstantIntOp>(loc, _usedCoreNum, 32);
          auto sltUsedCoreNum = builder.create<arith::CmpIOp>(
              loc, builder.getI1Type(), arith::CmpIPredicate::slt, coreId,
              usedCoreNum);
          return builder.create<arith::AndIOp>(
              loc, builder.getI1Type(), isCoreId0InsideGroup, sltUsedCoreNum);
        };
        Value cond;
        if (srcShape.size() == 1 || isAxisZero) {
          if (cacheAccumulates && bodyCanBeMasked && storeValueIsReduceResult) {
            // isAxisZero is false here: the axis-0 case that needs no guard at
            // all returned above, before any of this IR was built.
            cond = buildCoreCond();
          } else {
            auto loopNum = builder.create<arith::ConstantIntOp>(
                loc, reduceOp.getLoopNum() - 1, 32);
            auto lastLoopCond = builder.create<arith::CmpIOp>(
                loc, builder.getI1Type(), arith::CmpIPredicate::eq,
                reduceOp.getLoopIndex(), loopNum);
            // An axis-0 reduce distributes columns across cores, so it needs
            // only the last-iteration guard, not the core-ownership one.
            cond =
                isAxisZero
                    ? lastLoopCond
                    : builder
                          .create<arith::AndIOp>(loc, builder.getI1Type(),
                                                 buildCoreCond(), lastLoopCond)
                          .getResult();
          }
        } else {
          cond = buildCoreCond();
        }
        auto ifOp = builder.create<scf::IfOp>(loc, cond,
                                              /*withElseRegion=*/false);
        ifBodyMap[ifOp] = ifBodyOps;
        LLVM_DEBUG(llvm::dbgs() << "[StoreControl] GroupSize: " << _groupSize
                                << ", usedCoreNum: " << _usedCoreNum << "\n");
      }
    });
    for (auto &pair : ifBodyMap) {
      auto ifOp = cast<scf::IfOp>(pair.first);
      auto ifBodyOps = pair.second;
      for (auto op : ifBodyOps) {
        op->moveBefore(ifOp.thenBlock()->getTerminator());
      }
    }
  }
};

} // namespace xpu
} // namespace triton
} // namespace mlir
