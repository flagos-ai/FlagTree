#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "tle/dialect/include/IR/Dialect.h"
#include "tle/dialect/include/Transforms/Passes.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "triton/Dialect/TritonNvidiaGPU/IR/Dialect.h"

namespace mlir::triton::tle {

namespace ttg = mlir::triton::gpu;
namespace ttng = mlir::triton::nvidia_gpu;

#define GEN_PASS_DEF_TRITONTLELOWERBARRIERS
#include "tle/dialect/include/Transforms/Passes.h.inc"

namespace {

static int64_t getI32Attr(Operation *op, StringRef name) {
  return op->getAttrOfType<IntegerAttr>(name).getInt();
}

static ttg::MemDescType getBarrierSlotType(ttg::MemDescType arrayTy) {
  auto context = arrayTy.getContext();
  auto ctaLayout = ttg::CTAEncodingAttr::getDefault(context, 1);
  Attribute slotEncoding =
      ttg::SwizzledSharedEncodingAttr::get(context, 1, 1, 1, {0}, ctaLayout);
  return ttg::MemDescType::get({1}, arrayTy.getElementType(), slotEncoding,
                               arrayTy.getMemorySpace(),
                               arrayTy.getMutableMemory());
}

static Value createBarrierSlot(OpBuilder &builder, Location loc, Value array,
                               int64_t index) {
  auto arrayTy = cast<ttg::MemDescType>(array.getType());
  auto slotTy = getBarrierSlotType(arrayTy);
  Value idx = builder.create<arith::ConstantIntOp>(loc, index, 32);
  return builder.create<ttg::MemDescIndexOp>(loc, slotTy, array, idx);
}

#if !defined(__HCU__)
static std::pair<Value, Value>
createNamedBarrierOperands(OpBuilder &builder, Location loc, Operation *op) {
  Value id =
      builder.create<arith::ConstantIntOp>(loc, getI32Attr(op, "named_id"), 32);
  Value threads = builder.create<arith::ConstantIntOp>(
      loc, getI32Attr(op, "named_num_threads"), 32);
  return {id, threads};
}
#endif

struct TritonTleLowerBarriers
    : public impl::TritonTleLowerBarriersBase<TritonTleLowerBarriers> {
  void runOnOperation() override {
    ModuleOp module = getOperation();

    SmallVector<BarrierWaitOp> waits;
    SmallVector<BarrierArriveOp> arrives;
    SmallVector<BarrierAllocOp> allocs;
    module.walk([&](Operation *op) {
      if (auto wait = dyn_cast<BarrierWaitOp>(op))
        waits.push_back(wait);
      else if (auto arrive = dyn_cast<BarrierArriveOp>(op))
        arrives.push_back(arrive);
      else if (auto alloc = dyn_cast<BarrierAllocOp>(op))
        allocs.push_back(alloc);
    });

    for (BarrierWaitOp op : waits) {
      OpBuilder builder(op);
      Location loc = op.getLoc();
      StringRef backend = op->getAttrOfType<StringAttr>("backend").getValue();
      if (backend == "mbarrier") {
        builder.create<ttng::WaitBarrierOp>(loc, op.getBarrier(),
                                            op.getPhase());
      } else {
#if defined(__HCU__)
        op.emitOpError("named barrier lowering is only supported on NVIDIA "
                       "backend");
        signalPassFailure();
        return;
#else
        auto [id, threads] =
            createNamedBarrierOperands(builder, loc, op.getOperation());
        builder.create<ttng::NamedBarrierWaitOp>(loc, id, threads);
#endif
      }
      op.erase();
    }

    for (BarrierArriveOp op : arrives) {
      OpBuilder builder(op);
      Location loc = op.getLoc();
      StringRef backend = op->getAttrOfType<StringAttr>("backend").getValue();
      if (backend == "mbarrier") {
        int64_t count = getI32Attr(op.getOperation(), "arrive_count");
        builder.create<ttng::ArriveBarrierOp>(loc, op.getBarrier(),
                                              static_cast<uint32_t>(count));
      } else {
#if defined(__HCU__)
        op.emitOpError("named barrier lowering is only supported on NVIDIA "
                       "backend");
        signalPassFailure();
        return;
#else
        auto [id, threads] =
            createNamedBarrierOperands(builder, loc, op.getOperation());
        builder.create<ttng::NamedBarrierArriveOp>(loc, id, threads);
#endif
      }
      op.erase();
    }

    for (BarrierAllocOp op : allocs) {
      OpBuilder builder(op);
      Location loc = op.getLoc();
      auto arrayTy = op.getResult().getType();
      bool onlyUnusedViews = true;
      SmallVector<ttg::MemDescIndexOp> deadViews;
      for (OpOperand &use : op.getResult().getUses()) {
        auto view = dyn_cast<ttg::MemDescIndexOp>(use.getOwner());
        if (!view || !view->use_empty()) {
          onlyUnusedViews = false;
          break;
        }
        deadViews.push_back(view);
      }
      if (onlyUnusedViews) {
        for (auto view : deadViews)
          view.erase();
        op.erase();
        continue;
      }

      Value alloc = builder.create<ttg::LocalAllocOp>(loc, arrayTy);
      int64_t numBarriers = getI32Attr(op.getOperation(), "num_barriers");
      int64_t arriveCount = getI32Attr(op.getOperation(), "arrive_count");
      for (int64_t i = 0; i < numBarriers; ++i) {
        Value slot = createBarrierSlot(builder, loc, alloc, i);
        builder.create<ttng::InitBarrierOp>(loc, slot,
                                            static_cast<uint32_t>(arriveCount));
      }
      op.getResult().replaceAllUsesWith(alloc);
      op.erase();
    }
  }
};

} // namespace

} // namespace mlir::triton::tle
