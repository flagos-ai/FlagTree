#ifdef __TLE__

#include "Dialect/MUSATLE/IR/Dialect.h"
#include "TritonMUSAGPUTransforms/Passes.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "triton/Dialect/Triton/IR/Dialect.h"

#include "llvm/ADT/SmallVector.h"
#include <cstdint>
#include <limits>

namespace mlir {

#define GEN_PASS_DEF_TRITONMUSAGPUTLELOWEREXCLUSIVECUMSUM
#include "TritonMUSAGPUTransforms/Passes.h.inc"

namespace {

static Value createAddOp(OpBuilder &builder, Location loc, Value lhs, Value rhs,
                         Type elemTy) {
  if (isa<FloatType>(elemTy))
    return arith::AddFOp::create(builder, loc, lhs, rhs).getResult();
  if (elemTy.isIntOrIndex())
    return arith::AddIOp::create(builder, loc, lhs, rhs).getResult();
  return nullptr;
}

static Value createSubOp(OpBuilder &builder, Location loc, Value lhs, Value rhs,
                         Type elemTy) {
  if (isa<FloatType>(elemTy))
    return arith::SubFOp::create(builder, loc, lhs, rhs).getResult();
  if (elemTy.isIntOrIndex())
    return arith::SubIOp::create(builder, loc, lhs, rhs).getResult();
  return nullptr;
}

static LogicalResult buildScanAddRegion(OpBuilder &builder, triton::ScanOp scan,
                                        Type elemTy, Location loc) {
  OpBuilder::InsertionGuard guard(builder);
  Block *block = builder.createBlock(&scan.getCombineOp());
  block->addArgument(elemTy, loc);
  block->addArgument(elemTy, loc);
  builder.setInsertionPointToEnd(block);
  Value sum = createAddOp(builder, loc, block->getArgument(0),
                          block->getArgument(1), elemTy);
  if (!sum)
    return failure();
  triton::ScanReturnOp::create(builder, loc, ValueRange{sum});
  return success();
}

static LogicalResult buildReduceSelectByIndexRegion(OpBuilder &builder,
                                                    triton::ReduceOp reduce,
                                                    Type elemTy, Location loc,
                                                    bool pickMaxIndex) {
  OpBuilder::InsertionGuard guard(builder);
  Block *block = builder.createBlock(&reduce.getCombineOp());
  Type idxTy = builder.getI32Type();
  block->addArgument(idxTy, loc);
  block->addArgument(elemTy, loc);
  block->addArgument(idxTy, loc);
  block->addArgument(elemTy, loc);
  builder.setInsertionPointToEnd(block);

  Value lhsIdx = block->getArgument(0);
  Value lhsVal = block->getArgument(1);
  Value rhsIdx = block->getArgument(2);
  Value rhsVal = block->getArgument(3);

  arith::CmpIPredicate pred =
      pickMaxIndex ? arith::CmpIPredicate::sgt : arith::CmpIPredicate::slt;
  Value chooseLhs = arith::CmpIOp::create(builder, loc, pred, lhsIdx, rhsIdx);
  Value selectedIdx =
      arith::SelectOp::create(builder, loc, chooseLhs, lhsIdx, rhsIdx);
  Value selectedVal =
      arith::SelectOp::create(builder, loc, chooseLhs, lhsVal, rhsVal);
  triton::ReduceReturnOp::create(builder, loc,
                                 ValueRange{selectedIdx, selectedVal});
  return success();
}

class LowerExclusiveCumsumPass
    : public impl::TritonMUSAGPUTLELowerExclusiveCumsumBase<
          LowerExclusiveCumsumPass> {
  void runOnOperation() override {
    ModuleOp module = getOperation();
    SmallVector<triton::musa_tle::ExclusiveCumsumOp> ops;
    module.walk(
        [&](triton::musa_tle::ExclusiveCumsumOp op) { ops.push_back(op); });

    for (triton::musa_tle::ExclusiveCumsumOp op : ops) {
      if (!op)
        continue;

      auto srcTy = dyn_cast<RankedTensorType>(op.getSrc().getType());
      if (!srcTy) {
        op.emitOpError("expects ranked tensor input");
        signalPassFailure();
        return;
      }
      int64_t axisExtent = srcTy.getShape()[0];
      if (ShapedType::isDynamic(axisExtent) || axisExtent <= 0) {
        op.emitOpError("expects static, positive axis extent");
        signalPassFailure();
        return;
      }
      if (axisExtent >
          static_cast<int64_t>(std::numeric_limits<uint32_t>::max())) {
        op.emitOpError("axis extent is too large for tt.make_range");
        signalPassFailure();
        return;
      }

      const Type elemTy = srcTy.getElementType();
      OpBuilder builder(op);

      auto scan = triton::ScanOp::create(
          builder, op.getLoc(), ValueRange{op.getSrc()},
          static_cast<int>(op.getAxis()), op.getReverse());
      if (failed(buildScanAddRegion(builder, scan, elemTy, op.getLoc()))) {
        op.emitOpError("failed to build add combiner for triton.scan");
        signalPassFailure();
        return;
      }
      Value inclusive = scan.getResult()[0];
      Value exclusive =
          createSubOp(builder, op.getLoc(), inclusive, op.getSrc(), elemTy);
      if (!exclusive) {
        op.emitOpError("unsupported element type for exclusive subtraction");
        signalPassFailure();
        return;
      }

      RankedTensorType idxTy = RankedTensorType::get(
          srcTy.getShape(), builder.getI32Type(), srcTy.getEncoding());
      Value indices =
          triton::MakeRangeOp::create(builder, op.getLoc(), idxTy, /*start=*/0u,
                                      /*end=*/static_cast<uint32_t>(axisExtent))
              .getResult();
      auto reduce = triton::ReduceOp::create(builder, op.getLoc(),
                                             ValueRange{indices, inclusive},
                                             static_cast<int>(op.getAxis()));
      bool pickMaxIndex = !op.getReverse();
      if (failed(buildReduceSelectByIndexRegion(builder, reduce, elemTy,
                                                op.getLoc(), pickMaxIndex))) {
        op.emitOpError(
            "failed to build index-select combiner for triton.reduce");
        signalPassFailure();
        return;
      }
      Value total = reduce.getResult()[1];

      if (exclusive.getType() != op.getExclusive().getType() ||
          total.getType() != op.getTotal().getType()) {
        op.emitOpError("lowered value types do not match op result types");
        signalPassFailure();
        return;
      }

      op.getExclusive().replaceAllUsesWith(exclusive);
      op.getTotal().replaceAllUsesWith(total);
      op.erase();
    }
  }
};

} // namespace
} // namespace mlir

#endif // __TLE__
