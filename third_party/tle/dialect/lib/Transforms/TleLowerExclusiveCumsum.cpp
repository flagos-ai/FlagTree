// MIT License
//
// Copyright (c) 2025 The FlagOS Contributors
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.

#include "tle/dialect/include/IR/Dialect.h"
#include "tle/dialect/include/Transforms/Passes.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "triton/Dialect/Triton/IR/Dialect.h"

#include "llvm/ADT/SmallVector.h"
#include <limits>

namespace mlir::triton::tle {

#define GEN_PASS_DEF_TRITONTLELOWEREXCLUSIVECUMSUM
#include "tle/dialect/include/Transforms/Passes.h.inc"

namespace {

static Value createAddOp(OpBuilder &builder, Location loc, Value lhs, Value rhs,
                         Type elemTy) {
  if (isa<FloatType>(elemTy))
    return builder.create<arith::AddFOp>(loc, lhs, rhs).getResult();
  if (elemTy.isIntOrIndex())
    return builder.create<arith::AddIOp>(loc, lhs, rhs).getResult();
  return nullptr;
}

static Value createSubOp(OpBuilder &builder, Location loc, Value lhs, Value rhs,
                         Type elemTy) {
  if (isa<FloatType>(elemTy))
    return builder.create<arith::SubFOp>(loc, lhs, rhs).getResult();
  if (elemTy.isIntOrIndex())
    return builder.create<arith::SubIOp>(loc, lhs, rhs).getResult();
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
  builder.create<triton::ScanReturnOp>(loc, ValueRange{sum});
  return success();
}

static LogicalResult buildReduceSelectByIndexRegion(OpBuilder &builder,
                                                    triton::ReduceOp reduce,
                                                    Type elemTy, Location loc,
                                                    bool pickMaxIndex) {
  OpBuilder::InsertionGuard guard(builder);
  Block *block = builder.createBlock(&reduce.getCombineOp());
  Type idxTy = builder.getI32Type();
  // Reduce with 2 operands: (idx, value). Region argument order is:
  // (lhs_idx, lhs_val, rhs_idx, rhs_val).
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
  Value chooseLhs = builder.create<arith::CmpIOp>(loc, pred, lhsIdx, rhsIdx);
  Value selectedIdx =
      builder.create<arith::SelectOp>(loc, chooseLhs, lhsIdx, rhsIdx);
  Value selectedVal =
      builder.create<arith::SelectOp>(loc, chooseLhs, lhsVal, rhsVal);
  builder.create<triton::ReduceReturnOp>(loc,
                                         ValueRange{selectedIdx, selectedVal});
  return success();
}

class LowerExclusiveCumsumPass
    : public impl::TritonTleLowerExclusiveCumsumBase<LowerExclusiveCumsumPass> {
  void runOnOperation() override {
    ModuleOp module = getOperation();
    SmallVector<tle::ExclusiveCumsumOp> ops;
    module.walk([&](tle::ExclusiveCumsumOp op) { ops.push_back(op); });

    for (tle::ExclusiveCumsumOp op : ops) {
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

      auto scan = builder.create<triton::ScanOp>(
          op.getLoc(), ValueRange{op.getSrc()}, static_cast<int>(op.getAxis()),
          op.getReverse());
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
      Value indices = builder
                          .create<triton::MakeRangeOp>(
                              op.getLoc(), idxTy, /*start=*/0u,
                              /*end=*/static_cast<uint32_t>(axisExtent))
                          .getResult();
      auto reduce = builder.create<triton::ReduceOp>(
          op.getLoc(), ValueRange{indices, inclusive},
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
} // namespace mlir::triton::tle
