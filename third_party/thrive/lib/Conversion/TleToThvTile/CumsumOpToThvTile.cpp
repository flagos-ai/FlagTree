#include "Conversion/TleToThvTile/TleToThvTilePatterns.h"

#include "Conversion/TleToThvTile/TypeConverter.h"
#include "Dialect/ThvTile/IR/Dialect.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Transforms/DialectConversion.h"
#include "tle/dialect/include/IR/Dialect.h"

using namespace mlir;
using namespace mlir::triton;

namespace mlir::thvtile {

namespace {

// Build the associative combine region for thvtile.scan: add.
static LogicalResult buildScanAddRegion(OpBuilder &builder,
                                        thvtile::ScanOp scan, Type elemTy,
                                        Location loc) {
  OpBuilder::InsertionGuard guard(builder);
  Block *block = builder.createBlock(&scan.getCombineOp());
  block->addArgument(elemTy, loc);
  block->addArgument(elemTy, loc);
  builder.setInsertionPointToEnd(block);

  Value sum;
  if (isa<FloatType>(elemTy))
    sum = builder.create<arith::AddFOp>(loc, block->getArgument(0),
                                        block->getArgument(1));
  else if (elemTy.isIntOrIndex())
    sum = builder.create<arith::AddIOp>(loc, block->getArgument(0),
                                        block->getArgument(1));
  else
    return failure();

  builder.create<thvtile::ScanReturnOp>(loc, ValueRange{sum});
  return success();
}

// Convert tle.exclusive_cumsum to thvtile.scan {exclusive=true,
// return_total=true}.
//
//   tle.exclusive_cumsum(input) {axis, reverse}
//   -> exclusive : tensor<NxT>, total : T  (scalar)
//
//   Becomes:
//   %exclusive, %total = thvtile.scan(input)
//   {axis, reverse, exclusive=true, return_total=true}
//   : tensor<NxT> -> (tensor<NxT>, T)
struct ConvertExclusiveCumsumPattern
    : public OpConversionPattern<tle::ExclusiveCumsumOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(tle::ExclusiveCumsumOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto srcType = cast<RankedTensorType>(op.getSrc().getType());
    Type elemTy = srcType.getElementType();
    Location loc = op.getLoc();

    // tle.exclusive_cumsum currently only supports rank-1 axis=0 (enforced by
    // its verifier).

    if (!isa<FloatType>(elemTy) && !elemTy.isIntOrIndex())
      return op.emitError("unsupported element type for cumsum");

    // thvtile.scan(input) {axis, reverse, exclusive=true, return_total=true}
    //   totalTypes={elemTy} (scalar, directly matches tle's scalar total).
    auto scanOp = rewriter.create<thvtile::ScanOp>(
        loc,
        /*resultTypes=*/TypeRange{srcType},
        /*totalTypes=*/TypeRange{elemTy},
        /*inputs=*/ValueRange{adaptor.getSrc()},
        /*axis=*/op.getAxisAttr(),
        /*reverse=*/op.getReverseAttr(),
        /*exclusive=*/rewriter.getBoolAttr(true),
        /*return_total=*/rewriter.getBoolAttr(true));

    if (failed(buildScanAddRegion(rewriter, scanOp, elemTy, loc)))
      return op.emitError("failed to build add combiner for thvtile.scan");

    rewriter.replaceOp(op, {scanOp.getResult()[0], scanOp.getTotal()[0]});
    return success();
  }
};

} // namespace

void populateCumsumOpToThvTilePatterns(TleToThvTileTypeConverter &typeConverter,
                                       RewritePatternSet &patterns) {
  MLIRContext *ctx = patterns.getContext();
  patterns.add<ConvertExclusiveCumsumPattern>(typeConverter, ctx);
}

} // namespace mlir::thvtile
