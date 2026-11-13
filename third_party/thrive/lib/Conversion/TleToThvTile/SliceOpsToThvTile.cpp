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

// Delinearize a linear tile index into per-dimension offsets.
// Row-major order: last dim is fastest-varying.
//   offsets[i] = tileShape[i] * ((index / stride_{i+1}) % gridShape[i])
//
// If index is a constant, offsets are computed at compile time.
static SmallVector<Value>
delinearizeIndex(Location loc, Value index, ArrayRef<int64_t> srcShape,
                 ArrayRef<int64_t> tileShape,
                 ConversionPatternRewriter &rewriter) {
  int64_t rank = srcShape.size();
  auto indexType = rewriter.getIndexType();

  // Compute grid shape: gridShape[i] = srcShape[i] / tileShape[i]
  SmallVector<int64_t> gridShape(rank);
  for (int64_t i = 0; i < rank; ++i)
    gridShape[i] = srcShape[i] / tileShape[i];

  // Static path: index is a constant
  if (auto constOp = index.getDefiningOp<arith::ConstantOp>()) {
    int64_t idx = cast<IntegerAttr>(constOp.getValue()).getInt();
    SmallVector<int64_t> tileIndices(rank);
    int64_t remain = idx;
    for (int i = static_cast<int>(rank) - 1; i >= 0; --i) {
      tileIndices[i] = remain % gridShape[i];
      remain /= gridShape[i];
    }
    SmallVector<Value> offsets;
    for (int64_t i = 0; i < rank; ++i) {
      int64_t off = tileIndices[i] * tileShape[i];
      offsets.push_back(rewriter.create<arith::ConstantOp>(
          loc, indexType, rewriter.getIndexAttr(off)));
    }
    return offsets;
  }

  // Dynamic path: emit runtime delinearization
  Value idx = index;
  if (idx.getType() != indexType)
    idx = rewriter.create<arith::IndexCastOp>(loc, indexType, idx);

  // Compute offsets from right to left, then reverse
  SmallVector<Value> offsets(rank);
  Value remaining = idx;
  for (int i = static_cast<int>(rank) - 1; i >= 0; --i) {
    Value gridVal = rewriter.create<arith::ConstantOp>(
        loc, indexType, rewriter.getIndexAttr(gridShape[i]));
    Value tileVal = rewriter.create<arith::ConstantOp>(
        loc, indexType, rewriter.getIndexAttr(tileShape[i]));

    Value coord = rewriter.create<arith::RemUIOp>(loc, remaining, gridVal);
    remaining = rewriter.create<arith::DivUIOp>(loc, remaining, gridVal);
    offsets[i] = rewriter.create<arith::MulIOp>(loc, coord, tileVal);
  }

  return offsets;
}

// Convert TLE ExtractTileOp to ThvTile ExtractSliceOp.
struct ConvertExtractTilePattern
    : public OpConversionPattern<tle::ExtractTileOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(tle::ExtractTileOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto srcType = cast<RankedTensorType>(op.getSrc().getType());
    auto resultType = cast<RankedTensorType>(op.getResult().getType());
    int64_t rank = resultType.getRank();

    auto tileShapeAttr = op->getAttrOfType<DenseI64ArrayAttr>("tile_shape");
    if (!tileShapeAttr)
      return op.emitError("missing tile_shape attribute");
    SmallVector<int64_t> tileShape(tileShapeAttr.asArrayRef());

    SmallVector<int64_t> staticSizes(rank);
    SmallVector<int64_t> staticStrides(rank, 1);
    for (int64_t i = 0; i < rank; ++i)
      staticSizes[i] = resultType.getDimSize(i);

    SmallVector<Value> offsets =
        delinearizeIndex(op.getLoc(), adaptor.getIndex(), srcType.getShape(),
                         tileShape, rewriter);

    rewriter.replaceOpWithNewOp<thvtile::ExtractSliceOp>(
        op, resultType, adaptor.getSrc(), offsets,
        rewriter.getDenseI64ArrayAttr(staticSizes),
        rewriter.getDenseI64ArrayAttr(staticStrides));

    return success();
  }
};

// Convert TLE InsertTileOp to ThvTile InsertSliceOp.
struct ConvertInsertTilePattern
    : public OpConversionPattern<tle::InsertTileOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(tle::InsertTileOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto srcType = cast<RankedTensorType>(op.getSrc().getType());
    auto tileType = cast<RankedTensorType>(op.getTile().getType());
    auto resultType = cast<RankedTensorType>(op.getResult().getType());
    int64_t rank = resultType.getRank();

    SmallVector<int64_t> tileShape(rank);
    SmallVector<int64_t> staticSizes(rank);
    SmallVector<int64_t> staticStrides(rank, 1);
    for (int64_t i = 0; i < rank; ++i) {
      tileShape[i] = tileType.getDimSize(i);
      staticSizes[i] = tileType.getDimSize(i);
    }

    SmallVector<Value> offsets =
        delinearizeIndex(op.getLoc(), adaptor.getIndex(), srcType.getShape(),
                         tileShape, rewriter);

    rewriter.replaceOpWithNewOp<thvtile::InsertSliceOp>(
        op, resultType, adaptor.getTile(), adaptor.getSrc(), offsets,
        rewriter.getDenseI64ArrayAttr(staticSizes),
        rewriter.getDenseI64ArrayAttr(staticStrides));

    return success();
  }
};

} // namespace

void populateSliceOpsToThvTilePatterns(TleToThvTileTypeConverter &typeConverter,
                                       RewritePatternSet &patterns) {
  MLIRContext *ctx = patterns.getContext();
  patterns.add<ConvertExtractTilePattern, ConvertInsertTilePattern>(
      typeConverter, ctx);
}

} // namespace mlir::thvtile
