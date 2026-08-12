//===----------------------------------------------------------------------===//
//
// SPDX-FileCopyrightText: Copyright (c) 2025 SpacemiT. All rights reserved.
// SPDX-License-Identifier: MIT
//
// TLE → Linalg/Tensor lowering patterns:
//   xtle.extract_tile → tensor.extract_slice
//   xtle.insert_tile  → tensor.insert_slice
//
//===----------------------------------------------------------------------===//

#include "triton-shared/Conversion/TLEToLinalg/TLEToLinalg.h"
#include "triton-shared/Dialect/TLE/IR/TLEDialect.h"
#include "triton-shared/Dialect/TLE/IR/TLEOps.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#define DEBUG_TYPE "tle-to-linalg"

using namespace mlir;

namespace {

// ============================================================================
// Helper: delinearize a tile index into per-dimension offsets.
//
// Given:
//   srcShape  = [S0, S1, ..., Sn]
//   tileShape = [T0, T1, ..., Tn]
//   grid      = [S0/T0, S1/T1, ..., Sn/Tn]
//
// For a linear tile index `idx`, compute:
//   offsets[i] = (idx / stride_i % grid[i]) * tileShape[i]
//
// where stride_i = product(grid[i+1:]).
// ============================================================================
SmallVector<OpFoldResult> delinearizeIndex(Location loc, Value index,
                                           ArrayRef<int64_t> srcShape,
                                           ArrayRef<int64_t> tileShape,
                                           PatternRewriter &rewriter) {
  int64_t rank = srcShape.size();
  SmallVector<int64_t> grid(rank);
  for (int64_t i = 0; i < rank; ++i)
    grid[i] = srcShape[i] / tileShape[i];

  // Try to extract a static constant from the index value.
  std::optional<int64_t> staticIdx;
  if (auto constOp = index.getDefiningOp<arith::ConstantOp>()) {
    if (auto intAttr = dyn_cast<IntegerAttr>(constOp.getValue()))
      staticIdx = intAttr.getInt();
  }

  SmallVector<OpFoldResult> offsets;
  if (staticIdx.has_value()) {
    // Static path: compute offsets at compile time.
    int64_t remain = *staticIdx;
    for (int64_t i = 0; i < rank; ++i) {
      int64_t stride = 1;
      for (int64_t j = i + 1; j < rank; ++j)
        stride *= grid[j];
      int64_t tileIdx = remain / stride;
      remain %= stride;
      offsets.push_back(rewriter.getIndexAttr(tileIdx * tileShape[i]));
    }
  } else {
    // Dynamic path: emit arith ops to compute offsets at runtime.
    // Ensure index is index type.
    if (!index.getType().isIndex())
      index = arith::IndexCastOp::create(rewriter, loc, rewriter.getIndexType(),
                                         index);
    for (int64_t i = 0; i < rank; ++i) {
      int64_t stride = 1;
      for (int64_t j = i + 1; j < rank; ++j)
        stride *= grid[j];

      // tileIdx = (index / stride) % grid[i]
      Value tileIdx = index;
      if (stride != 1) {
        Value strideVal = arith::ConstantIndexOp::create(rewriter, loc, stride);
        tileIdx = arith::DivUIOp::create(rewriter, loc, tileIdx, strideVal);
      }
      if (grid[i] != 1) {
        // Only need modulo if not the first dimension (or grid > 1)
        Value gridVal = arith::ConstantIndexOp::create(rewriter, loc, grid[i]);
        tileIdx = arith::RemUIOp::create(rewriter, loc, tileIdx, gridVal);
      }
      // offset = tileIdx * tileShape[i]
      Value tileSizeVal =
          arith::ConstantIndexOp::create(rewriter, loc, tileShape[i]);
      Value offset = arith::MulIOp::create(rewriter, loc, tileIdx, tileSizeVal);
      offsets.push_back(offset);
    }
  }
  return offsets;
}

// ============================================================================
// ExtractTileOp → tensor.extract_slice
// ============================================================================
struct ExtractTileOpPattern
    : public OpRewritePattern<mlir::tle::ExtractTileOp> {
  using OpRewritePattern<mlir::tle::ExtractTileOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(mlir::tle::ExtractTileOp op,
                                PatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    Value src = op.getSrc();
    Value index = op.getIndex();

    auto srcTy = cast<RankedTensorType>(src.getType());
    auto dstTy = cast<RankedTensorType>(op.getResult().getType());
    auto srcShape = srcTy.getShape();
    auto tileShape = dstTy.getShape();
    int64_t rank = srcShape.size();

    // Compute per-dimension offsets from the linear tile index.
    SmallVector<OpFoldResult> offsets =
        delinearizeIndex(loc, index, srcShape, tileShape, rewriter);

    // Sizes = tileShape (all static).
    SmallVector<OpFoldResult> sizes;
    for (int64_t i = 0; i < rank; ++i)
      sizes.push_back(rewriter.getIndexAttr(tileShape[i]));

    // Strides = all 1.
    SmallVector<OpFoldResult> strides(rank, rewriter.getIndexAttr(1));

    auto extractSlice = tensor::ExtractSliceOp::create(
        rewriter, loc, dstTy, src, offsets, sizes, strides);

    rewriter.replaceOp(op, extractSlice.getResult());
    return success();
  }
};

// ============================================================================
// InsertTileOp → tensor.insert_slice
// ============================================================================
struct InsertTileOpPattern : public OpRewritePattern<mlir::tle::InsertTileOp> {
  using OpRewritePattern<mlir::tle::InsertTileOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(mlir::tle::InsertTileOp op,
                                PatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    Value src = op.getSrc();
    Value tile = op.getTile();
    Value index = op.getIndex();

    auto srcTy = cast<RankedTensorType>(src.getType());
    auto tileTy = cast<RankedTensorType>(tile.getType());
    auto srcShape = srcTy.getShape();
    auto tileShape = tileTy.getShape();
    int64_t rank = srcShape.size();

    // Compute per-dimension offsets from the linear tile index.
    SmallVector<OpFoldResult> offsets =
        delinearizeIndex(loc, index, srcShape, tileShape, rewriter);

    // Sizes = tileShape (all static).
    SmallVector<OpFoldResult> sizes;
    for (int64_t i = 0; i < rank; ++i)
      sizes.push_back(rewriter.getIndexAttr(tileShape[i]));

    // Strides = all 1.
    SmallVector<OpFoldResult> strides(rank, rewriter.getIndexAttr(1));

    auto insertSlice = tensor::InsertSliceOp::create(rewriter, loc, tile, src,
                                                     offsets, sizes, strides);

    rewriter.replaceOp(op, insertSlice.getResult());
    return success();
  }
};

} // namespace

// ============================================================================
// Pattern registration
// ============================================================================
void mlir::triton::populateTLEToLinalgConversionPatterns(
    RewritePatternSet &patterns) {
  patterns.add<ExtractTileOpPattern>(patterns.getContext());
  patterns.add<InsertTileOpPattern>(patterns.getContext());
}
