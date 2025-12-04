//===------------------- LinalgTiling.cpp --------------------------------===//
//
// Copyright (C) 2020-2025 Terapines Technology (Wuhan) Co., Ltd
// All rights reserved.
//
//===----------------------------------------------------------------------===//
//
// This file implements the patterns to tile linalg operations for better
// performance. It applies tiling transformations to improve data locality
// and parallelism, focusing on operations like linalg.reduce and
// linalg.generic.
//
//===----------------------------------------------------------------------===//

#include "tsingmicro-tx81/Conversion/LinalgTiling/LinalgTiling.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/Transforms/Transforms.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "llvm/Support/Debug.h"

#define DEBUG_TYPE "linalg-tiling"

using namespace mlir;

namespace {

// Extract the operations from a linalg op region
template <typename T> llvm::SmallVector<Operation *> getRegionOps(T linalgOp) {
  auto regionBlock = linalgOp.getBody();
  return llvm::map_to_vector(regionBlock->without_terminator(),
                             [](Operation &op) { return &op; });
}

struct TilingReduceRewrite : public OpRewritePattern<linalg::ReduceOp> {
  TilingReduceRewrite(MLIRContext *context)
      : OpRewritePattern<linalg::ReduceOp>(context, /*benefit=*/1) {}
  using OpRewritePattern<linalg::ReduceOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(linalg::ReduceOp op,
                                PatternRewriter &rewriter) const override {
    auto dims = op.getDimensions();
    if (dims.size() != 1) {
      op->emitError() << "Only support one dim reduce.";
      return rewriter.notifyMatchFailure(op, "Only support one dim reduce.");
    }

    auto dim = dims[0];
    auto inputType = cast<MemRefType>(op.getInputs()[0].getType());
    auto inputShape = inputType.getShape();

    // Tiling if shape[dim]>32768
    if (inputShape[dim] < 32768) {
      return failure();
    }

    linalg::LinalgTilingOptions tilingOptions;
    auto tileSizes = SmallVector<int64_t>(inputShape);
    assert(dim == 0 && "Expected tiling on the first dimension");
    assert(llvm::isPowerOf2_64(inputShape[dim]) &&
           "Expected power of 2 for tiling size");
    tileSizes[dim] = 16384;
    tilingOptions.setTileSizes(tileSizes);

    if (failed(linalg::tileLinalgOp(rewriter, op, tilingOptions)))
      return rewriter.notifyMatchFailure(op, "operation not supported yet.");

    rewriter.eraseOp(op);
    return success();
  }
};

} // namespace

void mlir::triton::populateLinalgTilingPatterns(RewritePatternSet &patterns) {
  patterns.add<TilingReduceRewrite>(patterns.getContext());
}
