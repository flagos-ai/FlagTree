/*
 * Copyright (c) Huawei Technologies Co.
 * Licensed under the MIT license.
 */

#include "ascend/include/TritonToHFusion/Passes.h"
#if __has_include("bishengir/Dialect/HFusion/IR/HFusion.h")
#include "bishengir/Dialect/HFusion/IR/HFusion.h"
#endif
#if __has_include("bishengir/Dialect/HFusion/IR/HFusionImpl.h")
#include "bishengir/Dialect/HFusion/IR/HFusionImpl.h"
#endif
#if __has_include("bishengir/Dialect/Tensor/IR/TensorImpl.h")
#include "bishengir/Dialect/Tensor/IR/TensorImpl.h"
#endif
#include "npu/Dialect/TritonAscend/IR/TritonAscendDialect.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/Attributes.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/LogicalResult.h"

namespace mlir {
namespace triton {
#define GEN_PASS_DEF_TRITONTOHFUSION
#include "ascend/include/TritonToHFusion/Passes.h.inc"
} // namespace triton
} // namespace mlir

using namespace mlir;
using namespace hfusion;

namespace {
struct TritonModToHFusionConversion : OpRewritePattern<triton::ascend::ModOp> {
  using OpRewritePattern<triton::ascend::ModOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(triton::ascend::ModOp op,
                                PatternRewriter &rewriter) const final {
    auto lhsType = dyn_cast<RankedTensorType>(op.getLhs().getType());
    auto rhsType = dyn_cast<RankedTensorType>(op.getRhs().getType());
    if (!lhsType || !rhsType) {
      return failure();
    }

    auto emptyTensor = rewriter.create<tensor::EmptyOp>(
        op.getLoc(), lhsType.getShape(), lhsType.getElementType());
    auto newOp =
        hfusion::createBinaryOp<hfusion::ElemwiseBinaryOp, hfusion::BinaryFn,
                                hfusion::BinaryFnAttr>(
            rewriter, op.getLoc(), hfusion::BinaryFn::mod,
            ValueRange({op.getLhs(), op.getRhs()}),
            ValueRange({emptyTensor.getResult()}));

    rewriter.replaceOp(op, newOp->getResult(0));
    return success();
  }
};

struct TritonHistogramToHFusionConversion
    : OpRewritePattern<triton::HistogramOp> {
  using OpRewritePattern<triton::HistogramOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(triton::HistogramOp op,
                                PatternRewriter &rewriter) const final {
    auto loc = op.getLoc();
    Value input = op.getSrc();
    auto resultType = op.getResult().getType();

    int64_t numBins = 256; // 256 is default fallback.
    if (auto rankedTy = dyn_cast<RankedTensorType>(resultType))
      if (rankedTy.hasStaticShape() && rankedTy.getNumElements() > 0)
        numBins = rankedTy.getNumElements();

    auto numBinsAttr = rewriter.getI64IntegerAttr(numBins);

    auto newOp = rewriter.create<hfusion::HistogramOp>(loc, resultType, input,
                                                       numBinsAttr, Value());

    rewriter.replaceOp(op, newOp.getResult());
    return success();
  }
};

struct TritonFpToFpToHFusionConversion : OpRewritePattern<triton::FpToFpOp> {
  using OpRewritePattern<triton::FpToFpOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(triton::FpToFpOp op,
                                PatternRewriter &rewriter) const final {
    auto loc = op.getLoc();
    Value input = op.getSrc();
    auto resultType = op.getResult().getType();

    // Only handle float-to-float conversions with non-RTNE rounding modes
    // RTNE (default) rounding is handled by TritonToLinalg pass using
    // arith.truncf/extf
    auto srcType = cast<TensorType>(input.getType());
    auto dstType = cast<TensorType>(resultType);
    if (!srcType.getElementType().isIntOrFloat() ||
        !dstType.getElementType().isIntOrFloat()) {
      return failure();
    }

    // Check if this has a non-RTNE rounding mode
    auto roundingMode = op.getRounding();
    if (!roundingMode.has_value() ||
        roundingMode.value() == triton::RoundingMode::RTNE) {
      // RTNE or no rounding mode specified: let TritonToLinalg handle it
      return failure();
    }

    // Map non-RTNE rounding modes to HFusion rounding mode

    hfusion::RoundMode hfusionRoundMode;
    switch (roundingMode.value()) {
    case triton::RoundingMode::RTZ:
      hfusionRoundMode = hfusion::RoundMode::TRUNC;
      break;
    default:
      return op.emitError("Unsupported rounding mode for HFusion conversion");
    }

    // Get or create destination tensor (destination-style)
    SmallVector<Value> dsts;
    if (failed(tensor::getOrCreateDestinations(rewriter, loc, op, dsts)))
      return failure();

    // Create the HFusion cast operation with round_mode attribute
    auto roundModeAttr =
        hfusion::RoundModeAttr::get(rewriter.getContext(), hfusionRoundMode);
    auto modeAttr = rewriter.getNamedAttr("mode", roundModeAttr);

    rewriter.replaceOpWithNewOp<hfusion::CastOp>(
        op, ValueRange{input}, ValueRange{dsts}, ArrayRef{modeAttr});

    return success();
  }
};
} // namespace

namespace {
struct TritonToHFusionPass
    : public mlir::triton::impl::TritonToHFusionBase<TritonToHFusionPass> {
  void runOnOperation() override;
};
} // namespace

void TritonToHFusionPass::runOnOperation() {
  auto module = getOperation();

  // Use greedy pattern rewriter for simpler pattern matching
  // Patterns decide themselves whether to convert (via returning
  // success/failure) success/failure)
  RewritePatternSet patterns(&getContext());
  patterns.add<TritonHistogramToHFusionConversion>(patterns.getContext());
  patterns.add<TritonFpToFpToHFusionConversion>(patterns.getContext());
  patterns.add<TritonModToHFusionConversion>(patterns.getContext());

  // Apply patterns with greedy rewriting
  // This allows patterns to return failure() without causing pass failure
  if (failed(applyPatternsAndFoldGreedily(module, std::move(patterns)))) {
    signalPassFailure();
  }
}

std::unique_ptr<OperationPass<ModuleOp>>
mlir::triton::createTritonToHFusionPass() {
  return std::make_unique<TritonToHFusionPass>();
}
