/*
 * Copyright (c) Huawei Technologies Co.
 * Licensed under the MIT license.
 */

#include "TritonToHFusion/Passes.h"

#include "bishengir/Dialect/HFusion/IR/HFusion.h"
#include "bishengir/Dialect/HFusion/IR/HFusionImpl.h"
#include "bishengir/Dialect/Tensor/IR/TensorImpl.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/Attributes.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "Dialect/TritonAscend/IR/TritonAscendDialect.h"
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
  struct TritonModToHFusionConversion
    : OpRewritePattern<triton::ascend::ModOp> {
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

      auto newOp = rewriter.create<hfusion::HistogramOp>(
        loc, resultType, input, numBinsAttr, Value());

      rewriter.replaceOp(op, newOp.getResult());
      return success();
    }
  };

  struct TritonFpToFpToHFusionConversion
    : OpRewritePattern<triton::FpToFpOp> {
    using OpRewritePattern<triton::FpToFpOp>::OpRewritePattern;

    LogicalResult matchAndRewrite(triton::FpToFpOp op,
      PatternRewriter &rewriter) const final {
      auto loc = op.getLoc();
      Value input = op.getSrc();
      auto resultType = op.getResult().getType();

      // Only handle float-to-float conversions with non-RTNE rounding modes
      // RTNE (default) rounding is handled by TritonToLinalg pass using arith.truncf/extf
      auto srcType = cast<TensorType>(input.getType());
      auto dstType = cast<TensorType>(resultType);
      if (!srcType.getElementType().isIntOrFloat() ||
          !dstType.getElementType().isIntOrFloat()) {
        return failure();
      }

      // Check if this has a non-RTNE rounding mode
      auto roundingMode = op.getRounding();
      if (!roundingMode.has_value() || roundingMode.value() == triton::RoundingMode::RTNE) {
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
      // Note: Only RTZ (and potential future non-RTNE modes) reach here

      // Get or create destination tensor (destination-style)
      SmallVector<Value> dsts;
      if (failed(tensor::getOrCreateDestinations(rewriter, loc, op, dsts)))
        return failure();

      // Create the HFusion cast operation with round_mode attribute
      auto roundModeAttr = hfusion::RoundModeAttr::get(
          rewriter.getContext(), hfusionRoundMode);
      auto modeAttr = rewriter.getNamedAttr("mode", roundModeAttr);

      rewriter.replaceOpWithNewOp<hfusion::CastOp>(
        op, ValueRange{input}, ValueRange{dsts}, ArrayRef{modeAttr});

      return success();
    }
  };

  struct TritonConv1dToHFusionConversion
    : OpRewritePattern<triton::ascend::Conv1dOp> {
    using OpRewritePattern<triton::ascend::Conv1dOp>::OpRewritePattern;

    LogicalResult matchAndRewrite(triton::ascend::Conv1dOp op,
      PatternRewriter &rewriter) const final {
      auto loc = op.getLoc();

      Value input = op.getInput();
      Value weight = op.getWeight();
      Value biasValue = op.getBias();
      int64_t stride = op.getStride();
      int64_t padding_size = op.getPaddingSize();
      int64_t dilation = op.getDilation();
      int64_t groups = op.getGroups();

      auto inputType = mlir::cast<RankedTensorType>(input.getType());
      auto weightType = mlir::cast<RankedTensorType>(weight.getType());
      if (!inputType.hasStaticShape() || !weightType.hasStaticShape()) {
        return failure();
      }

      ArrayRef<int64_t> inputShape = inputType.getShape();
      ArrayRef<int64_t> weightShape = weightType.getShape();

      bool isBatched = inputShape.size() == 3;
      int64_t N;
      if (isBatched) N = inputShape[0];
      int64_t L_in = inputShape[isBatched ? 2 : 1];
      int64_t C_out = weightShape[0];
      int64_t kernel_size = weightShape[2];

      if (stride == 0) {
        return failure();
      }
      int64_t L_out = (L_in + 2 * padding_size - dilation * (kernel_size - 1) - 1) / stride + 1;

      auto resultType = mlir::cast<RankedTensorType>(op.getResult().getType());
      Type resultElementType = resultType.getElementType();

      constexpr int64_t dim2 = 2;
      constexpr int64_t dim3 = 3;
      Value initTensor;
      if (isBatched) {
        SmallVector<int64_t, dim3> outputShape{N, C_out, L_out};
        initTensor = rewriter.create<tensor::EmptyOp>(loc, outputShape, resultElementType);
      } else {
        SmallVector<int64_t, dim2> outputShape{C_out, L_out};
        initTensor = rewriter.create<tensor::EmptyOp>(loc, outputShape, resultElementType);
      }

      SmallVector<Value, dim3> ins;
      ins.push_back(input);
      ins.push_back(weight);
      if (biasValue) {
        ins.push_back(biasValue);
      }

      auto newOp = rewriter.create<hfusion::Conv1DOp>(
          loc,
          ins,
          initTensor,
          stride,
          padding_size,
          dilation,
          groups
      );

      rewriter.replaceOp(op, newOp.getResult());

      return success();
    }
  };
}

namespace {
    struct TritonToHFusionPass
        : public mlir::triton::impl::TritonToHFusionBase<
            TritonToHFusionPass> {
    void runOnOperation() override;
    };
} // namespace

void TritonToHFusionPass::runOnOperation() {
  auto module = getOperation();

  // Use greedy pattern rewriter for simpler pattern matching
  // Patterns decide themselves whether to convert (via returning success/failure)
  RewritePatternSet patterns(&getContext());
  patterns.add<TritonHistogramToHFusionConversion>(patterns.getContext());
  patterns.add<TritonFpToFpToHFusionConversion>(patterns.getContext());
  patterns.add<TritonModToHFusionConversion>(patterns.getContext());
 	patterns.add<TritonConv1dToHFusionConversion>(patterns.getContext());

  // Apply patterns with greedy rewriting
  // This allows patterns to return failure() without causing pass failure
  if (failed(applyPatternsGreedily(module, std::move(patterns)))) {
    signalPassFailure();
  }
}

std::unique_ptr<OperationPass<ModuleOp>> mlir::triton::createTritonToHFusionPass() {
  return std::make_unique<TritonToHFusionPass>();
}
