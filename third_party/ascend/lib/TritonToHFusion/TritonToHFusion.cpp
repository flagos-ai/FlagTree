/*
 * Copyright (c) Huawei Technologies Co.
 * Licensed under the MIT license.
 */

#include "TritonToHFusion/Passes.h"

#include "Dialect/TritonAscend/IR/TritonAscendDialect.h"
#include "bishengir/Dialect/HFusion/IR/HFusion.h"
#include "bishengir/Dialect/HFusion/IR/HFusionImpl.h"
#include "bishengir/Dialect/Tensor/IR/TensorImpl.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/Attributes.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/LogicalResult.h"
#include <optional>

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
    Value mask = op.getMask();
    auto resultType = op.getResult().getType();

    int64_t numBins = 256; // 256 is default fallback.
    if (auto rankedTy = dyn_cast<RankedTensorType>(resultType))
      if (rankedTy.hasStaticShape() && rankedTy.getNumElements() > 0)
        numBins = rankedTy.getNumElements();

    auto numBinsAttr = rewriter.getI64IntegerAttr(numBins);

    auto newOp = rewriter.create<hfusion::HistogramOp>(loc, resultType, input,
                                                       numBinsAttr, mask);

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

    // RTNE (default) rounding is handled by TritonToLinalg using
    // arith.truncf/extf. Check this before querying tensor-only properties:
    // scalar fp_to_fp operations are valid Triton IR too.
    auto roundingMode = op.getRounding();
    if (!roundingMode.has_value() ||
        roundingMode.value() == triton::RoundingMode::RTNE) {
      // RTNE or no rounding mode specified: let TritonToLinalg handle it
      return failure();
    }

    // HFusion owns only non-RTNE tensor conversions. Keep the tensor
    // precondition checked here; a scalar must not silently fall through.
    auto srcType = cast<TensorType>(input.getType());
    auto dstType = cast<TensorType>(resultType);
    if (!srcType.getElementType().isIntOrFloat() ||
        !dstType.getElementType().isIntOrFloat()) {
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
    auto roundModeAttr =
        hfusion::RoundModeAttr::get(rewriter.getContext(), hfusionRoundMode);
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
    if (isBatched)
      N = inputShape[0];
    int64_t L_in = inputShape[isBatched ? 2 : 1];
    int64_t C_out = weightShape[0];
    int64_t kernel_size = weightShape[2];

    if (stride == 0) {
      return failure();
    }
    int64_t L_out =
        (L_in + 2 * padding_size - dilation * (kernel_size - 1) - 1) / stride +
        1;

    auto resultType = mlir::cast<RankedTensorType>(op.getResult().getType());
    Type resultElementType = resultType.getElementType();

    constexpr int64_t dim2 = 2;
    constexpr int64_t dim3 = 3;
    Value initTensor;
    if (isBatched) {
      SmallVector<int64_t, dim3> outputShape{N, C_out, L_out};
      initTensor =
          rewriter.create<tensor::EmptyOp>(loc, outputShape, resultElementType);
    } else {
      SmallVector<int64_t, dim2> outputShape{C_out, L_out};
      initTensor =
          rewriter.create<tensor::EmptyOp>(loc, outputShape, resultElementType);
    }

    SmallVector<Value, dim3> ins;
    ins.push_back(input);
    ins.push_back(weight);
    if (biasValue) {
      ins.push_back(biasValue);
    }

    auto newOp = rewriter.create<hfusion::Conv1DOp>(
        loc, ins, initTensor, stride, padding_size, dilation, groups);

    rewriter.replaceOp(op, newOp.getResult());

    return success();
  }
};

struct TritonConv2dToHFusionConversion
    : OpRewritePattern<triton::ascend::Conv2dOp> {
  using OpRewritePattern<triton::ascend::Conv2dOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(triton::ascend::Conv2dOp op,
                                PatternRewriter &rewriter) const final {
    auto loc = op.getLoc();

    Value input = op.getInput();
    Value weight = op.getWeight();
    Value biasValue = op.getBias();
    int64_t groups = op.getGroups();

    // Parse an integer (uniform scalar) or integer array attribute.
    auto parseParam =
        [](Attribute attr) -> std::optional<SmallVector<int64_t>> {
      if (auto intAttr = dyn_cast<IntegerAttr>(attr))
        return SmallVector<int64_t>{intAttr.getInt()};
      if (auto arrayAttr = dyn_cast<DenseI32ArrayAttr>(attr)) {
        SmallVector<int64_t> values;
        for (int32_t v : arrayAttr.asArrayRef())
          values.push_back(static_cast<int64_t>(v));
        return values;
      }
      return std::nullopt;
    };
    auto stride = parseParam(op.getStride());
    auto padding = parseParam(op.getPadding());
    auto dilation = parseParam(op.getDilation());
    if (!stride || !padding || !dilation) {
      return failure();
    }

    auto inputType = mlir::cast<RankedTensorType>(input.getType());
    auto weightType = mlir::cast<RankedTensorType>(weight.getType());
    if (!inputType.hasStaticShape() || !weightType.hasStaticShape()) {
      return failure();
    }

    ArrayRef<int64_t> inputShape = inputType.getShape();
    ArrayRef<int64_t> weightShape = weightType.getShape();

    bool isBatched = inputShape.size() == 4;
    int64_t N;
    if (isBatched)
      N = inputShape[0];
    int64_t H_in = inputShape[isBatched ? 2 : 1];
    int64_t W_in = inputShape[isBatched ? 3 : 2];
    int64_t C_out = weightShape[0];
    int64_t kH = weightShape[2];
    int64_t kW = weightShape[3];

    auto computeOutDim = [](int64_t in_size, int64_t pad_before,
                            int64_t pad_after, int64_t dil, int64_t k,
                            int64_t str) -> int64_t {
      return (in_size + pad_before + pad_after - dil * (k - 1) - 1) / str + 1;
    };

    // Expand padding to [pad_top, pad_bottom, pad_left, pad_right].
    SmallVector<int64_t> pads;
    if (padding->size() == 1) {
      pads.assign(4, (*padding)[0]);
    } else if (padding->size() == 2) {
      pads = {(*padding)[0], (*padding)[0], (*padding)[1], (*padding)[1]};
    } else {
      pads.assign(padding->begin(), padding->end());
    }
    // Uniform scalar or pair: front/back covers both forms.
    int64_t strideH = stride->front();
    int64_t strideW = stride->back();
    int64_t dilH = dilation->front();
    int64_t dilW = dilation->back();

    // pads = [pad_top, pad_bottom, pad_left, pad_right]
    int64_t H_out = computeOutDim(H_in, pads[0], pads[1], dilH, kH, strideH);
    int64_t W_out = computeOutDim(W_in, pads[2], pads[3], dilW, kW, strideW);

    auto resultType = mlir::cast<RankedTensorType>(op.getResult().getType());
    Type resultElementType = resultType.getElementType();

    constexpr int64_t dim3 = 3;
    constexpr int64_t dim4 = 4;
    Value initTensor;
    if (isBatched) {
      SmallVector<int64_t, dim4> outputShape{N, C_out, H_out, W_out};
      initTensor =
          rewriter.create<tensor::EmptyOp>(loc, outputShape, resultElementType);
    } else {
      SmallVector<int64_t, dim3> outputShape{C_out, H_out, W_out};
      initTensor =
          rewriter.create<tensor::EmptyOp>(loc, outputShape, resultElementType);
    }

    SmallVector<Value, dim3> ins;
    ins.push_back(input);
    ins.push_back(weight);
    if (biasValue) {
      ins.push_back(biasValue);
    }

    // Preserve the user's form: an integer stays a scalar IntegerAttr, a
    // 2-element sequence stays [pad_h, pad_w], a 4-element sequence stays
    // [pad_top, pad_bottom, pad_left, pad_right]. `pads` (the 4-element
    // expansion) is only used for the output-dim computation above.
    auto buildParamAttr = [&](const SmallVector<int64_t> &values) -> Attribute {
      if (values.size() == 1)
        return rewriter.getI64IntegerAttr(values[0]);
      return rewriter.getDenseI64ArrayAttr(values);
    };
    Attribute strideAttr = buildParamAttr(*stride);
    Attribute dilationAttr = buildParamAttr(*dilation);
    Attribute paddingAttr = buildParamAttr(*padding);

    auto newOp = rewriter.create<hfusion::Conv2DOp>(
        loc, ins, initTensor, strideAttr, paddingAttr, dilationAttr,
        static_cast<int32_t>(groups));

    rewriter.replaceOp(op, newOp.getResult());

    return success();
  }
};
} // namespace

namespace {
struct TritonToHFusionPass
    : public mlir::triton::impl::TritonToHFusionBase<TritonToHFusionPass> {
  bool compileOn91095 = false;
  void runOnOperation() override;
};
} // namespace

void TritonToHFusionPass::runOnOperation() {
  auto module = getOperation();

  // Use greedy pattern rewriter for simpler pattern matching
  // Patterns decide themselves whether to convert (via returning
  // success/failure)
  RewritePatternSet patterns(&getContext());
  // On 950 (910B4/91095), histogram is handled by
  // TTOpConverters::HistogramConverter in TritonToLinalg pass. On other
  // targets, use the HFusion lowering path.
  if (!compileOn91095) {
    patterns.add<TritonHistogramToHFusionConversion>(patterns.getContext());
  }
  patterns.add<TritonFpToFpToHFusionConversion>(patterns.getContext());
  patterns.add<TritonModToHFusionConversion>(patterns.getContext());
  patterns.add<TritonConv1dToHFusionConversion>(patterns.getContext());
  patterns.add<TritonConv2dToHFusionConversion>(patterns.getContext());

  // Apply patterns with greedy rewriting
  // This allows patterns to return failure() without causing pass failure
  if (failed(applyPatternsGreedily(module, std::move(patterns)))) {
    signalPassFailure();
  }
}

std::unique_ptr<OperationPass<ModuleOp>>
mlir::triton::createTritonToHFusionPass(bool compileOn91095) {
  auto pass = std::make_unique<TritonToHFusionPass>();
  pass->compileOn91095 = compileOn91095;
  return pass;
}
