#include "PatternPopulation.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "triton/Conversion/TritonGPUToLLVM/FMADotUtility.h"
#include "triton/Conversion/TritonGPUToLLVM/PatternTritonGPUOpToLLVM.h"
#include "triton/Conversion/TritonGPUToLLVM/Utility.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"

using namespace mlir;
using namespace mlir::triton;

using ::mlir::triton::gpu::BlockedEncodingAttr;
using ::mlir::triton::gpu::DotOperandEncodingAttr;
using ::mlir::triton::gpu::FMAVectorMultiplier;

namespace {

constexpr StringLiteral kStrategyAttr = "ttg.gpnpu.dot_strategy";
constexpr StringLiteral kTF32InputsAttr = "ttg.gpnpu.tf32_inputs";

static bool isTF32Dot(DotOp dotOp) {
  Type aType = dotOp.getA().getType().getElementType();
  Type bType = dotOp.getB().getType().getElementType();
  return aType.isF32() && bType.isF32() &&
         dotOp.getInputPrecision() == InputPrecision::TF32;
}

static bool shouldTruncateF32Inputs(DotOp dotOp) {
  // Direct TTGIR-to-LLVM tests have no strategy marker and represent an
  // original F32 TF32 dot. In the full pipeline, the semantic marker records
  // that fact before later passes can promote BF16/FP16 operands to F32.
  if (dotOp->hasAttr(kTF32InputsAttr))
    return true;
  return !dotOp->hasAttr(kStrategyAttr) && isTF32Dot(dotOp);
}

// GPNPU FMA multiplier: handles mixed-precision dot (e.g., bf16 A/B × f32 acc)
// by extending operands to the accumulator type before FMA.
//
// Contract: this multiplier is only invoked after DotOpConversion has
// verified that all operand element types are floating-point; non-FP dots
// are rejected earlier via notifyMatchFailure.
class GPNPUFMAVectorMultiplier : public FMAVectorMultiplier {
  ConversionPatternRewriter &rewriter;
  Location loc;
  bool bTruncateF32Inputs;

  Value truncateF32ToTF32(Value value) {
    auto b = TritonLLVMOpBuilder(loc, rewriter);
    Type i32Ty = rewriter.getI32Type();
    Value bits = b.bitcast(value, i32Ty);
    // TF32 keeps the FP32 sign and exponent plus the top ten mantissa bits.
    // Masking the low 13 bits implements raw truncation without a rounding
    // bias, matching the selected GPNPU fallback semantics.
    Value truncatedBits = b.and_(bits, b.i32_val(-8192));
    return b.bitcast(truncatedBits, value.getType());
  }

public:
  GPNPUFMAVectorMultiplier(ConversionPatternRewriter &rewriter, Location loc,
                           bool bTruncateF32Inputs)
      : rewriter(rewriter), loc(loc), bTruncateF32Inputs(bTruncateF32Inputs) {}

  Value multiplyVectors(ArrayRef<Value> a, ArrayRef<Value> b,
                        Value c) override {
    // Internal invariant from parametricConvertFMADot: |a| == |b| == K.
    assert(a.size() == b.size() && "FMA K-dim operand size mismatch");
    auto K = a.size();
    Value accum = c;
    Type accTy = accum.getType();

    for (size_t k = 0; k < K; ++k) {
      Value aElem = a[k];
      Value bElem = b[k];

      if (bTruncateF32Inputs) {
        assert(aElem.getType().isF32() && bElem.getType().isF32() &&
               "TF32 fallback requires FP32 inputs");
        aElem = truncateF32ToTF32(aElem);
        bElem = truncateF32ToTF32(bElem);
      }

      // Promote operands to accumulator type if needed (e.g., bf16 → f32).
      if (aElem.getType() != accTy)
        aElem = LLVM::FPExtOp::create(rewriter, loc, accTy, aElem);
      if (bElem.getType() != accTy)
        bElem = LLVM::FPExtOp::create(rewriter, loc, accTy, bElem);

      accum = LLVM::FMAOp::create(rewriter, loc, aElem, bElem, accum);
    }
    return accum;
  }
};

struct DotOpConversion : public ConvertOpToLLVMPattern<triton::DotOp> {
  using ConvertOpToLLVMPattern<triton::DotOp>::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(triton::DotOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto aType = cast<RankedTensorType>(op.getA().getType());
    auto bType = cast<RankedTensorType>(op.getB().getType());
    auto resultType = cast<RankedTensorType>(op.getType());

    // Result must live on GPNPU's blocked layout (FMA path).
    if (!isa<BlockedEncodingAttr>(resultType.getEncoding()))
      return rewriter.notifyMatchFailure(
          op, "GPNPU FMA dot requires a blocked result encoding");

    // Operands must be DotOperandEncodingAttr (opIdx = 0 / 1) whose parent
    // is the result's blocked layout.
    auto aEnc = dyn_cast<DotOperandEncodingAttr>(aType.getEncoding());
    auto bEnc = dyn_cast<DotOperandEncodingAttr>(bType.getEncoding());
    if (!aEnc || !bEnc)
      return rewriter.notifyMatchFailure(
          op, "GPNPU FMA dot requires dot_op encodings on A/B");

    // GPNPU FMA path only supports floating-point dot. int8 / int32 dot
    // would need a separate non-FP multiplier (future work).
    if (!isa<FloatType>(aType.getElementType()) ||
        !isa<FloatType>(bType.getElementType()) ||
        !isa<FloatType>(resultType.getElementType()))
      return rewriter.notifyMatchFailure(
          op, "GPNPU FMA dot only supports floating-point operands");

    bool bTruncateF32Inputs = shouldTruncateF32Inputs(op);
    GPNPUFMAVectorMultiplier multiplier(rewriter, op.getLoc(),
                                        bTruncateF32Inputs);
    return parametricConvertFMADot(op, adaptor, getTypeConverter(), rewriter,
                                   multiplier);
  }
};
} // namespace

namespace mlir::triton::Intellifusion {

void populateGPNPUDotOpToLLVMPatterns(LLVMTypeConverter &typeConverter,
                                      RewritePatternSet &patterns,
                                      PatternBenefit benefit) {
  patterns.add<DotOpConversion>(typeConverter, benefit);
}

} // namespace mlir::triton::Intellifusion
