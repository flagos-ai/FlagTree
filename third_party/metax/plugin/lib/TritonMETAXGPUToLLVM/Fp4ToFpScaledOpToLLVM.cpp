#include "PatternTritonGPUOpToLLVM.h"
#include "Utility.h"
#include "triton/Conversion/TritonGPUToLLVM/Utility.h"

using namespace mlir;
using namespace mlir::triton;

namespace {
constexpr int kC550ComputeCapability = 80;

class Fp4ToFpScaledOpPattern : public ConvertOpToLLVMPattern<Fp4ToFpScaledOp> {
  int computeCapability;

public:
  Fp4ToFpScaledOpPattern(LLVMTypeConverter &typeConverter,
                         int computeCapability, PatternBenefit benefit)
      : ConvertOpToLLVMPattern<Fp4ToFpScaledOp>(typeConverter, benefit),
        computeCapability(computeCapability) {}

  LogicalResult
  matchAndRewrite(Fp4ToFpScaledOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    auto b = TritonLLVMOpBuilder(loc, rewriter);
    auto values = unpackLLElements(loc, adaptor.getInput(), rewriter);
    auto scales = unpackLLElements(loc, adaptor.getScale(), rewriter);
    if (values.size() != scales.size())
      return rewriter.notifyMatchFailure(
          op, "input and scale register layouts must match");
    op.emitWarning("using non-native scaled FP4 conversion on MetaX: GPU "
                   "software decoding");
    // The fused decode relies on preserving an FP32 subnormal operand.
    bool isC550 = computeCapability == kC550ComputeCapability;
    bool useFusedDecode = isC550 && op->hasAttr("maca.fp4_dot_fused");
    SmallVector<Value> results;
    results.reserve(values.size() * 2);
    for (unsigned i = 0; i < values.size(); ++i) {
      Value scale = b.zext(i32_ty, scales[i]);
      Value isScaleZero = b.icmp_eq(scale, b.i32_val(0));
      Value scaleBits = b.shl(scale, b.i32_val(23));
      // C550 preserves the subnormal FP32 factor for E8M0 code zero.
      if (isC550)
        scaleBits = b.select(isScaleZero, b.i32_val(0x400000), scaleBits);
      // Code 255 is NaN, not Inf.
      scaleBits = b.select(b.icmp_eq(scale, b.i32_val(255)),
                           b.i32_val(0x7fc00000), scaleBits);
      Value factor = b.bitcast(scaleBits, f32_ty);
      Value byte = b.zext(i16_ty, values[i]);
      for (unsigned shift : {0, 4}) {
        Value nibble = b.lshr(byte, b.i16_val(shift));
        Value decoded;
        if (useFusedDecode) {
          // E2M1 bits in the low FP32 exponent positions encode x*2^-126.
          // C550 FP32 multiply preserves the .5 code's subnormal input.
          Value code = b.zext(i32_ty, nibble);
          Value sign = b.shl(b.and_(code, b.i32_val(8)), b.i32_val(28));
          Value magnitude = b.shl(b.and_(code, b.i32_val(7)), b.i32_val(22));
          Value raw = b.bitcast(b.or_(sign, magnitude), f32_ty);
          decoded = b.fmul(raw, b.bitcast(b.i32_val(0x7e800000), f32_ty));
        } else {
          Value bf16 = LLVM::METAX::decodeFp4E2M1(loc, rewriter, nibble, false);
          decoded = LLVM::METAX::convertBf16ToFp32(loc, rewriter, bf16);
        }
        Value product = b.fmul(decoded, factor);
        if (op.getType().getElementType().isF16()) {
          // With E8M0 code zero, all E2M1 values round to signed FP16 zero.
          results.push_back(
              LLVM::FPTruncOp::create(rewriter, loc, f16_ty, product));
        } else {
          // Every finite E2M1 * E8M0 product is exact in BF16, including
          // 2^-128. FP32 overflow and quiet NaNs retain their upper bits.
          // RNE therefore needs no rounding increment or NaN repair here.
          Value result = LLVM::METAX::convertFp32ToBf16(
              loc, op.getOperation(), rewriter, product, RoundingMode::RTZ);
          if (!isC550) {
            // Encode E2M1 * 2^-127 directly in BF16 so a target that flushes
            // FP32 subnormals still preserves the product's exact bits.
            Value magnitude = b.and_(nibble, b.i16_val(7));
            Value subnormal = b.shl(magnitude, b.i16_val(5));
            Value normal = b.shl(b.sub(magnitude, b.i16_val(2)), b.i16_val(6));
            Value bits = b.select(b.icmp_ult(magnitude, b.i16_val(4)),
                                  subnormal, normal);
            Value sign = b.shl(b.and_(nibble, b.i16_val(8)), b.i16_val(12));
            result = b.select(isScaleZero, b.or_(bits, sign), result);
          }
          results.push_back(result);
        }
      }
    }
    rewriter.replaceOp(op, packLLElements(loc, getTypeConverter(), results,
                                          rewriter, op.getType()));
    return success();
  }
};
} // namespace

void mlir::triton::METAX::populateFp4ToFpScaledToLLVMPatterns(
    LLVMTypeConverter &typeConverter, RewritePatternSet &patterns,
    int computeCapability, PatternBenefit benefit) {
  patterns.add<Fp4ToFpScaledOpPattern>(typeConverter, computeCapability,
                                       benefit);
}
