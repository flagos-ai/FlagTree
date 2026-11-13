#include "PatternTritonGPUOpToLLVM.h"
#include "Utility.h"
#include "triton/Conversion/TritonGPUToLLVM/Utility.h"

using namespace mlir;
using namespace mlir::triton;

namespace {
class Fp4ToFpScaledOpPattern : public ConvertOpToLLVMPattern<Fp4ToFpScaledOp> {
public:
  using ConvertOpToLLVMPattern<Fp4ToFpScaledOp>::ConvertOpToLLVMPattern;

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
    SmallVector<Value> results;
    results.reserve(values.size() * 2);
    for (unsigned i = 0; i < values.size(); ++i) {
      Value scale = b.zext(i32_ty, scales[i]);
      // E8M0 code zero is 2^-127, not zero. Code 255 is NaN, not Inf.
      Value scaleBits = b.shl(scale, b.i32_val(23));
      scaleBits = b.select(b.icmp_eq(scale, b.i32_val(0)), b.i32_val(0x400000),
                           scaleBits);
      scaleBits = b.select(b.icmp_eq(scale, b.i32_val(255)),
                           b.i32_val(0x7fc00000), scaleBits);
      Value factor = b.bitcast(scaleBits, f32_ty);
      Value byte = b.zext(i16_ty, values[i]);
      for (unsigned shift : {0, 4}) {
        Value nibble = b.lshr(byte, b.i16_val(shift));
        Value decoded;
        if (op->hasAttr("maca.fp4_dot_fused")) {
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
          results.push_back(
              LLVM::FPTruncOp::create(rewriter, loc, f16_ty, product));
        } else {
          // Every finite E2M1 * E8M0 product is exact in BF16, including
          // 2^-128. FP32 overflow and quiet NaNs retain their upper bits.
          // RNE therefore needs no rounding increment or NaN repair here.
          results.push_back(LLVM::METAX::convertFp32ToBf16(
              loc, op.getOperation(), rewriter, product, RoundingMode::RTZ));
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
    PatternBenefit benefit) {
  patterns.add<Fp4ToFpScaledOpPattern>(typeConverter, benefit);
}
