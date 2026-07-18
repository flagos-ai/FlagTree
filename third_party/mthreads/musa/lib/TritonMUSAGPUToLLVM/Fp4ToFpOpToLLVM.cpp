#include "PatternTritonGPUOpToLLVM.h"
#include "TritonMUSAGPUToLLVM/Utility.h"
#include "mlir/Conversion/LLVMCommon/Pattern.h"
#include "mlir/Transforms/DialectConversion.h"
#include "triton/Conversion/TritonGPUToLLVM/Utility.h"
#include "llvm/ADT/SmallVector.h"

using namespace mlir;
using namespace mlir::triton;
using namespace mlir::triton::gpu;

namespace {

Value decodeE2M1Nibble(RewriterBase &rewriter, Location loc, Value nibble,
                       bool toFp16) {
  auto b = TritonLLVMOpBuilder(loc, rewriter);

  int toMBits = toFp16 ? 10 : 7;
  int toBias = toFp16 ? 15 : 127;
  int toPoint5 = toFp16 ? 0x3800 : 16128;

  Value em = b.and_(nibble, b.i8_val(0x7));
  Value sign = b.and_(nibble, b.i8_val(0x8));
  Value em16 = b.zext(i16_ty, em);
  Value sign16 = b.zext(i16_ty, sign);

  // e2m1 uses S.EE.M. Build the target fp16/bf16 bit pattern directly:
  // normal non-zero values need the target exponent bias, +/-0.5 is the only
  // subnormal value, and zero keeps only the sign bit.
  Value bits =
      b.or_(b.shl(em16, b.i16_val(toMBits - 1)), b.shl(sign16, b.i16_val(12)));
  Value normal = b.icmp_ne(b.and_(em, b.i8_val(0x6)), b.i8_val(0));
  bits =
      b.select(normal, b.add(bits, b.i16_val((toBias - 1) << toMBits)), bits);

  Value signBit = b.and_(bits, b.i16_val(0x8000));
  Value point5 = b.or_(b.i16_val(toPoint5), signBit);
  bits = b.select(b.icmp_eq(em, b.i8_val(0x1)), point5, bits);

  return b.bitcast(bits, toFp16 ? f16_ty : bf16_ty);
}

struct Fp4ToFpOpPattern : public ConvertOpToLLVMPattern<Fp4ToFpOp> {
  Fp4ToFpOpPattern(LLVMTypeConverter &typeConverter, PatternBenefit benefit)
      : ConvertOpToLLVMPattern<Fp4ToFpOp>(typeConverter, benefit) {}

  LogicalResult
  matchAndRewrite(Fp4ToFpOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    auto elemType = op.getType().getElementType();
    assert((elemType == f16_ty || elemType == bf16_ty) &&
           "Fp4ToFpOp only supports fp16/bf16 results");
    bool toFp16 = elemType == f16_ty;

    auto xVals = unpackLLElements(loc, adaptor.getSrc(), rewriter);

    SmallVector<Value> results;
    results.reserve(xVals.size() * 2);
    auto b = TritonLLVMOpBuilder(loc, rewriter);
    for (Value packed : xVals) {
      Value low = b.and_(packed, b.i8_val(0x0f));
      Value high = b.lshr(packed, b.i8_val(4));
      results.push_back(decodeE2M1Nibble(rewriter, loc, low, toFp16));
      results.push_back(decodeE2M1Nibble(rewriter, loc, high, toFp16));
    }

    Value result = packLLElements(loc, getTypeConverter(), results, rewriter,
                                  op.getType());
    rewriter.replaceOp(op, result);
    return success();
  }
};

} // namespace

void mlir::triton::MUSA::populateFp4ToFpToLLVMPatterns(
    LLVMTypeConverter &typeConverter, RewritePatternSet &patterns,
    PatternBenefit benefit) {
  patterns.add<Fp4ToFpOpPattern>(typeConverter, benefit);
}
