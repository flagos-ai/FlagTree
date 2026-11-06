#include "PatternTritonGPUOpToLLVM.h"
#include "Utility.h"
#include "triton/Conversion/TritonGPUToLLVM/Utility.h"

using namespace mlir;
using namespace mlir::triton;
using namespace mlir::triton::gpu;

namespace {
class Fp4ToFpOpPattern : public ConvertOpToLLVMPattern<Fp4ToFpOp> {
public:
  using ConvertOpToLLVMPattern<Fp4ToFpOp>::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(Fp4ToFpOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    auto elemType = op.getType().getElementType();
    bool toFp16 = elemType.isF16();
    op.emitWarning(
        "using non-native FP4 conversion on MetaX: decoding e2m1 to ")
        << elemType << " with integer operations";

    auto b = TritonLLVMOpBuilder(loc, rewriter);
    auto packed = unpackLLElements(loc, adaptor.getSrc(), rewriter);
    SmallVector<Value> results;
    results.reserve(packed.size() * 2);
    for (Value byte : packed) {
      Value bits = b.zext(i16_ty, byte);
      // Fp4ToFpOp's layout places each byte's low nibble before its high
      // nibble.
      for (unsigned shift : {0, 4}) {
        Value nibble = b.lshr(bits, b.i16_val(shift));
        results.push_back(
            LLVM::METAX::decodeFp4E2M1(loc, rewriter, nibble, toFp16));
      }
    }
    rewriter.replaceOp(op, packLLElements(loc, getTypeConverter(), results,
                                          rewriter, op.getType()));
    return success();
  }
};
} // namespace

void mlir::triton::METAX::populateFp4ToFpToLLVMPatterns(
    LLVMTypeConverter &typeConverter, RewritePatternSet &patterns,
    PatternBenefit benefit) {
  patterns.add<Fp4ToFpOpPattern>(typeConverter, benefit);
}
