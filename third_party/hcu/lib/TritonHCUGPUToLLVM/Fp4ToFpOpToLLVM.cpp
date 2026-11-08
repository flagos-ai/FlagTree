#include "PatternTritonGPUOpToLLVM.h"

#include "Utility.h"
#include "mlir/Conversion/LLVMCommon/Pattern.h"
#include "mlir/IR/TypeUtilities.h"
#include "mlir/IR/ValueRange.h"
#include "mlir/Transforms/DialectConversion.h"
#include "triton/Conversion/TritonGPUToLLVM/Utility.h"
#include "llvm/ADT/SmallVector.h"

using namespace mlir;
using namespace mlir::triton;
using namespace mlir::triton::gpu;
using ::mlir::LLVM::HCU::upcast8xMxfp4_SW;

namespace {

class Fp4ToFpOpPattern : public ConvertOpToLLVMPattern<Fp4ToFpOp> {
public:
  Fp4ToFpOpPattern(LLVMTypeConverter &typeConverter, PatternBenefit benefit)
      : ConvertOpToLLVMPattern<Fp4ToFpOp>(typeConverter, benefit) {}

  LogicalResult
  matchAndRewrite(Fp4ToFpOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {

    auto loc = op.getLoc();
    auto elemType = op.getType().getElementType();
    assert(elemType == f16_ty || elemType == bf16_ty);
    bool toFp16 = elemType == f16_ty;

    auto xVals = unpackLLElements(loc, adaptor.getSrc(), rewriter);

    SmallVector<Value> results;
    results.reserve(xVals.size() * 2);
    auto b = TritonLLVMOpBuilder(loc, rewriter);
    for (int i = 0; i < xVals.size(); i += 4) {
      unsigned count = std::min<size_t>(4, xVals.size() - i);
      Value packedVec = b.undef(vec_ty(i8_ty, 4));
      for (int j : llvm::seq(4)) {
        Value v = j < count ? xVals[i + j] : b.i8_val(0);
        packedVec = b.insert_element(packedVec, v, b.i32_val(j));
      }
      SmallVector<Value> upcast =
          upcast8xMxfp4_SW(rewriter, op, toFp16, packedVec);
      results.append(upcast.begin(), upcast.begin() + 2 * count);
    }

    Value result = packLLElements(loc, getTypeConverter(), results, rewriter,
                                  op.getType());
    rewriter.replaceOp(op, result);
    return success();
  }
};
} // anonymous namespace

void mlir::triton::HCU::populateFp4ToFpToLLVMPatterns(
    LLVMTypeConverter &typeConverter, RewritePatternSet &patterns,
    const TargetInfo &, PatternBenefit benefit) {
  patterns.add<Fp4ToFpOpPattern>(typeConverter, benefit);
}
