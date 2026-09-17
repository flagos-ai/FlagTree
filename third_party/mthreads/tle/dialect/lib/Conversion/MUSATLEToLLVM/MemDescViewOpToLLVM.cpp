#ifdef __TLE__

#include "mlir/Conversion/LLVMCommon/Pattern.h"
#include "tle/dialect/include/IR/Dialect.h"
#include "triton/Conversion/TritonGPUToLLVM/PatternTritonGPUOpToLLVM.h"
#include "triton/Conversion/TritonGPUToLLVM/Utility.h"

using namespace mlir;
using namespace mlir::triton;
using ::mlir::LLVM::getSharedMemoryObjectFromStruct;

namespace {

struct MemDescAliasOpConversion
    : public ConvertOpToLLVMPattern<triton::tle::MemDescAliasOp> {
  using ConvertOpToLLVMPattern::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(triton::tle::MemDescAliasOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op->getLoc();
    auto b = TritonLLVMOpBuilder(loc, rewriter);
    auto srcTy = op.getSrc().getType();
    auto resultTy = op.getType();
    auto srcElemTy = getTypeConverter()->convertType(srcTy.getElementType());
    auto resultElemTy =
        getTypeConverter()->convertType(resultTy.getElementType());
    auto srcSmemObj = getSharedMemoryObjectFromStruct(loc, adaptor.getSrc(),
                                                      srcElemTy, rewriter);
    Value base = srcSmemObj.getShmemAffineBase(loc, rewriter, srcTy);
    int64_t offsetBytes = op.getOffsetBytesAttr().getInt();
    if (offsetBytes != 0)
      base = b.gep(base.getType(), i8_ty, base, b.i32_val(offsetBytes));
    auto dstSmemObj = SharedMemoryObject(base, resultElemTy, resultTy.getRank(),
                                         loc, rewriter);
    rewriter.replaceOp(
        op, LLVM::getStructFromSharedMemoryObject(loc, dstSmemObj, rewriter));
    return success();
  }
};

struct MemDescWGMMAViewOpConversion
    : public ConvertOpToLLVMPattern<triton::tle::MemDescWGMMAViewOp> {
  using ConvertOpToLLVMPattern::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(triton::tle::MemDescWGMMAViewOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op->getLoc();
    auto resultTy = op.getType();
    auto llvmElemTy =
        getTypeConverter()->convertType(resultTy.getElementType());
    auto srcSmemObj = getSharedMemoryObjectFromStruct(loc, adaptor.getSrc(),
                                                      llvmElemTy, rewriter);
    auto dstSmemObj = SharedMemoryObject(
        srcSmemObj.getBase(), srcSmemObj.getBaseElemType(),
        /*offsets=*/applyPermutation(srcSmemObj.getOffsets(), op.getOrder()));
    rewriter.replaceOp(
        op, LLVM::getStructFromSharedMemoryObject(loc, dstSmemObj, rewriter));
    return success();
  }
};

} // namespace

namespace mlir::triton::musa_tle {

void populateMUSATLEMemDescViewToLLVMPatterns(LLVMTypeConverter &typeConverter,
                                              RewritePatternSet &patterns,
                                              PatternBenefit benefit) {
  patterns.add<MemDescAliasOpConversion, MemDescWGMMAViewOpConversion>(
      typeConverter, benefit);
}

} // namespace mlir::triton::musa_tle

#endif // __TLE__
