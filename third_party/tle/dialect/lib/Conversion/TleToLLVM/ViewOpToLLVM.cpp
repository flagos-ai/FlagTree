#include "mlir/Conversion/LLVMCommon/Pattern.h"
#include "tle/dialect/include/IR/Dialect.h"
#include "tle/dialect/include/Transforms/PatternTleToLLVM.h"
#include "triton/Conversion/TritonGPUToLLVM/PatternTritonGPUOpToLLVM.h"
#include "triton/Conversion/TritonGPUToLLVM/Utility.h"

using namespace mlir;
using namespace mlir::triton;
using ::mlir::LLVM::getSharedMemoryObjectFromStruct;

namespace {

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
    auto retVal =
        LLVM::getStructFromSharedMemoryObject(loc, dstSmemObj, rewriter);
    rewriter.replaceOp(op, retVal);
    return success();
  }
};

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

    auto dstSmemObj =
        SharedMemoryObject(base, resultElemTy, resultTy.getRank(), loc,
                           rewriter);
    auto retVal =
        LLVM::getStructFromSharedMemoryObject(loc, dstSmemObj, rewriter);
    rewriter.replaceOp(op, retVal);
    return success();
  }
};

} // namespace

void mlir::triton::tle::populateMemDescWGMMAViewOpToLLVMPatterns(
    LLVMTypeConverter &typeConverter, RewritePatternSet &patterns,
    unsigned benefit) {
  patterns.add<MemDescWGMMAViewOpConversion>(typeConverter, benefit);
}

void mlir::triton::tle::populateMemDescAliasOpToLLVMPatterns(
    LLVMTypeConverter &typeConverter, RewritePatternSet &patterns,
    unsigned benefit) {
  patterns.add<MemDescAliasOpConversion>(typeConverter, benefit);
}
