#ifdef __ILUVATAR_TLE__

#include "Conversion/TleToLLVM/ViewOpToLLVM.h"

#include "IR/Dialect.h"
#include "mlir/Conversion/LLVMCommon/Pattern.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/LLVMIR/LLVMTypes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "triton/Conversion/TritonGPUToLLVM/Utility.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"

namespace {

using namespace mlir;
using namespace mlir::triton;
namespace ttg = mlir::triton::gpu;
namespace iluvatar_tle = mlir::triton::iluvatar_tle;
using ::mlir::LLVM::getSharedMemoryObjectFromStruct;

struct MemDescAliasOpConversion
    : public ConvertOpToLLVMPattern<iluvatar_tle::MemDescAliasOp> {
  using ConvertOpToLLVMPattern::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(iluvatar_tle::MemDescAliasOp op, OpAdaptor adaptor,
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
    auto retVal =
        LLVM::getStructFromSharedMemoryObject(loc, dstSmemObj, rewriter);
    rewriter.replaceOp(op, retVal);
    return success();
  }
};

} // namespace

void iluvatar_tle::populateMemDescAliasOpToLLVMPatterns(
    LLVMTypeConverter &typeConverter, RewritePatternSet &patterns,
    PatternBenefit benefit) {
  patterns.add<MemDescAliasOpConversion>(typeConverter, benefit);
}

#endif
