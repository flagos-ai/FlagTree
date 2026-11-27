#include "triton/Conversion/TritonXPUToLLVM/FlagCxOpToLLVM/FlagcxUtils.h"
#include "PatternTritonXPUOpToLLVM.h"

#include "mlir/Conversion/LLVMCommon/Pattern.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "triton/Dialect/TritonXPU/IR/Dialect.h"

using namespace mlir;

namespace {

using triton::xpu::getFlagcxGlobalPtrType;

struct RemotePointerConversion
    : public ConvertOpToLLVMPattern<triton::xpu::RemotePointerOp> {
  using ConvertOpToLLVMPattern::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(triton::xpu::RemotePointerOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    if (getTypeConverter()->convertType(op.getType()) !=
        getFlagcxGlobalPtrType(rewriter.getContext()))
      return op.emitOpError(
          "remote pointer must use XPU global address space 1");
    Value elementBytes = rewriter.create<LLVM::ConstantOp>(
        loc, rewriter.getI64Type(),
        rewriter.getI64IntegerAttr(op.getElementBytes()));
    Value byteOffset =
        rewriter.create<LLVM::MulOp>(loc, adaptor.getOffset(), elementBytes);
    auto call = triton::xpu::getIntraPointerFuncCall(
        op, rewriter, adaptor.getMemory(), byteOffset, adaptor.getPeer());
    if (failed(call))
      return failure();
    rewriter.replaceOp(op, call->getResult());
    return success();
  }
};

} // namespace

void mlir::triton::xpu::populateRemotePointerOpToLLVMPatterns(
    LLVMTypeConverter &typeConverter, RewritePatternSet &patterns,
    PatternBenefit benefit) {
  patterns.add<RemotePointerConversion>(typeConverter, benefit);
}
