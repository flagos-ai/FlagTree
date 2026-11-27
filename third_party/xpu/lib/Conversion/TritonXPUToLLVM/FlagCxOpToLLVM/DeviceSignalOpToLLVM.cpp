#include "triton/Conversion/TritonXPUToLLVM/FlagCxOpToLLVM/DeviceSignalOpToLLVM.h"

#include "triton/Conversion/TritonXPUToLLVM/FlagCxOpToLLVM/FlagcxUtils.h"
#include "mlir/Conversion/LLVMCommon/Pattern.h"
#include "triton/Dialect/TritonXPU/IR/Dialect.h"

using namespace mlir;

namespace {

struct SignalOpConversion
    : public ConvertOpToLLVMPattern<triton::xpu::SignalOp> {
  using ConvertOpToLLVMPattern::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(triton::xpu::SignalOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto call = triton::xpu::getSignalFuncCall(
        op, rewriter, adaptor.getComm(), op.getTeamKindAttr().getInt(),
        adaptor.getPeer(), adaptor.getSlot(), adaptor.getAddend(),
        op.getContextIdAttr().getInt(), op.getCoopKindAttr().getInt(),
        op.getScopeAttr().getInt());
    if (failed(call))
      return failure();
    rewriter.eraseOp(op);
    return success();
  }
};

struct SignalWaitOpConversion
    : public ConvertOpToLLVMPattern<triton::xpu::SignalWaitOp> {
  using ConvertOpToLLVMPattern::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(triton::xpu::SignalWaitOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto call = triton::xpu::getDevNetWaitFuncCallByKind(
        op, rewriter, adaptor.getComm(), adaptor.getSlot(), adaptor.getLeast(),
        op.getContextIdAttr().getInt(), op.getCoopKindAttr().getInt(),
        op.getOrderAttr().getInt(), op.getWaitKindAttr().getValue());
    if (failed(call))
      return failure();
    rewriter.eraseOp(op);
    return success();
  }
};

} // namespace

void mlir::triton::xpu::populateDeviceSignalOpToLLVMPatterns(
    LLVMTypeConverter &typeConverter, RewritePatternSet &patterns,
    PatternBenefit benefit) {
  patterns.add<SignalOpConversion, SignalWaitOpConversion>(typeConverter,
                                                           benefit);
}
