#include "triton/Conversion/TritonXPUToLLVM/FlagCxOpToLLVM/DeviceIntraBarrierOpToLLVM.h"

#include "triton/Conversion/TritonXPUToLLVM/FlagCxOpToLLVM/FlagcxUtils.h"
#include "mlir/Conversion/LLVMCommon/Pattern.h"
#include "triton/Dialect/TritonXPU/IR/Dialect.h"

using namespace mlir;

namespace {

struct DistributedBarrierOpConversion
    : public ConvertOpToLLVMPattern<triton::xpu::DistributedBarrierOp> {
  using ConvertOpToLLVMPattern::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(triton::xpu::DistributedBarrierOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto call = triton::xpu::getIntraBarrierFuncCall(
        op, rewriter, adaptor.getComm(), op.getCoopKindAttr().getInt(),
        op.getBarrierIndexAttr().getInt(), op.getOrderAttr().getInt(),
        op.getKindAttr().getValue());
    if (failed(call))
      return failure();
    rewriter.eraseOp(op);
    return success();
  }
};

} // namespace

void mlir::triton::xpu::populateDeviceIntraBarrierOpToLLVMPatterns(
    LLVMTypeConverter &typeConverter, RewritePatternSet &patterns,
    PatternBenefit benefit) {
  patterns.add<DistributedBarrierOpConversion>(typeConverter, benefit);
}
