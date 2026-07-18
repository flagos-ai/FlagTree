#include "PatternTritonGPUOpToLLVM.h"
#include "mlir/Conversion/LLVMCommon/Pattern.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "triton/Conversion/TritonGPUToLLVM/Utility.h"

using namespace mlir;

namespace {

struct TTGBarrierOpConversion
    : public ConvertOpToLLVMPattern<triton::gpu::BarrierOp> {
  TTGBarrierOpConversion(LLVMTypeConverter &typeConverter,
                         const mlir::triton::MUSA::TargetInfo &targetInfo,
                         PatternBenefit benefit)
      : ConvertOpToLLVMPattern<triton::gpu::BarrierOp>(typeConverter, benefit),
        targetInfo(targetInfo) {}

  LogicalResult
  matchAndRewrite(triton::gpu::BarrierOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    targetInfo.barrier(op.getLoc(), rewriter, op.getAddrSpace());
    rewriter.eraseOp(op);
    return success();
  }

private:
  const mlir::triton::MUSA::TargetInfo &targetInfo;
};

struct GPUBarrierOpConversion
    : public ConvertOpToLLVMPattern<mlir::gpu::BarrierOp> {
  GPUBarrierOpConversion(LLVMTypeConverter &typeConverter,
                         const mlir::triton::MUSA::TargetInfo &targetInfo,
                         PatternBenefit benefit)
      : ConvertOpToLLVMPattern<mlir::gpu::BarrierOp>(typeConverter, benefit),
        targetInfo(targetInfo) {}

  LogicalResult
  matchAndRewrite(mlir::gpu::BarrierOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    targetInfo.barrier(op.getLoc(), rewriter, triton::gpu::AddrSpace::Local);
    rewriter.eraseOp(op);
    return success();
  }

private:
  const mlir::triton::MUSA::TargetInfo &targetInfo;
};

} // namespace

void mlir::triton::MUSA::populateBarrierOpToLLVMPatterns(
    LLVMTypeConverter &typeConverter, RewritePatternSet &patterns,
    PatternBenefit benefit, const TargetInfo &targetInfo) {
  patterns.add<TTGBarrierOpConversion>(typeConverter, targetInfo, benefit);
  patterns.add<GPUBarrierOpConversion>(typeConverter, targetInfo, benefit);
}
