#include "triton/Conversion/TritonXPUToLLVM/FlagCxOpToLLVM/DeviceInitOpToLLVM.h"

#include "triton/Conversion/TritonXPUToLLVM/FlagCxOpToLLVM/FlagcxUtils.h"
#include "mlir/Conversion/LLVMCommon/Pattern.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "triton/Dialect/TritonXPU/IR/Dialect.h"

using namespace mlir;

namespace {

// XSHMEMI_LOCAL_BUF_LEN / __simd__ from the vendor's xshmem_common.h.
// flagcxDevInit takes the scratch as an argument because XTDK's __local__ is
// only an alignment attribute, so it has to come from the kernel's own frame.
constexpr int kXshmemLocalBufBytes = 4096;
constexpr int kXshmemLocalBufAlign = 64;

struct DeviceInitOpConversion
    : public ConvertOpToLLVMPattern<triton::xpu::DeviceInitOp> {
  using ConvertOpToLLVMPattern::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(triton::xpu::DeviceInitOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    auto bufTy = LLVM::LLVMPointerType::get(rewriter.getContext());
    Value size = rewriter.create<LLVM::ConstantOp>(
        loc, rewriter.getI32Type(),
        rewriter.getI32IntegerAttr(kXshmemLocalBufBytes));
    Value buf = rewriter.create<LLVM::AllocaOp>(
        loc, bufTy, rewriter.getI8Type(), size, kXshmemLocalBufAlign);
    if (failed(triton::xpu::getDeviceInitFuncCall(op, rewriter,
                                                 adaptor.getComm(), buf)))
      return failure();
    rewriter.eraseOp(op);
    return success();
  }
};

} // namespace

void mlir::triton::xpu::populateDeviceInitOpToLLVMPatterns(
    LLVMTypeConverter &typeConverter, RewritePatternSet &patterns,
    PatternBenefit benefit) {
  patterns.add<DeviceInitOpConversion>(typeConverter, benefit);
}
