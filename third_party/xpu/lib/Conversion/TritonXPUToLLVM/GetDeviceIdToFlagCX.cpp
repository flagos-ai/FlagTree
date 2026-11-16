#include "triton/Conversion/TritonXPUToLLVM/GetDeviceIdToFlagCX.h"

#include "mlir/Conversion/LLVMCommon/Pattern.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/IR/BuiltinOps.h"
#include "triton/Dialect/TritonXPU/IR/Dialect.h"

namespace {

using namespace mlir;

struct GetDeviceIdOpConversion
    : public ConvertOpToLLVMPattern<triton::xpu::GetDeviceIdOp> {
  using ConvertOpToLLVMPattern::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(triton::xpu::GetDeviceIdOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Value comm = adaptor.getComm();
    if (!comm) {
      auto module = op->getParentOfType<ModuleOp>();
      auto func = op->getParentOfType<LLVM::LLVMFuncOp>();
      if (!module || !module->hasAttr("xpu.implicit_distributed_params") ||
          !func || func.getNumArguments() < 2)
        return op.emitOpError(
            "implicit device rank requires FLAGTREE_LITE_DIST and a registered "
            "FlagCX context");
      comm = func.getArgument(0);
    }
    auto rank = rewriter.create<triton::xpu::GetLocalRankOp>(
        op.getLoc(), rewriter.getI32Type(), comm);
    rewriter.replaceOp(op, rank.getResult());
    return success();
  }
};

} // namespace

void mlir::triton::xpu::populateGetDeviceIdOpToFlagCXPatterns(
    LLVMTypeConverter &typeConverter, RewritePatternSet &patterns,
    PatternBenefit benefit) {
  patterns.add<GetDeviceIdOpConversion>(typeConverter, benefit);
}
