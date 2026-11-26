#include "triton/Conversion/TritonXPUToLLVM/FlagCxOpToLLVM/GetLocalRankOpToLLVM.h"

#include "FlagcxUtils.h"
#include "mlir/Conversion/LLVMCommon/Pattern.h"
#include "triton/Dialect/TritonXPU/IR/Dialect.h"

using namespace mlir;

namespace {

using triton::xpu::createFlagcxDeviceCall;
using triton::xpu::FlagcxDeviceFunction;

template <typename Op>
struct FlagCXCommQueryConversion : public ConvertOpToLLVMPattern<Op> {
  FlagCXCommQueryConversion(const LLVMTypeConverter &converter,
                            FlagcxDeviceFunction function,
                            PatternBenefit benefit)
      : ConvertOpToLLVMPattern<Op>(converter, benefit), function(function) {}

  LogicalResult
  matchAndRewrite(Op op, typename Op::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto call =
        createFlagcxDeviceCall(op, rewriter, function, adaptor.getComm());
    if (failed(call))
      return failure();
    rewriter.replaceOp(op, call->getResult());
    return success();
  }

private:
  FlagcxDeviceFunction function;
};

} // namespace

void mlir::triton::xpu::populateGetLocalRankOpToLLVMPatterns(
    LLVMTypeConverter &typeConverter, RewritePatternSet &patterns,
    PatternBenefit benefit) {
  patterns.add<FlagCXCommQueryConversion<GetLocalRankOp>>(
      typeConverter, FlagcxDeviceFunction::GetIntraRank, benefit);
}

void mlir::triton::xpu::populateGetNumPesOpToLLVMPatterns(
    LLVMTypeConverter &typeConverter, RewritePatternSet &patterns,
    PatternBenefit benefit) {
  patterns.add<FlagCXCommQueryConversion<GetNumPesOp>>(
      typeConverter, FlagcxDeviceFunction::GetIntraSize, benefit);
}
