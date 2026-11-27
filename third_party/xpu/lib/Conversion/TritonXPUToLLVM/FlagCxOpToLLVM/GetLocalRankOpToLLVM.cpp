#include "triton/Conversion/TritonXPUToLLVM/FlagCxOpToLLVM/GetLocalRankOpToLLVM.h"

#include "triton/Conversion/TritonXPUToLLVM/FlagCxOpToLLVM/FlagcxUtils.h"
#include "mlir/Conversion/LLVMCommon/Pattern.h"
#include "triton/Dialect/TritonXPU/IR/Dialect.h"

using namespace mlir;

namespace {

template <typename Op, auto emitCall>
struct FlagCXCommQueryConversion : public ConvertOpToLLVMPattern<Op> {
  using ConvertOpToLLVMPattern<Op>::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(Op op, typename Op::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto call = emitCall(op, rewriter, adaptor.getComm());
    if (failed(call))
      return failure();
    rewriter.replaceOp(op, call->getResult());
    return success();
  }
};

} // namespace

void mlir::triton::xpu::populateGetLocalRankOpToLLVMPatterns(
    LLVMTypeConverter &typeConverter, RewritePatternSet &patterns,
    PatternBenefit benefit) {
  patterns.add<FlagCXCommQueryConversion<GetLocalRankOp, getLocalPeFuncCall>>(
      typeConverter, benefit);
}

void mlir::triton::xpu::populateGetNumPesOpToLLVMPatterns(
    LLVMTypeConverter &typeConverter, RewritePatternSet &patterns,
    PatternBenefit benefit) {
  patterns.add<FlagCXCommQueryConversion<GetNumPesOp, getNumPesFuncCall>>(
      typeConverter, benefit);
}
