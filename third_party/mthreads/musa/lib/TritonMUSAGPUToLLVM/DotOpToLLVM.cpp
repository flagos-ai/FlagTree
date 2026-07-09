#include "DotOpToLLVM/DotOpToLLVM.h"
#include "PatternTritonGPUOpToLLVM.h"
#include "triton/Conversion/TritonGPUToLLVM/PatternTritonGPUOpToLLVM.h"
#include "triton/Conversion/TritonGPUToLLVM/Utility.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "llvm/Support/ErrorHandling.h"

using namespace mlir;
using namespace mlir::triton::gpu;

namespace {

struct DotOpConversion : public ConvertOpToLLVMPattern<triton::DotOp> {
  using ConvertOpToLLVMPattern<triton::DotOp>::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(triton::DotOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto resultTy = cast<RankedTensorType>(op.getType());
    auto lhsTy = cast<RankedTensorType>(op.getA().getType());
    auto rhsTy = cast<RankedTensorType>(op.getB().getType());
    if ((op.getInputPrecision() == triton::InputPrecision::BF16x3 ||
         op.getInputPrecision() == triton::InputPrecision::BF16x6) &&
        lhsTy.getElementType().isF32() && rhsTy.getElementType().isF32()) {
      return op.emitError(
          "bf16x3/bf16x6 tt.dot must be rewritten by TritonGPUF32DotTC "
          "before MUSA LLVM lowering");
    }
    if (isa<MUSAWmmaEncodingAttr, MUSASqmmaEncodingAttr>(
            resultTy.getEncoding()))
      return op.emitError("MUSA matmul with mma encoding must be rewritten to "
                          "ttmg.wmma_dot/ttmg.squad_dot before LLVM lowering");
    if (isa<BlockedEncodingAttr>(resultTy.getEncoding()))
      return convertFMADot(op, adaptor, getTypeConverter(), rewriter);

    llvm::report_fatal_error(
        "Unsupported MUSA DotOp encoding in DotOp lowering.");
  }
};

} // namespace

void mlir::triton::MUSA::populateDotOpToLLVMPatterns(
    LLVMTypeConverter &typeConverter, RewritePatternSet &patterns,
    PatternBenefit benefit) {
  patterns.add<DotOpConversion>(typeConverter, benefit);
}
