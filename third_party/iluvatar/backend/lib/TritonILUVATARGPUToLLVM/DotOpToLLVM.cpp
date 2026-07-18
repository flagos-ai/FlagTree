#include "Utility.h"
#include "triton/Conversion/TritonGPUToLLVM/PatternTritonGPUOpToLLVM.h"
#include "triton/Dialect/TritonGPU/IR/Attributes.h"

using namespace mlir;

using ::mlir::triton::gpu::IluvatarMmaEncodingAttr;

namespace mlir::triton::ILUVATAR {
LogicalResult convertILUVATARFMADot(triton::DotOp op,
                                    triton::DotOp::Adaptor adaptor,
                                    const LLVMTypeConverter *typeConverter,
                                    ConversionPatternRewriter &rewriter);

LogicalResult convertTCU161616(triton::DotOp op, triton::DotOp::Adaptor adaptor,
                               const LLVMTypeConverter *typeConverter,
                               ConversionPatternRewriter &rewriter);

LogicalResult convertMFMA(triton::DotOp op, triton::DotOp::Adaptor adaptor,
                          const LLVMTypeConverter *typeConverter,
                          ConversionPatternRewriter &rewriter);

LogicalResult convertScaledMFMA(triton::DotScaledOp op,
                                triton::DotScaledOp::Adaptor adaptor,
                                const LLVMTypeConverter *typeConverter,
                                ConversionPatternRewriter &rewriter);

LogicalResult convertWMMA(triton::DotOp op, triton::DotOp::Adaptor adaptor,
                          const LLVMTypeConverter *typeConverter,
                          ConversionPatternRewriter &rewriter);

LogicalResult convertScaledWMMA(triton::DotScaledOp op,
                                triton::DotScaledOp::Adaptor adaptor,
                                const LLVMTypeConverter *typeConverter,
                                ConversionPatternRewriter &rewriter);
} // namespace mlir::triton::ILUVATAR

namespace {
struct DotOpConversion : public ConvertOpToLLVMPattern<triton::DotOp> {
  using ConvertOpToLLVMPattern::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(triton::DotOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    // D = A * B + C
    Value D = op.getResult();

    auto dEncoding = cast<RankedTensorType>(D.getType()).getEncoding();

    if (auto mmaLayout = dyn_cast<IluvatarMmaEncodingAttr>(dEncoding)) {
      if (mmaLayout.isVolta())
        return ILUVATAR::convertTCU161616(op, adaptor, getTypeConverter(),
                                          rewriter);
      llvm::report_fatal_error(
          "Unsupported Iluvatar MMA version found when converting DotOp.");
    }

    if (isa<BlockedEncodingAttr>(
            cast<RankedTensorType>(D.getType()).getEncoding()))
      return ILUVATAR::convertILUVATARFMADot(op, adaptor, getTypeConverter(),
                                             rewriter);

    llvm::report_fatal_error(
        "Unsupported DotOp found when converting TritonGPU to LLVM.");
  }
};

} // namespace

namespace mlir::triton::ILUVATAR {
void populateDotOpToLLVMPatterns(LLVMTypeConverter &typeConverter,
                                 RewritePatternSet &patterns,
                                 ModuleAxisInfoAnalysis &axisInfoAnalysis,
                                 PatternBenefit benefit) {
  patterns.add<DotOpConversion>(typeConverter, benefit);
  // patterns.add<ScaledDotOpConversion>(typeConverter, benefit);
}
} // namespace mlir::triton::ILUVATAR
