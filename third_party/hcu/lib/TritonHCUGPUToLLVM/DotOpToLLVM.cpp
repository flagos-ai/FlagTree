#include "Utility.h"
#include "triton/Conversion/TritonGPUToLLVM/PatternTritonGPUOpToLLVM.h"
#include "triton/Dialect/TritonGPU/IR/Attributes.h"

using namespace mlir;

using ::mlir::triton::gpu::AMDWmmaEncodingAttr;
using ::mlir::triton::gpu::getShapePerCTA;

namespace mlir::triton::HCU {
LogicalResult convertHCUFMADot(triton::DotOp op, triton::DotOp::Adaptor adaptor,
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
} // namespace mlir::triton::HCU

namespace {
struct DotOpConversion : public ConvertOpToLLVMPattern<triton::DotOp> {
  using ConvertOpToLLVMPattern::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(triton::DotOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op->getLoc();
    // D = A * B + C
    Value D = op.getResult();

    auto dEncoding = cast<RankedTensorType>(D.getType()).getEncoding();
    if (isa<AMDMfmaEncodingAttr>(dEncoding)) {
      return HCU::convertMFMA(op, adaptor, getTypeConverter(), rewriter);
    }
    if (isa<AMDWmmaEncodingAttr>(dEncoding)) {
      return HCU::convertWMMA(op, adaptor, getTypeConverter(), rewriter);
    }

    if (isa<BlockedEncodingAttr>(
            cast<RankedTensorType>(D.getType()).getEncoding()))
      return HCU::convertHCUFMADot(op, adaptor, getTypeConverter(), rewriter);

    llvm::report_fatal_error(
        "Unsupported DotOp found when converting TritonGPU to LLVM.");
  }
};

struct ScaledDotOpConversion
    : public ConvertOpToLLVMPattern<triton::DotScaledOp> {
  using ConvertOpToLLVMPattern::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(triton::DotScaledOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Value D = op.getResult();

    auto dEncoding = cast<RankedTensorType>(D.getType()).getEncoding();

    if (isa<AMDMfmaEncodingAttr>(dEncoding)) {
      return HCU::convertScaledMFMA(op, adaptor, getTypeConverter(), rewriter);
    }
    if (isa<AMDWmmaEncodingAttr>(dEncoding)) {
      return HCU::convertScaledWMMA(op, adaptor, getTypeConverter(), rewriter);
    }

    llvm::report_fatal_error(
        "Unsupported DotScaleOp found when converting TritonGPU to LLVM.");
  }
};
} // namespace

namespace mlir::triton::HCU {
void populateDotOpToLLVMPatterns(LLVMTypeConverter &typeConverter,
                                 RewritePatternSet &patterns,
                                 ModuleAxisInfoAnalysis &axisInfoAnalysis,
                                 PatternBenefit benefit) {
  patterns.add<DotOpConversion>(typeConverter, benefit);
  patterns.add<ScaledDotOpConversion>(typeConverter, benefit);
}
} // namespace mlir::triton::HCU
