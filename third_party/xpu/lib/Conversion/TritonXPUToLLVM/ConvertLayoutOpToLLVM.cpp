#include "PatternTritonXPUOpToLLVM.h"
#include "triton/Conversion/TritonXPUToLLVM/LegacyLLVMHelpers.h" // LLVM22 dragon-style macros for XPU only

namespace {

using namespace mlir;
using namespace mlir::triton;

struct XPUConvertLayoutOpConversion
    : public ConvertOpToLLVMPattern<triton::xpu::ConvertLayoutOp> {
  XPUConvertLayoutOpConversion(LLVMTypeConverter &converter,
                               const xpu::TargetInfo &targetInfo,
                               PatternBenefit benefit)
      : ConvertOpToLLVMPattern<triton::xpu::ConvertLayoutOp>(converter,
                                                             benefit) {}

  bool isaXPUValidLayout(const Attribute &layout) const {
    if (!layout)
      return false;
    return mlir::isa<triton::xpu::ClusterLayoutAttr>(layout) ||
           (mlir::isa<triton::gpu::SliceEncodingAttr>(layout) &&
            mlir::isa<triton::xpu::ClusterLayoutAttr>(
                mlir::cast<triton::gpu::SliceEncodingAttr>(layout)
                    .getParent()));
  }

  LogicalResult
  matchAndRewrite(triton::xpu::ConvertLayoutOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Value src = op.getSrc();
    Value dst = op.getResult();
    auto srcTy = cast<RankedTensorType>(src.getType());
    auto dstTy = cast<RankedTensorType>(dst.getType());
    Attribute srcLayout = srcTy.getEncoding();
    Attribute dstLayout = dstTy.getEncoding();

    // Case 1: Both sides have valid XPU layouts — proper layout conversion.
    if (isaXPUValidLayout(srcLayout) && isaXPUValidLayout(dstLayout)) {
      return lowerOperand(op, adaptor, rewriter);
    }
    // Case 2: One side is unencoded (TLE path) — passthrough conversion.
    if (isaXPUValidLayout(srcLayout) || isaXPUValidLayout(dstLayout)) {
      return lowerOperand(op, adaptor, rewriter);
    }
    return failure();
  };

  LogicalResult lowerOperand(triton::xpu::ConvertLayoutOp op, OpAdaptor adaptor,
                             ConversionPatternRewriter &rewriter) const {
    auto loc = op.getLoc();
    auto typeConverter = getTypeConverter();
    Value src = op.getSrc();
    Value dst = op.getResult();
    auto srcTy = cast<RankedTensorType>(src.getType());
    auto dstTy = cast<RankedTensorType>(dst.getType());

    auto vals = unpackLLElements(loc, adaptor.getSrc(), rewriter);
    auto dstStructTy =
        cast<LLVM::LLVMStructType>(typeConverter->convertType(dstTy));
    unsigned dstElems = dstStructTy.getBody().size();
    if (vals.size() > dstElems) {
      // Truncate: UnrollControl may divide dst but not loop-invariant src.
      vals = SmallVector<Value>(vals.begin(), vals.begin() + dstElems);
    } else if (vals.size() < dstElems && vals.size() > 0) {
      // Expand: broadcast-like convert where src has fewer elems per thread.
      // Replicate the source values cyclically to fill the destination.
      SmallVector<Value> expanded;
      expanded.reserve(dstElems);
      for (unsigned i = 0; i < dstElems; ++i) {
        expanded.push_back(vals[i % vals.size()]);
      }
      vals = std::move(expanded);
    }
    Value ret = packLLElements(loc, typeConverter, vals, rewriter, dstTy);

    rewriter.replaceOp(op, ret);
    return success();
  }
};

} // namespace

void mlir::triton::xpu::populateConvertLayoutOpToLLVMPatterns(
    LLVMTypeConverter &typeConverter, const TargetInfo &targetInfo,
    RewritePatternSet &patterns, PatternBenefit benefit) {
  patterns.add<XPUConvertLayoutOpConversion>(typeConverter, targetInfo,
                                             benefit);
}
