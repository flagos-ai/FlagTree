// Copyright 2026 FlagOS Contributors
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.

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
    return mlir::isa<triton::xpu::ClusterLayoutAttr>(layout) ||
           mlir::isa<triton::xpu::ClusterLayoutAttr>(
               mlir::cast<triton::gpu::SliceEncodingAttr>(layout).getParent());
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

    if (isaXPUValidLayout(srcLayout) && isaXPUValidLayout(dstLayout)) {
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
