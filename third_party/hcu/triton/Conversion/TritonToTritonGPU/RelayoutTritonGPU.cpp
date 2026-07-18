// Copyright 2026 FlagOS Contributors
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.

#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"
#include "triton/Conversion/TritonToTritonGPU/Passes.h"
#include "triton/Dialect/TritonGPU/Transforms/TritonGPUConversion.h"
#include "triton/Dialect/TritonNvidiaGPU/IR/Dialect.h"

namespace mlir::triton {
#define GEN_PASS_DEF_RELAYOUTTRITONGPU
#include "triton/Conversion/TritonToTritonGPU/Passes.h.inc"
} // namespace mlir::triton

namespace {

using namespace mlir;
using namespace triton;
using namespace triton::gpu;
namespace ttng = triton::nvidia_gpu;

// Given a tensor and its representation in tensor memory, determine its
// distributed layout.
RankedTensorType getTMEMTensorLayout(const TypeConverter *tc,
                                     RankedTensorType type, MemDescType memdesc,
                                     unsigned numWarps) {
  type = cast<RankedTensorType>(tc->convertType(type));
  auto ctaLayout = getCTALayout(type.getEncoding());
  auto encoding =
      ttng::getDefaultLayoutForTmemLdSt(memdesc, numWarps, ctaLayout);
  return type.cloneWithEncoding(encoding);
}

struct TMEMLoadOpPattern : public OpConversionPattern<ttng::TMEMLoadOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(ttng::TMEMLoadOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    RankedTensorType type = getTMEMTensorLayout(
        typeConverter, op.getType(), op.getSrc().getType(), lookupNumWarps(op));
    rewriter.modifyOpInPlace(op, [&] { op.getResult().setType(type); });
    Type resultType = getTypeConverter()->convertType(op.getType());
    rewriter.setInsertionPointAfter(op);
    auto cvt = ConvertLayoutOp::create(rewriter, op.getLoc(), resultType,
                                       op.getResult());
    rewriter.replaceAllUsesExcept(op.getResult(), cvt, cvt);
    return success();
  }
};

struct TMEMStoreOpPattern : public OpConversionPattern<ttng::TMEMStoreOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(ttng::TMEMStoreOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    RankedTensorType type =
        getTMEMTensorLayout(typeConverter, op.getSrc().getType(),
                            op.getDst().getType(), lookupNumWarps(op));
    Value src =
        ConvertLayoutOp::create(rewriter, op.getLoc(), type, adaptor.getSrc());
    rewriter.modifyOpInPlace(op, [&] { op.getSrcMutable().assign(src); });
    return success();
  }
};

struct TMEMAllocOpPattern : public OpConversionPattern<ttng::TMEMAllocOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(ttng::TMEMAllocOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    if (!op.getSrc())
      return success();
    RankedTensorType type = getTMEMTensorLayout(
        typeConverter, op.getSrc().getType(), op.getType(), lookupNumWarps(op));
    Value src =
        ConvertLayoutOp::create(rewriter, op.getLoc(), type, adaptor.getSrc());
    rewriter.modifyOpInPlace(op, [&] { op.getSrcMutable().assign(src); });
    return success();
  }
};

class RelayoutTritonGPU
    : public triton::impl::RelayoutTritonGPUBase<RelayoutTritonGPU> {
public:
  using RelayoutTritonGPUBase::RelayoutTritonGPUBase;

  void runOnOperation() override {
    MLIRContext *context = &getContext();
    ModuleOp mod = getOperation();

    int numWarps = lookupNumWarps(mod);
    int threadsPerWarp = TritonGPUDialect::getThreadsPerWarp(mod);
    int numCTAs = TritonGPUDialect::getNumCTAs(mod);

    // type converter
    TritonGPUTypeConverter typeConverter(context, numWarps, threadsPerWarp,
                                         numCTAs, /*enableSourceRemat=*/true);
    TritonGPUConversionTarget target(*context, typeConverter);
    target.addDynamicallyLegalDialect<ttng::TritonNvidiaGPUDialect>(
        [&](Operation *op) {
          return TritonGPUConversionTarget::isDynamicallyLegal(op,
                                                               typeConverter);
        });

    // rewrite patterns
    RewritePatternSet patterns(context);
    // add rules
    patterns.insert<
        // clang-format off
        GatherScatterOpPattern<ttng::AsyncTMAGatherOp>,
        GatherScatterOpPattern<ttng::AsyncTMAScatterOp>,
        TMEMLoadOpPattern,
        TMEMStoreOpPattern,
        TMEMAllocOpPattern
        // clang-format on
        >(typeConverter, context);

    if (failed(applyPartialConversion(mod, target, std::move(patterns))))
      return signalPassFailure();
  }
};

} // namespace
