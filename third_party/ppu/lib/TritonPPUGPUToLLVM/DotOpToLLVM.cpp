/*
 * Copyright (c) 2026 T-Head Semiconductor Co., Ltd. All rights reserved.
 *
 * Permission is hereby granted, free of charge, to any person obtaining
 * a copy of this software and associated documentation files
 * (the "Software"), to deal in the Software without restriction,
 * including without limitation the rights to use, copy, modify, merge,
 * publish, distribute, sublicense, and/or sell copies of the Software,
 * and to permit persons to whom the Software is furnished to do so,
 * subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be
 * included in all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
 * EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
 * MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
 * IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY
 * CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,
 * TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE
 * SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
 */

#include "PatternTritonGPUOpToLLVM.h"
#include "Utility.h"

#include "triton/Conversion/TritonGPUToLLVM/PatternTritonGPUOpToLLVM.h"

using namespace mlir;
using namespace mlir::triton;

using ::mlir::triton::gpu::getShapePerCTA;
using ::mlir::triton::gpu::PPUMmaEncodingAttr;

LogicalResult convertPPUMmaV1(triton::DotOp op, triton::DotOp::Adaptor adaptor,
                              const LLVMTypeConverter *typeConverter,
                              ConversionPatternRewriter &rewriter);

LogicalResult convertPPUMmaV2(triton::DotOp op, triton::DotOp::Adaptor adaptor,
                              const LLVMTypeConverter *typeConverter,
                              ConversionPatternRewriter &rewriter);

LogicalResult convertPPUMmaV2DotScaled(triton::DotScaledOp op,
                                       triton::DotScaledOp::Adaptor adaptor,
                                       const LLVMTypeConverter *typeConverter,
                                       ConversionPatternRewriter &rewriter);

namespace {
struct ScaledDotOpConversion
    : public ConvertOpToLLVMPattern<triton::DotScaledOp> {
  using ConvertOpToLLVMPattern<triton::DotScaledOp>::ConvertOpToLLVMPattern;

  ScaledDotOpConversion(LLVMTypeConverter &converter, int computeCapability,
                        PatternBenefit benefit)
      : ConvertOpToLLVMPattern<triton::DotScaledOp>(converter, benefit),
        computeCapability(computeCapability) {}

  LogicalResult
  matchAndRewrite(triton::DotScaledOp op, triton::DotScaledOp::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    PPUMmaEncodingAttr mmaLayout = dyn_cast<PPUMmaEncodingAttr>(
        cast<RankedTensorType>(op.getResult().getType()).getEncoding());
    if (mmaLayout) {
      if (mmaLayout.isPPU0015())
        return convertPPUMmaV2DotScaled(op, adaptor, getTypeConverter(),
                                        rewriter);

      llvm::report_fatal_error(
          "Unsupported MMA kind found when converting DotScaledOp to LLVM.");
    }
    llvm::report_fatal_error(
        "Unsupported DotScaledOp found when converting TritonGPU to LLVM.");
  }

private:
  int computeCapability;
};

struct DotOpConversion : public ConvertOpToLLVMPattern<triton::DotOp> {
  using ConvertOpToLLVMPattern<triton::DotOp>::ConvertOpToLLVMPattern;

  DotOpConversion(LLVMTypeConverter &converter, int computeCapability,
                  PatternBenefit benefit)
      : ConvertOpToLLVMPattern<triton::DotOp>(converter, benefit),
        computeCapability(computeCapability) {}

  LogicalResult
  matchAndRewrite(triton::DotOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    // D = A * B + C
    Value A = op.getA();
    Value D = op.getResult();

    PPUMmaEncodingAttr mmaLayout = dyn_cast<PPUMmaEncodingAttr>(
        cast<RankedTensorType>(D.getType()).getEncoding());
    if (mmaLayout) {
      if (mmaLayout.getVersionMajor() == 1) {
        return convertPPUMmaV1(op, adaptor, getTypeConverter(), rewriter);
      } else if (mmaLayout.getVersionMajor() == 2) {
        return convertPPUMmaV2(op, adaptor, getTypeConverter(), rewriter);
      }

      llvm::report_fatal_error(
          "Unsupported MMA kind found when converting DotOp to LLVM.");
    }

    if (isa<BlockedEncodingAttr>(
            cast<RankedTensorType>(D.getType()).getEncoding()))
      return convertFMADot(op, adaptor, getTypeConverter(), rewriter);

    llvm::report_fatal_error(
        "Unsupported DotOp found when converting TritonGPU to LLVM.");
  }

private:
  int computeCapability;
};
} // namespace

void mlir::triton::ppu::populateDotOpToLLVMPatterns(
    LLVMTypeConverter &typeConverter, RewritePatternSet &patterns,
    int computeCapability, PatternBenefit benefit) {
  patterns.add<DotOpConversion>(typeConverter, computeCapability, benefit);
  patterns.add<ScaledDotOpConversion>(typeConverter, computeCapability,
                                      benefit);
}
