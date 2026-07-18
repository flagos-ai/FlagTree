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

#ifndef TRITON_CONVERSION_TRITONGPU_TO_LLVM_PATTERNS_TRITON_GPU_OP_TO_LLVM_H
#define TRITON_CONVERSION_TRITONGPU_TO_LLVM_PATTERNS_TRITON_GPU_OP_TO_LLVM_H

#include "TargetInfoBase.h"
#include "mlir/Conversion/LLVMCommon/TypeConverter.h"
#include "triton/Analysis/AxisInfo.h"

using namespace mlir;
using namespace mlir::triton;

using ::mlir::triton::gpu::BlockedEncodingAttr;
LogicalResult convertFMADot(triton::DotOp op, triton::DotOp::Adaptor adaptor,
                            const LLVMTypeConverter *typeConverter,
                            ConversionPatternRewriter &rewriter);
namespace mlir {
namespace triton {

constexpr int patternBenefitDefault = 1;
constexpr int patternBenefitPrioritizeOverLLVMConversions = 10;
constexpr int patternBenefitClampOptimizedPattern = 20;
constexpr int patternBenefitConvertLayoutOptimizedPattern = 20;
constexpr int patternBenefitNvidiaTensorCoreSubviewPattern = 20;

void populateElementwiseOpToLLVMPatterns(
    LLVMTypeConverter &typeConverter, RewritePatternSet &patterns,
    ModuleAxisInfoAnalysis &axisInfoAnalysis, const TargetInfoBase &targetInfo,
    PatternBenefit benefit);

// The given callback is invoked at the end of a successful rewrite. The
// callback receives 1) the current source op, 2) the number of issued LLVM
// instructions and 3) their input types. Each MLIR backend can provide a
// callback and, thus, handle backend-specific behaviors.
void populateMemoryOpToLLVMPatterns(LLVMTypeConverter &typeConverter,
                                    const TargetInfoBase &targetInfo,
                                    RewritePatternSet &patterns,
                                    PatternBenefit benefit);

void populateAssertOpToLLVMPattern(LLVMTypeConverter &typeConverter,
                                   RewritePatternSet &patterns,
                                   const TargetInfoBase &targetInfo,
                                   PatternBenefit benefit);

void populateMakeRangeOpToLLVMPattern(LLVMTypeConverter &typeConverter,
                                      const TargetInfoBase &targetInfo,
                                      RewritePatternSet &patterns,
                                      PatternBenefit benefit);

void populateViewOpToLLVMPatterns(LLVMTypeConverter &typeConverter,
                                  RewritePatternSet &patterns,
                                  PatternBenefit benefit);

void populateMinMaxFOpToLLVMPattern(LLVMTypeConverter &typeConverter,
                                    RewritePatternSet &patterns,
                                    ModuleAxisInfoAnalysis &axisInfoAnalysis,
                                    bool hwNanPropagationSupported,
                                    PatternBenefit benefit);
void populateClampFOpToLLVMPattern(LLVMTypeConverter &typeConverter,
                                   RewritePatternSet &patterns,
                                   ModuleAxisInfoAnalysis &axisInfoAnalysis,
                                   const TargetInfoBase &targetInfo,
                                   PatternBenefit benefit);

void populateHistogramOpToLLVMPatterns(LLVMTypeConverter &typeConverter,
                                       RewritePatternSet &patterns,
                                       const TargetInfoBase &targetInfo,
                                       PatternBenefit benefit);
void populateReduceOpToLLVMPatterns(LLVMTypeConverter &typeConverter,
                                    RewritePatternSet &patterns,
                                    const TargetInfoBase &targetInfo,
                                    PatternBenefit benefit);
void populateScanOpToLLVMPatterns(LLVMTypeConverter &typeConverter,
                                  RewritePatternSet &patterns,
                                  const TargetInfoBase &targetInfo,
                                  PatternBenefit benefit);
void populateGatherOpToLLVMPatterns(LLVMTypeConverter &typeConverter,
                                    RewritePatternSet &patterns,
                                    const TargetInfoBase &targetInfo,
                                    PatternBenefit benefit);

void populateConvertLayoutOpToLLVMPatterns(LLVMTypeConverter &typeConverter,
                                           const TargetInfoBase &targetInfo,
                                           RewritePatternSet &patterns,
                                           PatternBenefit benefit);

void populateControlFlowOpToLLVMPattern(LLVMTypeConverter &typeConverter,
                                        RewritePatternSet &patterns,
                                        const TargetInfoBase &targetInfo,
                                        PatternBenefit benefit);

void populateSPMDOpToLLVMPattern(LLVMTypeConverter &typeConverter,
                                 RewritePatternSet &patterns,
                                 const TargetInfoBase &targetInfo,
                                 PatternBenefit benefit);

void populateFuncOpConversionPattern(LLVMTypeConverter &typeConverter,
                                     RewritePatternSet &patterns,
                                     const TargetInfoBase &targetInfo,
                                     PatternBenefit benefit);

void populatePrintOpToLLVMPattern(LLVMTypeConverter &typeConverter,
                                  RewritePatternSet &patterns,
                                  const TargetInfoBase &targetInfo,
                                  PatternBenefit benefit);

void populateInstrumentationToLLVMPatterns(LLVMTypeConverter &typeConverter,
                                           const TargetInfoBase &targetInfo,
                                           RewritePatternSet &patterns,
                                           PatternBenefit benefit);

} // namespace triton
} // namespace mlir

#endif
