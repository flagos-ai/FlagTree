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

#ifndef TRITON_CONVERSION_TRITONMETAXGPU_TO_LLVM_PATTERNS_TRITON_GPU_OP_TO_LLVM_H
#define TRITON_CONVERSION_TRITONMETAXGPU_TO_LLVM_PATTERNS_TRITON_GPU_OP_TO_LLVM_H

#include "TargetInfo.h"
#include "mlir/Conversion/LLVMCommon/TypeConverter.h"
#include "triton/Analysis/AxisInfo.h"

namespace mlir {
namespace triton {

namespace METAX {

void populateBarrierOpToLLVMPatterns(LLVMTypeConverter &typeConverter,
                                     RewritePatternSet &patterns,
                                     PatternBenefit benefit);

void populateConvertLayoutOpToLLVMPatterns(LLVMTypeConverter &typeConverter,
                                           const TargetInfo &targetInfo,
                                           RewritePatternSet &patterns,
                                           PatternBenefit benefit);

void populateConvertLayoutOpToLLVMOptimizedPatterns(
    LLVMTypeConverter &typeConverter, const TargetInfo &targetInfo,
    RewritePatternSet &patterns, PatternBenefit benefit);

void populateDotOpToLLVMPatterns(LLVMTypeConverter &typeConverter,
                                 RewritePatternSet &patterns,
                                 PatternBenefit benefit);

void populateSyncOpToLLVMPatterns(LLVMTypeConverter &typeConverter,
                                  const TargetInfo &targetInfo,
                                  RewritePatternSet &patterns,
                                  PatternBenefit benefit);

void populateElementwiseOpToLLVMPatterns(
    LLVMTypeConverter &typeConverter, RewritePatternSet &patterns,
    ModuleAxisInfoAnalysis &axisInfoAnalysis, int computeCapability,
    const TargetInfo &targetInfo, PatternBenefit benefit);

void populateLoadStoreOpToLLVMPatterns(LLVMTypeConverter &typeConverter,
                                       const TargetInfo &targetInfo,
                                       RewritePatternSet &patterns,
                                       ModuleAxisInfoAnalysis &axisInfoAnalysis,
                                       PatternBenefit benefit);

void populateScanOpToLLVMPatterns(LLVMTypeConverter &typeConverter,
                                  const TargetInfo &targetInfo,
                                  RewritePatternSet &patterns,
                                  PatternBenefit benefit);

void populateHistogramOpToLLVMPatterns(LLVMTypeConverter &typeConverter,
                                       RewritePatternSet &patterns,
                                       const TargetInfo &targetInfo,
                                       PatternBenefit benefit);

void populateReduceOpToLLVMPatterns(LLVMTypeConverter &typeConverter,
                                    const TargetInfo &targetInfo,
                                    RewritePatternSet &patterns,
                                    PatternBenefit benefit);

void populateTensorPtrOpsToLLVMPatterns(LLVMTypeConverter &typeConverter,
                                        RewritePatternSet &patterns,
                                        PatternBenefit benefit);

void populateSPMDOpToLLVMPattern(LLVMTypeConverter &typeConverter,
                                 RewritePatternSet &patterns,
                                 PatternBenefit benefit);

void populateViewOpToLLVMPatterns(LLVMTypeConverter &typeConverter,
                                  RewritePatternSet &patterns,
                                  PatternBenefit benefit);

void populateMemoryOpToLLVMPatterns(LLVMTypeConverter &typeConverter,
                                    const TargetInfoBase &targetInfo,
                                    RewritePatternSet &patterns,
                                    PatternBenefit benefit);

void populateFp4ToFpScaledToLLVMPatterns(LLVMTypeConverter &typeConverter,
                                         RewritePatternSet &patterns,
                                         PatternBenefit benefit);

} // namespace METAX
} // namespace triton
} // namespace mlir

#endif
