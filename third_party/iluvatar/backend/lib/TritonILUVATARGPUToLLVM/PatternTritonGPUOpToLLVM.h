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

#ifndef TRITON_THIRD_PARTY_ILUVATAR_LIB_TRITONILUVATARGPUTOLLVM_PATTERNTRITONGPUOPTOLLVM_H_
#define TRITON_THIRD_PARTY_ILUVATAR_LIB_TRITONILUVATARGPUTOLLVM_PATTERNTRITONGPUOPTOLLVM_H_

#include "TargetInfo.h"
#include "mlir/Conversion/LLVMCommon/TypeConverter.h"
#include "triton/Analysis/Allocation.h"
#include "triton/Analysis/AxisInfo.h"

namespace mlir::triton::ILUVATAR {
void populateConvertLayoutOpToLLVMPatterns(LLVMTypeConverter &typeConverter,
                                           const TargetInfo &targetInfo,
                                           RewritePatternSet &patterns,
                                           PatternBenefit benefit);

void populateMemoryOpToLLVMPatterns(LLVMTypeConverter &typeConverter,
                                    RewritePatternSet &patterns,
                                    const TargetInfo &targetInfo,
                                    PatternBenefit benefit);

void populateDotOpToLLVMPatterns(LLVMTypeConverter &typeConverter,
                                 RewritePatternSet &patterns,
                                 ModuleAxisInfoAnalysis &axisInfoAnalysis,
                                 PatternBenefit benefit);
void populateElementwiseOpToLLVMPatterns(
    LLVMTypeConverter &typeConverter, RewritePatternSet &patterns, bool ftz,
    ModuleAxisInfoAnalysis &axisInfoAnalysis, ModuleAllocation &allocation,
    const TargetInfo &targetInfo, PatternBenefit benefit);

// Manipulates with execution mode register which is per-wavefront one.
// The register controls execution of instructions - e.g., rounding modes,
// exception handling, etc.
void adjustModeRegister(ModuleOp mod, const TargetInfo &targetInfo);

void populateLoadStoreOpToLLVMPatterns(LLVMTypeConverter &typeConverter,
                                       const TargetInfo &targetInfo,
                                       RewritePatternSet &patterns,
                                       ModuleAxisInfoAnalysis &axisInfoAnalysis,
                                       PatternBenefit benefit);

void populateSPMDOpToLLVMPattern(LLVMTypeConverter &typeConverter,
                                 RewritePatternSet &patterns,
                                 PatternBenefit benefit);

void populateUpcastMXFPToLLVMPatterns(LLVMTypeConverter &typeConverter,
                                      RewritePatternSet &patterns,
                                      const TargetInfo &targetInfo,
                                      PatternBenefit benefit);

void populateFp4ToFpToLLVMPatterns(LLVMTypeConverter &typeConverter,
                                   RewritePatternSet &patterns,
                                   const TargetInfo &targetInfo,
                                   PatternBenefit benefit);

void populateMaskedOpsToLLVMPatterns(RewritePatternSet &patterns,
                                     const TargetInfo &targetInfo);

void populateTensorPtrOpsToLLVMPatterns(LLVMTypeConverter &typeConverter,
                                        RewritePatternSet &patterns,
                                        PatternBenefit benefit);

void populateBarrierOpToLLVMPatterns(LLVMTypeConverter &typeConverter,
                                     RewritePatternSet &patterns,
                                     PatternBenefit benefit);
} // namespace mlir::triton::ILUVATAR

#endif // TRITON_THIRD_PARTY_ILUVATAR_LIB_TRITONILUVATARGPUTOLLVM_PATTERNTRITONGPUOPTOLLVM_H_
