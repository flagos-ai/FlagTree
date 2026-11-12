#ifndef TRITON_GPNPUTOLLVM_PATTERN_POPULATION_H
#define TRITON_GPNPUTOLLVM_PATTERN_POPULATION_H

#include "mlir/Conversion/LLVMCommon/TypeConverter.h"
#include "mlir/Rewrite/FrozenRewritePatternSet.h"
#include "mlir/Transforms/DialectConversion.h"
#include "triton/Analysis/AxisInfo.h"
#include "triton/Conversion/TritonGPUToLLVM/TargetInfoBase.h"

namespace mlir {
namespace triton {

namespace Intellifusion {

class TargetInfo;

void populateGPNPUArithToLLVMPatterns(RewritePatternSet &patterns,
                                      LLVMTypeConverter &typeConverter);

void populateGPNPUMemoryToLLVMPatterns(RewritePatternSet &patterns,
                                       LLVMTypeConverter &typeConverter,
                                       const TargetInfo &targetInfo,
                                       ModuleAxisInfoAnalysis &axisInfoAnalysis,
                                       PatternBenefit benefit);

void populateGPNPUSPMDOpToLLVMPatterns(LLVMTypeConverter &typeConverter,
                                       RewritePatternSet &patterns,
                                       PatternBenefit benefit);

void populateGPNPUElementwiseToLLVMPatterns(
    LLVMTypeConverter &typeConverter, RewritePatternSet &patterns,
    ModuleAxisInfoAnalysis &axisInfoAnalysis, const TargetInfoBase &targetInfo,
    PatternBenefit benefit);

void populateGPNPUMathToLLVMPatterns(LLVMTypeConverter &typeConverter,
                                     RewritePatternSet &patterns,
                                     ModuleAxisInfoAnalysis &axisInfoAnalysis,
                                     const TargetInfoBase &targetInfo,
                                     PatternBenefit benefit);

void populateGPNPUDotOpToLLVMPatterns(LLVMTypeConverter &typeConverter,
                                      RewritePatternSet &patterns,
                                      PatternBenefit benefit);

void populateGPNPUFp4ToFpToLLVMPatterns(LLVMTypeConverter &typeConverter,
                                        RewritePatternSet &patterns,
                                        PatternBenefit benefit);

void populateGPNPUConvertLayoutOpToLLVMPatterns(
    LLVMTypeConverter &typeConverter, const TargetInfoBase &targetInfo,
    RewritePatternSet &patterns, PatternBenefit benefit);

void populateGPNPUTMAToLLVMPatterns(LLVMTypeConverter &typeConverter,
                                    const TargetInfo &targetInfo,
                                    RewritePatternSet &patterns,
                                    PatternBenefit benefit);

void populateGPNPUTMABarrierOpToLLVMPatterns(LLVMTypeConverter &typeConverter,
                                             RewritePatternSet &patterns,
                                             PatternBenefit benefit,
                                             TargetInfo &targetInfo);

void populateGPNPUClusterOpToLLVMPatterns(LLVMTypeConverter &typeConverter,
                                          RewritePatternSet &patterns,
                                          PatternBenefit benefit,
                                          const TargetInfo &targetInfo);

void populateGPNPUTMALoadStoreOpToLLVMPatterns(
    LLVMTypeConverter &typeConverter, const TargetInfo &targetInfo,
    RewritePatternSet &patterns, ModuleAxisInfoAnalysis &axisInfoAnalysis,
    PatternBenefit benefit);

void populateGPNPUTensorMemoryOpToLLVMPatterns(LLVMTypeConverter &typeConverter,
                                               RewritePatternSet &patterns,
                                               PatternBenefit benefit);

void populateGPNPUTensorMemorySubviewOpToLLVMPatterns(
    LLVMTypeConverter &typeConverter, RewritePatternSet &patterns,
    PatternBenefit benefit);

void populateGPNPUTCGen5MMAOpToLLVMPatterns(LLVMTypeConverter &typeConverter,
                                            RewritePatternSet &patterns,
                                            PatternBenefit benefit,
                                            const TargetInfo &targetInfo);

} // namespace Intellifusion
} // namespace triton
} // namespace mlir

#endif // TRITON_GPNPUTOLLVM_PATTERN_POPULATION_H
