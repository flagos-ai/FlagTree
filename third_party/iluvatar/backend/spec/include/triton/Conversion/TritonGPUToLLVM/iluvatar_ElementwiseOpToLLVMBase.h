#ifndef ILUVATAR_TRITON_CONVERSION_TRITONGPU_TO_ELEMENTWISE_OP_H
#define ILUVATAR_TRITON_CONVERSION_TRITONGPU_TO_ELEMENTWISE_OP_H

namespace mlir {
class LLVMTypeConverter;
class RewritePatternSet;
class PatternBenefit;
namespace triton {
class ModuleAxisInfoAnalysis;
class TargetInfoBase;
namespace ILUVATAR {
void populateExp2OpToLLVMPatterns(LLVMTypeConverter &typeConverter,
                                  RewritePatternSet &patterns,
                                  ModuleAxisInfoAnalysis &axisInfoAnalysis,
                                  const TargetInfoBase &targetInfo,
                                  PatternBenefit benefit);
} // namespace ILUVATAR
} // namespace triton
} // namespace mlir

#define FLAGTREE_SPEC_ElementwiseOpConversionBase_maybeDeduplicate
#define FLAGTREE_SPEC_ElementwiseOpConversionBase_matchAndRewrite
#define FLAGTREE_SPEC_Conversion_TritonGPUToLLVM_ElementwiseOpToLLVMBase_reorderValues
#define FLAGTREE_SPEC_Conversion_TritonGPUToLLVM_ElementwiseOpToLLVMBase_unpackI32
#define FLAGTREE_SPEC_Conversion_TritonGPUToLLVM_ElementwiseOpToLLVMBase_packI32
#define FLAGTREE_SPEC_Conversion_TritonGPUToLLVM_ElementwiseOpToLLVM_exp2      \
  ::mlir::triton::ILUVATAR::populateExp2OpToLLVMPatterns(                      \
      typeConverter, patterns, axisInfoAnalysis, targetInfo, benefit)

#endif // ILUVATAR_TRITON_CONVERSION_TRITONGPU_TO_ELEMENTWISE_OP_H
