#include "third_party/hcu/include/TritonHCUGPUToLLVM/PatternTritonHCUGPUToLLVM.h"
#include "triton/Conversion/TritonGPUToLLVM/PatternTritonGPUOpToLLVM.h"

namespace mlir::triton::HCU {
void populateTritonHCUGPUToLLVMPatterns(LLVMTypeConverter &typeConverter,
                                        RewritePatternSet &patterns,
                                        PatternBenefit benefit) {
  populateExtractSliceOpToLLVMPatterns(typeConverter, patterns, benefit);
  populateInThreadTransposeOpToTTGPatterns(patterns, benefit);
  populateConcatOpToLLVMPatterns(typeConverter, patterns, benefit);
  populateScaledUpcastOpToLLVMPatterns(typeConverter, patterns, benefit);
}
} // namespace mlir::triton::HCU
