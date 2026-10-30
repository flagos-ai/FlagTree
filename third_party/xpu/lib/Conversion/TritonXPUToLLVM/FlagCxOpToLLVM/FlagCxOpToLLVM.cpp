#include "triton/Conversion/TritonXPUToLLVM/FlagCxOpToLLVM/FlagCxOpToLLVM.h"

#include "triton/Conversion/TritonXPUToLLVM/FlagCxOpToLLVM/DeviceIntraBarrierOpToLLVM.h"
#include "triton/Conversion/TritonXPUToLLVM/FlagCxOpToLLVM/GetLocalRankOpToLLVM.h"
#include "triton/Conversion/TritonXPUToLLVM/GetDeviceIdToFlagCX.h"

namespace mlir::triton::xpu {

void populateFlagCxOpToLLVMPatterns(LLVMTypeConverter &typeConverter,
                                    RewritePatternSet &patterns,
                                    PatternBenefit benefit) {
  populateGetDeviceIdOpToFlagCXPatterns(typeConverter, patterns, benefit);
  populateGetLocalRankOpToLLVMPatterns(typeConverter, patterns, benefit);
  populateGetNumPesOpToLLVMPatterns(typeConverter, patterns, benefit);
  populateDeviceIntraBarrierOpToLLVMPatterns(typeConverter, patterns, benefit);
}

} // namespace mlir::triton::xpu
