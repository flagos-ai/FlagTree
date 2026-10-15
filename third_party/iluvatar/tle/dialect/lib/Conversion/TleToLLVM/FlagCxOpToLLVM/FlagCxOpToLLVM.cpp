#include "Conversion/TleToLLVM/FlagCxOpToLLVM/FlagCxOpToLLVM.h"

#include "Conversion/TleToLLVM/FlagCxOpToLLVM/DeviceIntraBarrierOpToLLVM.h"
#include "Conversion/TleToLLVM/FlagCxOpToLLVM/GetLocalRankOpToLLVM.h"
#include "Conversion/TleToLLVM/GetDeviceIdToFlagCX.h"

namespace mlir::triton::iluvatar_tle {

void populateFlagCxOpToLLVMPatterns(LLVMTypeConverter &typeConverter,
                                    RewritePatternSet &patterns,
                                    PatternBenefit benefit) {
#ifdef FLAGCX_ENABLED
  mlir::triton::iluvatar_tle::populateGetDeviceIdOpToFlagCxPatterns(
      typeConverter, patterns, benefit);
  mlir::triton::iluvatar_tle::populateGetLocalRankOpToLLVMPatterns(
      typeConverter, patterns, benefit);
  mlir::triton::iluvatar_tle::populateGetNumPesOpToLLVMPatterns(
      typeConverter, patterns, benefit);
  mlir::triton::iluvatar_tle::populateDeviceIntraBarrierOpToLLVMPatterns(
      typeConverter, patterns, benefit);
#endif
}

} // namespace mlir::triton::iluvatar_tle
