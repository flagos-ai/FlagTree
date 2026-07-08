#include "tle/dialect/include/Conversion/TleToLLVM/FlagCxOpToLLVM/DeviceIntraBarrierOpToLLVM.h"
#include "tle/dialect/include/Conversion/TleToLLVM/FlagCxOpToLLVM/GetLocalRankOpToLLVM.h"
#include "tle/dialect/include/Conversion/TleToLLVM/GetDeviceIdToFlagCX.h"

namespace mlir::triton::tle {
void populateFlagCxOpToLLVMPatterns(LLVMTypeConverter &typeConverter,
                                    RewritePatternSet &patterns,
                                    PatternBenefit benefit) {
#ifdef FLAGCX_ENABLED
  mlir::triton::tle::populateGetDeviceIdOpToFlagCxPatterns(typeConverter,
                                                           patterns, benefit);
  mlir::triton::tle::populateGetLocalRankOpToLLVMPatterns(typeConverter,
                                                          patterns, benefit);
  mlir::triton::tle::populateGetNumPesOpToLLVMPatterns(typeConverter, patterns,
                                                       benefit);
  mlir::triton::tle::populateDeviceIntraBarrierOpToLLVMPatterns(
      typeConverter, patterns, benefit);
#endif
}

} // namespace mlir::triton::tle
