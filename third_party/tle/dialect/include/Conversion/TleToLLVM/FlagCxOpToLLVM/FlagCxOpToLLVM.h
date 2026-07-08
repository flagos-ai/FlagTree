#include "tle/dialect/include/Conversion/TleToLLVM/FlagCxOpToLLVM/DeviceIntraBarrierOpToLLVM.h"
#include "tle/dialect/include/Conversion/TleToLLVM/FlagCxOpToLLVM/GetLocalRankOpToLLVM.h"

namespace mlir::triton::tle {
void populateFlagCxOpToLLVMPatterns(LLVMTypeConverter &typeConverter,
                                    RewritePatternSet &patterns,
                                    PatternBenefit benefit);
}
