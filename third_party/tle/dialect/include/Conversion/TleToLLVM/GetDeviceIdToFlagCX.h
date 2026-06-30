#include "mlir/Conversion/LLVMCommon/TypeConverter.h"
// #include "triton/Conversion/TritonGPUToLLVM/TargetInfoBase.h"

namespace mlir::triton::tle {

void populateGetDeviceIdOpToFlagCxPatterns(LLVMTypeConverter &typeConverter,
                                           RewritePatternSet &patterns,
                                           PatternBenefit benefit);

} // namespace mlir::triton::tle
