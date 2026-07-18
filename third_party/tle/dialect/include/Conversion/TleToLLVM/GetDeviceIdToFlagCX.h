#ifndef TLE_CONVERSION_TLETOLLVM_GETDEVICEIDTOFLAGCX_H
#define TLE_CONVERSION_TLETOLLVM_GETDEVICEIDTOFLAGCX_H

#include "mlir/Conversion/LLVMCommon/TypeConverter.h"

namespace mlir::triton::tle {

void populateGetDeviceIdOpToFlagCxPatterns(LLVMTypeConverter &typeConverter,
                                           RewritePatternSet &patterns,
                                           PatternBenefit benefit);

} // namespace mlir::triton::tle

#endif
