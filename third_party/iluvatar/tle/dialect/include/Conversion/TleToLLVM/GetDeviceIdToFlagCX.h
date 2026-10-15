#ifndef TRITON_THIRD_PARTY_ILUVATAR_TLE_CONVERSION_GETDEVICEIDTOFLAGCX_H_
#define TRITON_THIRD_PARTY_ILUVATAR_TLE_CONVERSION_GETDEVICEIDTOFLAGCX_H_

#include "mlir/Conversion/LLVMCommon/TypeConverter.h"
#include "mlir/IR/PatternMatch.h"

namespace mlir::triton::iluvatar_tle {

void populateGetDeviceIdOpToFlagCxPatterns(LLVMTypeConverter &typeConverter,
                                           RewritePatternSet &patterns,
                                           PatternBenefit benefit);

} // namespace mlir::triton::iluvatar_tle

#endif // TRITON_THIRD_PARTY_ILUVATAR_TLE_CONVERSION_GETDEVICEIDTOFLAGCX_H_
