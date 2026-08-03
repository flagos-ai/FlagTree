#ifndef TLE_CONVERSION_TLETOLLVM_PUTMEMORVALUEOPTOLLVM_H
#define TLE_CONVERSION_TLETOLLVM_PUTMEMORVALUEOPTOLLVM_H

#include "mlir/Conversion/LLVMCommon/TypeConverter.h"

namespace mlir::triton::tle {

void populatePutMemOrValueOpToLLVMPatterns(LLVMTypeConverter &typeConverter,
                                           RewritePatternSet &patterns,
                                           PatternBenefit benefit);
} // namespace mlir::triton::tle

#endif
