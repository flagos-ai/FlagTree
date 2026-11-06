#ifndef TRITON_XPU_GET_DEVICE_ID_TO_FLAGCX_H_
#define TRITON_XPU_GET_DEVICE_ID_TO_FLAGCX_H_

#include "mlir/Conversion/LLVMCommon/TypeConverter.h"
#include "mlir/IR/PatternMatch.h"

namespace mlir::triton::xpu {

void populateGetDeviceIdOpToFlagCXPatterns(LLVMTypeConverter &typeConverter,
                                           RewritePatternSet &patterns,
                                           PatternBenefit benefit);

}

#endif
