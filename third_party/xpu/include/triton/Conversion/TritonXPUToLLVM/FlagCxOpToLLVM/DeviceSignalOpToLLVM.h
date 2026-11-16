#ifndef TRITON_CONVERSION_TRITONXPU_TO_LLVM_FLAGCX_DEVICE_SIGNAL_H
#define TRITON_CONVERSION_TRITONXPU_TO_LLVM_FLAGCX_DEVICE_SIGNAL_H

#include "mlir/Conversion/LLVMCommon/TypeConverter.h"
#include "mlir/Transforms/DialectConversion.h"

namespace mlir::triton::xpu {

void populateDeviceSignalOpToLLVMPatterns(LLVMTypeConverter &typeConverter,
                                          RewritePatternSet &patterns,
                                          PatternBenefit benefit);

} // namespace mlir::triton::xpu

#endif // TRITON_CONVERSION_TRITONXPU_TO_LLVM_FLAGCX_DEVICE_SIGNAL_H
