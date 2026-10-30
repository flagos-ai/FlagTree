#ifndef TRITON_XPU_DEVICE_INTRA_BARRIER_OP_TO_LLVM_H_
#define TRITON_XPU_DEVICE_INTRA_BARRIER_OP_TO_LLVM_H_

#include "mlir/Conversion/LLVMCommon/TypeConverter.h"
#include "mlir/IR/PatternMatch.h"

namespace mlir::triton::xpu {

void populateDeviceIntraBarrierOpToLLVMPatterns(
    LLVMTypeConverter &typeConverter, RewritePatternSet &patterns,
    PatternBenefit benefit);

} // namespace mlir::triton::xpu

#endif // TRITON_XPU_DEVICE_INTRA_BARRIER_OP_TO_LLVM_H_
