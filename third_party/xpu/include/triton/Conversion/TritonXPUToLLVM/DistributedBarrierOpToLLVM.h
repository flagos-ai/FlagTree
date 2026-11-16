#ifndef TRITON_XPU_DISTRIBUTED_BARRIER_TO_LLVM_H_
#define TRITON_XPU_DISTRIBUTED_BARRIER_TO_LLVM_H_

#include "triton/Conversion/TritonXPUToLLVM/FlagCxOpToLLVM/DeviceIntraBarrierOpToLLVM.h"

namespace mlir::triton::xpu {

inline void
populateDistributedBarrierOpToLLVMPatterns(LLVMTypeConverter &typeConverter,
                                           RewritePatternSet &patterns,
                                           PatternBenefit benefit) {
  populateDeviceIntraBarrierOpToLLVMPatterns(typeConverter, patterns, benefit);
}

} // namespace mlir::triton::xpu

#endif // TRITON_XPU_DISTRIBUTED_BARRIER_TO_LLVM_H_
