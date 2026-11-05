#ifndef TRITON_GPNPUTOLLVM_PASSES_H
#define TRITON_GPNPUTOLLVM_PASSES_H

#include "mlir/Pass/Pass.h"
#include <memory>

namespace mlir {
namespace triton {
namespace Intellifusion {

// Create the GPNPU to LLVM conversion pass.
std::unique_ptr<Pass> createConvertTritonGPNPUToLLVM(int numWarps);

std::unique_ptr<Pass> createGPNPUAllocateSharedMemoryPass();
std::unique_ptr<Pass> createGPNPUCompatToLLVMPass();

void registerGPNPUConversionPasses();

} // namespace Intellifusion
} // namespace triton
} // namespace mlir

#endif // TRITON_GPNPUTOLLVM_PASSES_H
