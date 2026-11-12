#include "Passes.h"

#include "mlir/Pass/PassRegistry.h"

namespace mlir::triton::Intellifusion {

void registerGPNPUConversionPasses() {
  registerPass([] { return createGPNPUAllocateSharedMemoryPass(); });
  registerPass([] { return createGPNPUCompatToLLVMPass(); });
  registerPass([] { return createConvertTritonGPNPUToLLVM(/*numWarps=*/4); });
}

} // namespace mlir::triton::Intellifusion
