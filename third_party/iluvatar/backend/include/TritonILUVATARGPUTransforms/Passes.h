#ifndef TRITON_DIALECT_TRITONILUVATARGPU_TRANSFORMS_PASSES_H_
#define TRITON_DIALECT_TRITONILUVATARGPU_TRANSFORMS_PASSES_H_

#include "mlir/Pass/Pass.h"

namespace mlir {

std::unique_ptr<Pass>
createTritonILUVATARGPUAccelerateMatmulPass(int computeCapability = 80,
                                            unsigned useSme = 0);

std::unique_ptr<Pass>
createTritonILUVATARGPUSmeLoadPass(int computeCapability = 80);

std::unique_ptr<Pass> createTritonILUVATARGPUOptimizeEpiloguePass();

std::unique_ptr<Pass> createTritonILUVATARGPUMMAReduceThreadLocalityPass();

/// Generate the code for registering passes.
#define GEN_PASS_REGISTRATION
#include "TritonILUVATARGPUTransforms/Passes.h.inc"

} // namespace mlir
#endif
