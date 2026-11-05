#include "Passes.h"

#include "mlir/Pass/PassRegistry.h"

namespace mlir::triton::Intellifusion {

void registerGPNPUTransformPasses() {
  registerPass([] { return createGPNPUPlanCTAPass(); });
  registerPass([] { return createGPNPUAccelerateMatmulPass(); });
  registerPass([] { return createGPNPUOptimizeDescriptorEncodingPass(); });
  registerPass([] { return createGPNPUPromoteLHSToTMemPass(); });
  registerPass([] { return createGPNPURemoveTMEMTokensPass(); });
  registerPass([] { return createGPNPUOptimizeTMemLayoutsPass(); });
  registerPass([] { return createGPNPUFuseTMEMLoadReducePass(); });
  registerPass([] { return createGPNPUTMALoweringPass(); });
  registerPass([] { return createGPNPUInterleaveTMemPass(); });
  registerPass([] { return createGPNPUFenceInsertionPass(); });
  registerPass([] { return createGPNPUMMALoweringPass(); });
  registerPass([] { return createGPNPUTensorMemoryAllocationPass(); });
  registerPass([] { return createGPNPUProxyFenceInsertionPass(); });
  registerPass([] { return createGPNPUCheckMatmulTwoCTAPass(); });
  registerPass([] { return createGPNPUTMemBarrierInsertionPass(); });
  registerPass([] { return createGPNPUDotStrategyPass(); });
  registerPass([] { return createGPNPUStageScaledFp4Pass(); });
}

} // namespace mlir::triton::Intellifusion
