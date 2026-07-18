#ifndef TRITON_DIALECT_TRITONMETAXGPU_TRANSFORMS_PASSES_H_
#define TRITON_DIALECT_TRITONMETAXGPU_TRANSFORMS_PASSES_H_

#include "mlir/Pass/Pass.h"

namespace mlir {

std::unique_ptr<Pass> createTritonMETAXGPUAccelerateMatmulPass(
    int numStages = 2, bool disablePrefetch = false, bool storeCoalesce = false,
    int computeCapability = 80);

std::unique_ptr<Pass> createTritonMETAXGPUPipelineMACAPass(
    int numStages = 2, int pipelineLoadNum = -1, bool isFullStage = false,
    bool isSingleShm = false);

std::unique_ptr<Pass> createTritonMETAXGPUPipelineAsyncBasePass(
    int numStages = 2, bool isFullStage = false, bool mixed = false);
std::unique_ptr<Pass>
createTritonMETAXGPUPipelineAsyncTNPass(int numStages = 2, int innerStageM = 0,
                                        int innerStageN = 0);
std::unique_ptr<Pass>
createTritonMETAXGPUPipelineAsyncTTPass(int numStages = 2);
std::unique_ptr<Pass>
createTritonMETAXGPUAddPtrOptPass(int numStages = 2, bool isFullStage = false,
                                  bool mixed = false);

std::unique_ptr<Pass> createTritonMETAXGPUChangeTransOpGraphPass();

std::unique_ptr<Pass> createTritonMETAXGPUChangeLayoutFromRepNToElemNPass();

std::unique_ptr<Pass> createTritonMETAXGPUChangeLayoutForConstancyLoadPass();

std::unique_ptr<Pass> createTritonMETAXGPUOptimizeCStorePass(int numStages = 2);

std::unique_ptr<Pass> createTritonMETAXGPUChangeLayoutForInt8Pass(
    int numStages = 2, std::string pipeline = std::string());

std::unique_ptr<Pass>
createTritonMETAXGPUOptimizeSmemUsage(bool forceNoVectorize = false);

/// Generate the code for registering passes.
#define GEN_PASS_REGISTRATION
#include "TritonMETAXGPUTransforms/Passes.h.inc"

} // namespace mlir
#endif
