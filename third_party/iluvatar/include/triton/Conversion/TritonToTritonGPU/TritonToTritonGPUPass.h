#ifndef TRITON_CONVERSION_TRITONTOTRITONGPU_TRITONTOTRITONGPUPASS_H
#define TRITON_CONVERSION_TRITONTOTRITONGPU_TRITONTOTRITONGPUPASS_H

#include <memory>
#include <optional>

#include "flagtree_spec.h"

namespace mlir {

class ModuleOp;
template <typename T> class OperationPass;

namespace triton {

constexpr static char AttrNumWarpsName[] = "triton_gpu.num-warps";
constexpr static char AttrNumCTAsName[] = "triton_gpu.num-ctas";
constexpr static char AttrTargetName[] = "triton_gpu.target";

constexpr static char AttrNumThreadsPerWarp[] = "triton_gpu.threads-per-warp";

#ifdef FLAGTREE_SPEC_Conversion_TritonToTritonGPU_TritonToTritonGPUPass_AttrNumStagesForDot
constexpr static char AttrNumStagesForDot[] = "triton_gpu.dot.num-stages";
#endif

// Create the pass with numWarps passed from cl::opt.
std::unique_ptr<OperationPass<ModuleOp>> createConvertTritonToTritonGPUPass();

// Create the pass with numWarps set explicitly.
#ifdef FLAGTREE_SPEC_Conversion_TritonToTritonGPU_TritonToTritonGPUPass_createConvertTritonToTritonGPUPass_ARG
std::unique_ptr<OperationPass<ModuleOp>>
createConvertTritonToTritonGPUPass(const std::string &target, int numWarps,
                                   int threadsPerWarp = 32, int numCTAs = 1,
                                   FLAGTREE_SPEC_Conversion_TritonToTritonGPU_TritonToTritonGPUPass_createConvertTritonToTritonGPUPass_ARG spec_arg = 1);
#else
std::unique_ptr<OperationPass<ModuleOp>>
createConvertTritonToTritonGPUPass(const std::string &target, int numWarps,
                                   int threadsPerWarp = 32, int numCTAs = 1);
#endif

} // namespace triton
} // namespace mlir

#endif
