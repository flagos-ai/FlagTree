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

// Create the pass with numWarps passed from cl::opt.
std::unique_ptr<OperationPass<ModuleOp>> createConvertTritonToTritonGPUPass();

// Create the pass with numWarps set explicitly.
std::unique_ptr<OperationPass<ModuleOp>>
createConvertTritonToTritonGPUPass(const std::string &target, int numWarps,
#ifndef FLAGTREE_SPEC_Conversion_TritonToTritonGPU_TritonToTritonGPUPass_createConvertTritonToTritonGPUPass_ARG
                                   int threadsPerWarp = 32, int numCTAs = 1);
#else
                                   int threadsPerWarp = 32, int numCTAs = 1,
                                   FLAGTREE_SPEC_Conversion_TritonToTritonGPU_TritonToTritonGPUPass_createConvertTritonToTritonGPUPass_ARG
                                       spec_arg = 1);
#endif

#ifdef FLAGTREE_SPEC_Conversion_TritonToTritonGPU_TritonToTritonGPUPass_ConvertTritonToTritonGPU_setAttrNumStagesForDot
void ConvertTritonToTritonGPU_setAttrNumStagesForDot(ModuleOp &mod,
                                                     IntegerType i32_ty,
                                                     int numStages);
#endif

} // namespace triton
} // namespace mlir

#endif
