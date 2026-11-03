#include "Analysis/HCUGPUAllocation.h"
#include "TritonHCUGPUToLLVM/Passes.h"
#include "triton/Analysis/Allocation.h"
#include "triton/Analysis/Utility.h"
#include "triton/Conversion/TritonGPUToLLVM/AllocateSharedMemoryUtility.h"

using namespace mlir;
using namespace mlir::triton;
using namespace mlir::triton::HCU;

namespace mlir::triton {
#define GEN_PASS_DEF_ALLOCATEHCUGPUSHAREDMEMORY
#include "TritonHCUGPUToLLVM/Passes.h.inc"
} // namespace mlir::triton

namespace mlir::triton::HCU {
LogicalResult prepareLegacyMlsV0JointPhase(ModuleOp module);
}

namespace {

struct AllocateHCUGPUSharedMemory
    : public mlir::triton::impl::AllocateHCUGPUSharedMemoryBase<
          AllocateHCUGPUSharedMemory> {
  void runOnOperation() override {
    ModuleOp mod = getOperation();
    // Final SCF-to-CF/inliner/index conversion have already run. Compute new
    // allocation from the moved IR; never hand-install offsets or LDS size.
    if (failed(HCU::prepareLegacyMlsV0JointPhase(mod)))
      return signalPassFailure();
    ModuleAllocation allocation(mod, HCUAllocationAnalysisScratchSizeFn);

    mlir::triton::gpu::attachAllocationSizeAndOffsetAttr(mod, allocation);
  }
};

} // namespace
