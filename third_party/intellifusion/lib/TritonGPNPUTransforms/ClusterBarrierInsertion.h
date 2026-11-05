#ifndef TRITON_INTELLIFUSION_TRANSFORMS_CLUSTERBARRIERINSERTION_H_
#define TRITON_INTELLIFUSION_TRANSFORMS_CLUSTERBARRIERINSERTION_H_

#include "mlir/Support/LogicalResult.h"
#include "triton/Analysis/Allocation.h"

namespace mlir::triton::Intellifusion {

inline constexpr int kGPNPUClusterComputeCapability = 100;

void runGPNPUClusterBarrierInsertion(ModuleAllocation &moduleAllocation);

LogicalResult
runGPNPUCrossCTAMBarrierInitSyncInsertion(ModuleAllocation &moduleAllocation);

} // namespace mlir::triton::Intellifusion

#endif // TRITON_INTELLIFUSION_TRANSFORMS_CLUSTERBARRIERINSERTION_H_
