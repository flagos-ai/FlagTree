#ifndef TRITONMUSA_TRANSFORMS_TME_PIPELINE_UTILS_H
#define TRITONMUSA_TRANSFORMS_TME_PIPELINE_UTILS_H

#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Support/LogicalResult.h"
#include "triton/Dialect/TritonGPU/Transforms/Schedule.h"

namespace mlir::triton::musa::pipeline {

scf::ForOp lowerTMADescriptors(scf::ForOp forOp,
                               mlir::triton::CoarseSchedule &schedule);
FailureOr<bool> pipelineTMEStores(scf::ForOp forOp);

} // namespace mlir::triton::musa::pipeline

#endif // TRITONMUSA_TRANSFORMS_TME_PIPELINE_UTILS_H
