#ifndef TRITONMUSA_TRANSFORMS_SQMMA_PIPELINE_UTILS_H
#define TRITONMUSA_TRANSFORMS_SQMMA_PIPELINE_UTILS_H

#include "mlir/IR/BuiltinOps.h"

namespace mlir::triton::musa::pipeline {

void pipelineSqmma(ModuleOp moduleOp, unsigned numStages);

} // namespace mlir::triton::musa::pipeline

#endif // TRITONMUSA_TRANSFORMS_SQMMA_PIPELINE_UTILS_H
