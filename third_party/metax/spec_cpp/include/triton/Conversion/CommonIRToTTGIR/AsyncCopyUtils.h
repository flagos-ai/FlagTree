#ifndef TRITON_METAX_COMMON_IR_ASYNC_COPY_UTILS_H
#define TRITON_METAX_COMMON_IR_ASYNC_COPY_UTILS_H

#include "triton/Analysis/AxisInfo.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"

namespace mlir::triton::metax {

bool isAsyncCopyDestinationSupported(gpu::MemDescType destination);

gpu::MemDescType getAsyncCopyLinearDestination(gpu::MemDescType destination);

unsigned getAsyncCopyVectorSize(RankedTensorType source,
                               gpu::MemDescType destination,
                               const AxisInfo *sourceInfo);

// Validate mask/zero-fill semantics and bound vector width in elements.
// The destination may be a proposed shared type during layout planning.
FailureOr<unsigned> getAsyncCopyMaxVectorSize(
    gpu::AsyncCopyGlobalToLocalOp copy, gpu::MemDescType destination,
    ModuleAxisInfoAnalysis &axisInfo);

// ldg.bsm uses a wave-uniform shared base and implicit lane offsets.
bool isAsyncCopyLayoutCompatible(RankedTensorType source,
                                gpu::MemDescType destination, unsigned vec);

// Keep a compatible source layout; otherwise fit it to the fixed destination.
// Returns null when the supported blocked/shared combination cannot be formed.
// minVec/maxVec bound the copy width in elements; minVec=1 allows scalar copies.
gpu::BlockedEncodingAttr chooseAsyncCopyLayout(
    RankedTensorType source, gpu::MemDescType destination,
    const AxisInfo *sourceInfo, unsigned minVec = 1, unsigned maxVec = ~0u);

} // namespace mlir::triton::metax

#endif
