#ifndef TRITON_CONVERSION_TRITONMUSAGPU_TO_LLVM_ALLOCATION_H
#define TRITON_CONVERSION_TRITONMUSAGPU_TO_LLVM_ALLOCATION_H

#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Operation.h"

#include <functional>

namespace mlir {
namespace triton {
class TargetInfoBase;

namespace musa_gpu {
bool needsMusaRepDisjointGenericScratch(RankedTensorType srcTy,
                                        RankedTensorType dstTy,
                                        const TargetInfoBase &targetInfo);

std::function<unsigned(Operation *)>
getMusaAllocationAnalysisScratchSizeFn(const TargetInfoBase &targetInfo);

} // namespace musa_gpu
} // namespace triton
} // namespace mlir

#endif // TRITON_CONVERSION_TRITONMUSAGPU_TO_LLVM_ALLOCATION_H
