#ifndef TRITON_INTELLIFUSION_GPNPU_ALLOCATION_H
#define TRITON_INTELLIFUSION_GPNPU_ALLOCATION_H

#include "mlir/IR/Operation.h"

#include <functional>

namespace mlir::triton {
class TargetInfoBase;

namespace Intellifusion {
std::function<unsigned(Operation *)>
getGPNPUAllocationAnalysisScratchSizeFn(TargetInfoBase &targetInfo);
} // namespace Intellifusion
} // namespace mlir::triton

#endif // TRITON_INTELLIFUSION_GPNPU_ALLOCATION_H
