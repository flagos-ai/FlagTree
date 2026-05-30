#pragma once

#include "RPU/IR/Dialect.h"
#include "mlir/IR/Builders.h"
#include "mlir/Support/LogicalResult.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include <cstdint>

namespace mlir {
namespace rpu {

struct Elementwise16ValueMapLoweringRequest {
  int64_t n = 0;
  int64_t logicalN = 0;
  bool masked = false;
  unsigned outputArgIndex = 0;
  SmallVector<unsigned, 4> inputArgIndices;
  SmallVector<exec::ExecutableCompactVectorBinaryBuildOp, 3> ops;
};

struct Elementwise16ValueMapLoweringPlan {
  unsigned outputArgIndex = 0;
  SmallVector<unsigned, 4> inputArgIndices;
  SmallVector<exec::ExecutableCompactVectorBinaryBuildOp, 3> ops;
};

FailureOr<bool> buildElementwise16ValueMapLoweringPlan(
    const Elementwise16ValueMapLoweringRequest &request,
    Elementwise16ValueMapLoweringPlan &plan);

LogicalResult materializeElementwise16ValueMapLoweringPlan(
    OpBuilder &builder, Location loc, exec::KernelOp kernel,
    const Elementwise16ValueMapLoweringPlan &plan, llvm::StringRef consumer);

} // namespace rpu
} // namespace mlir
