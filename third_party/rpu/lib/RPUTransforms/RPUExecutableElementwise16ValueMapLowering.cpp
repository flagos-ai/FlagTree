#include "RPUExecutableElementwise16ValueMapLowering.h"

namespace mlir {
namespace rpu {

FailureOr<bool> buildElementwise16ValueMapLoweringPlan(
    const Elementwise16ValueMapLoweringRequest &operands,
    Elementwise16ValueMapLoweringPlan &plan) {
  plan = Elementwise16ValueMapLoweringPlan();
  if (operands.outputArgIndex != 0 || operands.n != 16 ||
      operands.logicalN != 16 || operands.masked ||
      operands.inputArgIndices.size() < 2 ||
      operands.inputArgIndices.size() > 4 || operands.ops.size() < 1 ||
      operands.ops.size() > 3)
    return false;

  if (operands.inputArgIndices.size() != operands.ops.size() + 1)
    return false;
  for (size_t i = 0, e = operands.inputArgIndices.size(); i < e; ++i) {
    if (operands.inputArgIndices[i] != static_cast<unsigned>(i + 1))
      return false;
  }

  for (size_t opIndex = 1; opIndex < operands.ops.size(); ++opIndex) {
    const int64_t previousResultSlot =
        static_cast<int64_t>(operands.inputArgIndices.size() + opIndex - 1);
    const exec::ExecutableCompactVectorBinaryBuildOp &op =
        operands.ops[opIndex];
    if (op.lhs != previousResultSlot && op.rhs != previousResultSlot)
      return false;
  }

  int64_t availableSlots =
      static_cast<int64_t>(operands.inputArgIndices.size());
  for (const exec::ExecutableCompactVectorBinaryBuildOp &op : operands.ops) {
    if (op.lhs < 0 || op.rhs < 0 || op.lhs >= availableSlots ||
        op.rhs >= availableSlots)
      return false;
    ++availableSlots;
  }

  plan.outputArgIndex = operands.outputArgIndex;
  plan.inputArgIndices = operands.inputArgIndices;

  plan.ops.reserve(operands.ops.size());
  plan.ops = operands.ops;

  return true;
}

LogicalResult materializeElementwise16ValueMapLoweringPlan(
    OpBuilder &builder, Location loc, exec::KernelOp kernel,
    const Elementwise16ValueMapLoweringPlan &plan, llvm::StringRef consumer) {
  exec::ExecutableElementwise16ValueMapBuildSpec spec;
  spec.outputArgIndex = plan.outputArgIndex;
  spec.inputArgIndices = plan.inputArgIndices;
  spec.ops = plan.ops;
  return exec::buildExecutableElementwise16ValueMapBody(builder, loc, kernel,
                                                        spec, consumer);
}

} // namespace rpu
} // namespace mlir
