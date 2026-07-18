// Copyright 2026 FlagOS Contributors
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.

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
