// Copyright 2026 FlagOS Contributors
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.

#ifndef TRITON_TRITONGPU_TRANSFORMS_PARTITIONBUILDER_H
#define TRITON_TRITONGPU_TRANSFORMS_PARTITIONBUILDER_H

#include "mlir/IR/ImplicitLocOpBuilder.h"
#include "llvm/ADT/SetVector.h"

namespace mlir::triton::gpu {

class Partition;

using StageCluster = std::optional<std::pair<int, int>>;

// Get the stage and cluster for an operation, if it has one assigned.
void setStageCluster(OpBuilder &b, Operation *op, StageCluster stageCluster);
StageCluster getStageCluster(Operation *op);

struct PartitionBuilder : public ImplicitLocOpBuilder {
  using ImplicitLocOpBuilder::ImplicitLocOpBuilder;

  Value intCst(int value, unsigned width = 32);
  Value boolCst(bool value);

  void assignPartition(Operation *op, Partition &partition);

  template <typename OpT, typename... Args>
  auto createInto(Partition &partition, StageCluster stageCluster,
                  Args &&...args) {
    auto op = create<OpT>(std::forward<Args>(args)...);
    assignPartition(op, partition);
    setStageCluster(*this, op, stageCluster);
    return op;
  }
};

template <typename OpT, typename... Args>
OpT createInto(OpBuilder &b, Location loc,
               std::optional<SetVector<int>> partitionSet,
               StageCluster stageCluster, Args &&...args) {
  auto op = OpT::create(b, loc, std::forward<Args>(args)...);
  if (partitionSet) {
    setPartition(op, *partitionSet);
    setStageCluster(b, op, stageCluster);
  }
  return op;
}

} // namespace mlir::triton::gpu

#endif // TRITON_TRITONGPU_TRANSFORMS_PARTITIONBUILDER_H
