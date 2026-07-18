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

#ifndef UTILS_UTILS_H
#define UTILS_UTILS_H

#include "mlir/IR/BuiltinOps.h"

#include "cuda_tile/Dialect/CudaTile/IR/Attributes.h"
#include <optional>

namespace mlir {
namespace triton {
namespace utils {

// Helper function to iterate through parent ForOp and find
// num_stages attribute
std::optional<int> getNumStagesFromParentForOp(Operation *op);

// Helper function to find the num_stages for the op and convert it to
// OptimizationHintsAttr.
std::optional<cuda_tile::OptimizationHintsAttr>
convertNumStagesToOptHint(Operation *op, MLIRContext *ctx,
                          const DenseMap<Operation *, int> &numStagesMap,
                          int computeCapability, std::optional<int> numStages);

// Helper function to convert a num_stages value to OptimizationHintsAttr.
std::optional<cuda_tile::OptimizationHintsAttr>
cvtNumStagesToOptHintAttr(MLIRContext *ctx, int computeCapability,
                          int numStages);

} // namespace utils
} // namespace triton
} // namespace mlir

#endif // UTILS_UTILS_H
