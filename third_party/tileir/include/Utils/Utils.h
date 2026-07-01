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
