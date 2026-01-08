#include "triton/Dialect/Triton/IR/Traits.h"

#include <numeric>

#include "mlir/IR/TypeUtilities.h"
#include "triton/Dialect/Triton/IR/Types.h"
#include "triton/Dialect/Triton/IR/Utility.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "llvm/Support/ErrorHandling.h"

using namespace mlir;
namespace ttg = mlir::triton::gpu;

LogicalResult OpTrait::impl::verifyTensorSize(Operation *op) {
  for (auto opType : op->getOperandTypes()) {
    if (auto tensorType = dyn_cast<RankedTensorType>(opType)) {
      int64_t numElements = 1;
      for (int64_t s : tensorType.getShape())
        numElements *= s;
      if (numElements > maxTensorNumElements)
        return op->emitError("Maximum allowed number of elements is ")
               << maxTensorNumElements << ", but " << *op
               << " has more than that";
      // if ((numElements & (numElements - 1)) != 0)
      //   return op->emitError("Number of elements must be power-of-two, but ")
      //          << *op << " doesn't follow the rule (" << numElements << ")"
      //          << " elements";
    }
  }
  for (auto opType : op->getResultTypes()) {
    if (auto tensorType = dyn_cast<RankedTensorType>(opType)) {
      int64_t numElements = 1;
      for (int64_t s : tensorType.getShape())
        numElements *= s;
      if (numElements > maxTensorNumElements)
        return op->emitError("Maximum allowed number of elements is ")
               << maxTensorNumElements << ", but " << *op
               << " has more than that";
      // if ((numElements & (numElements - 1)) != 0)
      //   return op->emitError("Number of elements must be power-of-two, but ")
      //          << *op << " doesn't follow the rule (" << numElements << ")"
      //          << " elements";
    }
  }
  return success();
}
