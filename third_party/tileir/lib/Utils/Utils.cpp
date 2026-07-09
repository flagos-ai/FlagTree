#include "Utils/Utils.h"

#include "cuda_tile/Dialect/CudaTile/IR/Attributes.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Types.h"
#include "mlir/Support/LLVM.h"

#include "triton/Dialect/Triton/IR/Dialect.h"

namespace mlir {
namespace triton {
namespace utils {

std::optional<int> getNumStagesFromParentForOp(Operation *op) {
  auto parentOp = op->getParentOfType<scf::ForOp>();
  while (parentOp) {
    // Check for tt.num_stages attribute on ForOp
    if (auto numStagesAttr = parentOp->getAttr("tt.num_stages")) {
      if (auto intAttr = dyn_cast<IntegerAttr>(numStagesAttr)) {
        return intAttr.getInt();
      }
    }
    parentOp = parentOp->getParentOfType<scf::ForOp>();
  }
  return std::nullopt;
}

std::optional<cuda_tile::OptimizationHintsAttr>
convertNumStagesToOptHint(Operation *op, MLIRContext *ctx,
                          const DenseMap<Operation *, int> &numStagesMap,
                          int computeCapability, std::optional<int> numStages) {
  int numStagesValue = -1;
  if (numStagesMap.find(op) != numStagesMap.end()) {
    numStagesValue = numStagesMap.at(op);
  } else if (numStages.has_value()) {
    numStagesValue = numStages.value();
  }
  // The cost is valid between 1 and 10.
  // Will clip to 10 if numStages is greater than 10.
  // For 0 or negative values, we will use the default cost indicated by a null
  // OptHintAttr.
  int clippedNumStages = std::min(numStagesValue, 10);
  if (clippedNumStages > 0)
    return mlir::triton::utils::cvtNumStagesToOptHintAttr(
        ctx, computeCapability, clippedNumStages);

  return std::nullopt;
}

std::optional<cuda_tile::OptimizationHintsAttr>
cvtNumStagesToOptHintAttr(MLIRContext *ctx, int computeCapability,
                          int numStages) {

  std::string arch = ("sm_" + llvm::Twine(computeCapability)).str();
  if (numStages == -1)
    return std::nullopt;
  return cuda_tile::OptimizationHintsAttr::get(
      ctx,
      mlir::DictionaryAttr::get(
          ctx,
          mlir::NamedAttribute(
              arch, mlir::DictionaryAttr::get(
                        ctx, mlir::NamedAttribute(
                                 mlir::StringAttr::get(ctx, "latency"),
                                 mlir::IntegerAttr::get(
                                     IntegerType::get(ctx, 32), numStages))))));
}

} // namespace utils
} // namespace triton
} // namespace mlir
