#include "tle/dialect/include/Transforms/TleUtility.h"
#include "llvm/ADT/STLExtras.h"

namespace mlir {
bool isSingleForLoop(scf::ForOp forOp) {
  auto parent = forOp->getParentOp();
  return !parent || !isa<scf::ForOp>(parent);
}

bool isFromIterArg(Value operand, scf::ForOp forOp) {
  auto blockArg = dyn_cast<BlockArgument>(operand);
  return llvm::is_contained(forOp.getRegionIterArgs(), blockArg);
}
} // namespace mlir
