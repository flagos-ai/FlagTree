#ifdef __ILUVATAR_TLE__

#include "Transforms/TleUtility.h"
#include "llvm/ADT/STLExtras.h"

namespace mlir::triton::iluvatar_tle {
bool isSingleForLoop(scf::ForOp forOp) {
  auto parent = forOp->getParentOp();
  return !parent || !isa<scf::ForOp>(parent);
}

bool isFromIterArg(Value operand, scf::ForOp forOp) {
  auto blockArg = dyn_cast<BlockArgument>(operand);
  return llvm::is_contained(forOp.getRegionIterArgs(), blockArg);
}
} // namespace mlir::triton::iluvatar_tle

#endif // __ILUVATAR_TLE__
