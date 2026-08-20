#ifndef ILUVATAR_TLE_UTILITY_H
#define ILUVATAR_TLE_UTILITY_H

#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/Value.h"
#include "llvm/ADT/STLExtras.h"

// Trunk declares these in namespace `mlir` (third_party/tle/dialect/include/
// Transforms/TleUtility.h). Iluvatar keeps them inside `iluvatar_tle` so the
// backend plugin never injects symbols into the shared `mlir` namespace.
namespace mlir::triton::iluvatar_tle {
bool isSingleForLoop(scf::ForOp forOp);
bool isFromIterArg(Value operand, scf::ForOp forOp);
} // namespace mlir::triton::iluvatar_tle

#endif
