#ifndef TLE_UTILITY_H
#define TLE_UTILITY_H

#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/Value.h"
#include "llvm/ADT/STLExtras.h"

namespace mlir {
bool isSingleForLoop(scf::ForOp forOp);
bool isFromIterArg(Value operand, scf::ForOp forOp);
} // namespace mlir

#endif
