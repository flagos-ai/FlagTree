#ifndef TLE_DIALECT_TRANSFORMS_REORDERLOOPLOADS_H
#define TLE_DIALECT_TRANSFORMS_REORDERLOOPLOADS_H

#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/SCF/Utils/Utils.h"
#include "mlir/IR/Block.h"
#include "mlir/IR/Operation.h"

namespace mlir::triton::tle {

bool hasReorderLoopLoadsAttr(scf::ForOp forOp);

void removeReorderLoopLoadsAttr(scf::ForOp forOp);

bool willFullyUnroll(scf::ForOp forOp, int64_t unrollFactor);

void reorderLoopLoadsAfterUnroll(const UnrolledLoopInfo &info,
                                 Block *parentBlock, Operation *opBeforeLoop,
                                 scf::ForOp originalLoop, bool fullyUnrolls);

} // namespace mlir::triton::tle

#endif // TLE_DIALECT_TRANSFORMS_REORDERLOOPLOADS_H
