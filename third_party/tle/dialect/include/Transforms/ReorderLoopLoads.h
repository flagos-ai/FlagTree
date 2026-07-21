#ifndef TLE_DIALECT_TRANSFORMS_REORDERLOOPLOADS_H
#define TLE_DIALECT_TRANSFORMS_REORDERLOOPLOADS_H

#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/SCF/Utils/Utils.h"
#include "mlir/IR/Block.h"
#include "mlir/IR/Operation.h"

namespace mlir::triton::tle {

// Name of the frontend attribute marking a reorder-enabled loop.
bool hasReorderLoopLoadsAttr(scf::ForOp forOp);

// Removes the reorder attribute from the loop (called before unrolling so the
// attribute does not leak onto the unrolled ops).
void removeReorderLoopLoadsAttr(scf::ForOp forOp);

// Returns true if unrolling `forOp` by `unrollFactor` will fully eliminate the
// loop (i.e. the remaining main loop has <= 1 iteration and MLIR inlines it).
bool willFullyUnroll(scf::ForOp forOp, int64_t unrollFactor);

// Reorders loads after `loopUnrollByFactor` has run.
//
//   parentBlock  - the block that contained the loop before unrolling.
//   opBeforeLoop - the op immediately preceding the loop before unrolling
//                  (may be null if the loop was first in its block).
//   info         - result of loopUnrollByFactor.
//   fullyUnrolls - value computed by willFullyUnroll() *before* unrolling.
//
// For a fully-unrolled loop the inlined ops now live in `parentBlock`; for a
// partially-unrolled loop the (still-present) main loop body is reordered.
void reorderLoopLoadsAfterUnroll(const UnrolledLoopInfo &info,
                                 Block *parentBlock, Operation *opBeforeLoop,
                                 scf::ForOp originalLoop, bool fullyUnrolls);

} // namespace mlir::triton::tle

#endif // TLE_DIALECT_TRANSFORMS_REORDERLOOPLOADS_H
