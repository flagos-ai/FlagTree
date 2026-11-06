#ifndef TLE_DIALECT_TRANSFORMS_REORDERLOOPLOADS_H
#define TLE_DIALECT_TRANSFORMS_REORDERLOOPLOADS_H

#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/Block.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/Operation.h"

#include <optional>

namespace mlir::triton::tle {

// Prefetch distance requested by the `tt.reorder` hint on `forOp`:
// std::nullopt without a hint, -1 for reorder=True (every unrolled iteration's
// loads go first), k >= 1 for reorder=k.
std::optional<int64_t> getReorderLoopLoadsDistance(scf::ForOp forOp);

void removeReorderLoopLoadsAttr(scf::ForOp forOp);

// `annotateFn` for loopUnrollByFactor: tags `op` with the unrolled iteration
// `iteration` it was cloned for.
void annotateUnrolledIteration(unsigned iteration, Operation *op, OpBuilder b);

// Reorders the loads of a loop that loopUnrollByFactor unrolled with
// annotateUnrolledIteration. The ops that came out of the loop sit between
// `opBeforeLoop` (nullptr when the loop was the first op of `parentBlock`) and
// `opAfterLoop` in `parentBlock`: the inlined body of a fully unrolled loop, or
// the main (and epilogue) loop of a partially unrolled one. Removes the
// iteration tags in either case.
void reorderLoopLoadsAfterUnroll(Block *parentBlock, Operation *opBeforeLoop,
                                 Operation *opAfterLoop, int64_t distance);

} // namespace mlir::triton::tle

#endif // TLE_DIALECT_TRANSFORMS_REORDERLOOPLOADS_H
