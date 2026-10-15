#ifndef ANALYSIS_TKO_DEPENDENCE_ANALYSIS_H
#define ANALYSIS_TKO_DEPENDENCE_ANALYSIS_H

#include "cuda_tile/Dialect/CudaTile/IR/Ops.h"
#include "mlir/IR/Operation.h"
#include "llvm/ADT/DenseMap.h"

#include <utility>

namespace mlir {
namespace triton {
namespace tileir {

/// Bounded loop-store disjointness proof for AutoGenMemoryToken.
///
/// This is not a general dependence analysis. It does not answer arbitrary
/// pairwise RAW/WAR/WAW/RAR questions, and it does not replace MLIR alias
/// analysis. Today it has one purpose: after TkoAliasAnalysis has grouped TKO
/// memory ops by provenance root, prove that one top-level store in a
/// cuda_tile.for touches disjoint footprints across distinct loop iterations.
///
/// AutoGenMemoryToken uses a true result for one narrow optimization: the store
/// can take a loop-invariant input token instead of the previous iteration's
/// lastStore token. The caller still owns all token-planning preconditions,
/// including rejecting a memory root that has other same-root reads, stores,
/// fences, or nested memory effects in the loop body.
///
/// Example:
///
///   cuda_tile.for %i in (%c0 to %c4, step %c1) {
///     %tile = cuda_tile.make_partition_view %base[%i]
///     cuda_tile.store_view_tko %tile, %value
///     cuda_tile.continue
///   }
///
/// If `%i` selects a different tile in every iteration,
/// isLoopCarriedStoreDisjoint(loop, storeOp) returns true for the store. If no
/// other effect touches the same root in the loop body, AutoGenMemoryToken may
/// avoid feeding the previous iteration's lastStore token into that store.
///
/// A false return is conservative: the access may collide, the store kind may
/// be unsupported, or the proof may be outside the small affine subset
/// implemented here.
class TkoDependenceAnalysis {
public:
  TkoDependenceAnalysis() = default;

  /// Return true only when `storeOp` is proven to touch disjoint footprints
  /// across distinct iterations of `loop`. The store must be a top-level op in
  /// the loop body. Return false conservatively for non-stores, unsupported
  /// store forms, or possible collisions.
  bool isLoopCarriedStoreDisjoint(cuda_tile::ForOp loop, Operation *storeOp);

private:
  using CacheKey = std::pair<Operation *, Operation *>;
  DenseMap<CacheKey, bool> loopCarriedStoreCache;
};

} // namespace tileir
} // namespace triton
} // namespace mlir

#endif // ANALYSIS_TKO_DEPENDENCE_ANALYSIS_H
