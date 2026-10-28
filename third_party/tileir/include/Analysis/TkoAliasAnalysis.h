#ifndef ANALYSIS_TKO_ALIAS_ANALYSIS_H
#define ANALYSIS_TKO_ALIAS_ANALYSIS_H

#include "mlir/IR/Attributes.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/Value.h"
#include "llvm/ADT/BitVector.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/SmallVector.h"

namespace mlir {
namespace triton {
namespace tileir {

/// Function-local memory-root analysis for token-ordered (TKO) memory
/// operations.
///
/// The public query is operation based: getAccessRoots(memOp) returns stable
/// ids for the allocation/provenance roots that a recognized TKO memory op may
/// touch. AutoGenMemoryToken uses those root ids as the keys for per-root token
/// state. Clients should not extract pointer/view operands and ask arbitrary
/// pairwise value-alias questions; operand decoding is owned by TkoOpSemantics
/// and this analysis.
///
/// A memory root is coarser than an access footprint. Two memory ops with
/// disjoint root sets cannot touch the same allocation. Two memory ops that
/// share a root may still be independent if TkoDependenceAnalysis can prove
/// their concrete footprints disjoint, including loop-carried disjointness.
///
/// Example:
///   %view = make_tensor_view(%arg0)
///   %tile = make_partition_view(%view)
///   store_view_tko(..., %tile)
///
/// getAccessRoots(store_view_tko) returns the root id for `%arg0`. Another
/// memory op derived from `%arg0` returns the same root id; a memory op derived
/// from a distinct function argument returns a different root id.
///
/// What this analysis does NOT do (future work):
///   - Index / layout disjointness. Two disjoint tile-regions of one base still
///     share a memory root here and may be separated later by dependence
///     analysis.
///   - Injective IV pattern-match for parallel-store (separate commit).
///   - Cross-function / interprocedural alias.
class TkoAliasAnalysis {
public:
  using MemoryRootId = unsigned;

  /// Construct and eagerly analyze `funcLike`. The op is expected to be
  /// an EntryOp or FuncOp. Analysis is function-local.
  ///
  /// Walks every TKO memory op in the body; their access values' reachable
  /// terminal roots populate the root table.
  explicit TkoAliasAnalysis(Operation *funcLike);

  /// Returns the memory roots touched by a recognized TKO memory op. The
  /// caller does not need to know which operand carries the access value; that
  /// is part of the op-semantics layer consumed here. Returns an empty list for
  /// non-memory ops.
  SmallVector<MemoryRootId, 2> getAccessRoots(Operation *memOp);

  /// Total number of distinct terminal memory roots in this function.
  unsigned getNumRoots() const { return numRoots; }

  /// Debug dump: list each memory root id with its representative SSA root.
  void print(raw_ostream &os) const;

private:
  using MemoryRootSet = llvm::BitVector;

  /// Walk `v` to its terminal roots, memoizing in `rootsCache`.
  /// `visiting` breaks cycles (e.g. scf.for iter_arg referring back);
  /// if re-entered the recursion returns an empty slice and relies on
  /// the other join-branch to contribute roots.
  ArrayRef<Value> computeRoots(Value v, DenseSet<Value> &visiting);

  /// Return the memory-root bitset for an arbitrary SSA value. This is an
  /// implementation detail of the root walk; pass clients should ask for
  /// access roots of memory ops instead of extracting access values manually.
  MemoryRootSet getValueRoots(Value v);

  /// Assign root IDs to terminal roots in function traversal order.
  void finalize();

  Operation *funcLike;

  /// Per-value cache of terminal roots (usually size 1; >1 on join).
  DenseMap<Value, SmallVector<Value, 2>> rootsCache;

  /// Canonical terminal root for each cuda_tile.get_global symbol.
  DenseMap<Attribute, Value> globalRootByName;

  /// After finalize(): terminal root → memory root id.
  DenseMap<Value, MemoryRootId> rootIds;

  /// After finalize(): cached MemoryRootSet per value.
  DenseMap<Value, MemoryRootSet> rootSetCache;

  unsigned numRoots = 0;
  bool finalized = false;
};

} // namespace tileir
} // namespace triton
} // namespace mlir

#endif // ANALYSIS_TKO_ALIAS_ANALYSIS_H
