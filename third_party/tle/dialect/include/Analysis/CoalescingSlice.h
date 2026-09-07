// Copyright 2026 FlagOS Contributors

#ifndef TRITON_TLE_ANALYSIS_COALESCING_SLICE_H
#define TRITON_TLE_ANALYSIS_COALESCING_SLICE_H

#include "mlir/IR/Operation.h"
#include "mlir/IR/OpDefinition.h"
#include "llvm/ADT/SetVector.h"

namespace mlir::triton::tle {

/// The membership of mlir::getSlice(root) with default slice options, without
/// topological ordering. Coalescing reduces the set with max(), so order is not
/// observable. Expand each direct edge once instead of recomputing a transitive
/// forward and backward slice for every member of the same fixed point.
///
/// The graph is directed: forward traversal includes nested operations, but
/// backward traversal reaches a parent only through a block argument and stops
/// at IsolatedFromAbove. Do not replace it with a parent walk or an undirected
/// connected-component search, either of which would join unrelated accesses.
inline llvm::SetVector<Operation *> getCoalescingSlice(Operation *root) {
  llvm::SetVector<Operation *> slice;
  slice.insert(root);
  for (size_t index = 0; index != slice.size(); ++index) {
    Operation *op = slice[index];
    for (Region &region : op->getRegions())
      for (Block &block : region)
        for (Operation &child : block)
          slice.insert(&child);
    for (Value result : op->getResults())
      for (Operation *user : result.getUsers())
        slice.insert(user);

    if (op->hasTrait<OpTrait::IsIsolatedFromAbove>())
      continue;
    for (Value operand : op->getOperands()) {
      Operation *predecessor = operand.getDefiningOp();
      if (!predecessor) {
        auto argument = cast<BlockArgument>(operand);
        predecessor = argument.getOwner()->getParentOp();
        assert((!predecessor ||
                (predecessor->getNumRegions() == 1 &&
                 predecessor->getRegion(0).hasOneBlock())) &&
               "default backward slice requires a single-region single-block "
               "block-argument owner");
      }
      if (predecessor &&
          !predecessor->hasTrait<OpTrait::IsIsolatedFromAbove>())
        slice.insert(predecessor);
    }
  }
  return slice;
}

} // namespace mlir::triton::tle

#endif
