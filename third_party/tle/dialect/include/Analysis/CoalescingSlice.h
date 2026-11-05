// Copyright 2026 FlagOS Contributors

#ifndef TRITON_TLE_ANALYSIS_COALESCING_SLICE_H
#define TRITON_TLE_ANALYSIS_COALESCING_SLICE_H

#include "mlir/Analysis/DataFlow/DeadCodeAnalysis.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/Operation.h"
#include "mlir/Interfaces/FunctionInterfaces.h"
#include "triton/Analysis/Utility.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/SetVector.h"

namespace mlir::triton::tle {

/// Reuse the same executable-block proof as AxisInfo's sparse dataflow solver.
/// Unreachable blocks deliberately have bottom AxisInfo; they cannot
/// participate in memory coalescing or contribute same-order dependencies.
inline LogicalResult
collectNonExecutableCoalescingBlocks(FunctionOpInterface function,
                                     llvm::DenseSet<Block *> &blocks) {
  auto solver = mlir::createDataFlowSolver();
  if (failed(solver->initializeAndRun(function)))
    return failure();
  function.walk([&](Block *block) {
    auto *state = solver->lookupState<dataflow::Executable>(
        solver->getProgramPointBefore(block));
    if (state && !state->isLive())
      blocks.insert(block);
  });
  return success();
}

/// Coalescing's dependency closure, without topological ordering. This matches
/// mlir::getSlice(root) on its supported single-region block-argument owners,
/// and also handles carried arguments of multi-region operations such as
/// scf.while. Coalescing reduces the set with max(), so order is not
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
        // A carried argument connects to its owning operation regardless of
        // region count. Visiting that owner reaches its inputs, results and
        // every region, including both the condition and update of scf.while.
        // The isolation check below still stops at function boundaries.
      }
      if (predecessor && !predecessor->hasTrait<OpTrait::IsIsolatedFromAbove>())
        slice.insert(predecessor);
    }
  }
  return slice;
}

} // namespace mlir::triton::tle

#endif
