#include "tle/dialect/include/Transforms/ReorderLoopLoads.h"

#include "tle/dialect/include/Transforms/TransformAttrs.h"

#include "mlir/IR/Matchers.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Debug.h"

#define DEBUG_TYPE "tle-reorder-loop-loads"
#define DBGS() (llvm::dbgs() << "[" DEBUG_TYPE "]: ")
#define LDBG(X) LLVM_DEBUG(DBGS() << X << "\n")

namespace mlir::triton::tle {

bool hasReorderLoopLoadsAttr(scf::ForOp forOp) {
  if (auto reorderAttr =
          forOp->getAttrOfType<BoolAttr>(kTleReorderLoopLoadsAttr))
    return reorderAttr.getValue();
  return false;
}

void removeReorderLoopLoadsAttr(scf::ForOp forOp) {
  forOp->removeAttr(kTleReorderLoopLoadsAttr);
}

bool willFullyUnroll(scf::ForOp forOp, int64_t unrollFactor) {
  IntegerAttr lbAttr, ubAttr, stepAttr;
  if (!matchPattern(forOp.getLowerBound(), m_Constant(&lbAttr)) ||
      !matchPattern(forOp.getUpperBound(), m_Constant(&ubAttr)) ||
      !matchPattern(forOp.getStep(), m_Constant(&stepAttr)))
    return false;
  int64_t lb = lbAttr.getInt();
  int64_t ub = ubAttr.getInt();
  int64_t step = stepAttr.getInt();
  if (step <= 0 || ub <= lb)
    return false;
  int64_t tripCount = llvm::divideCeil(ub - lb, step);
  return tripCount > 0 && tripCount / unrollFactor <= 1;
}

namespace {

// Collect all transitive in-block dependencies of `op`.
void collectDepsInBlock(Operation *op, Block *block,
                        llvm::SetVector<Operation *> &deps) {
  for (Value operand : op->getOperands()) {
    auto *defOp = operand.getDefiningOp();
    if (!defOp || defOp->getBlock() != block)
      continue;
    if (deps.insert(defOp)) {
      collectDepsInBlock(defOp, block, deps);
    }
  }
}

// Reorder loads in a range of operations within a block. Moves all load ops
// and their transitive dependencies before other ops in the range, preserving
// relative order within each group.
void reorderLoadsInRange(Block *block, Operation *beginOp, Operation *endOp) {
  // Collect all ops in the range [beginOp, endOp)
  SmallVector<Operation *, 32> rangeOps;
  for (auto it = Block::iterator(beginOp); &*it != endOp; ++it) {
    rangeOps.push_back(&*it);
  }

  if (rangeOps.empty())
    return;

  // Identify loads and their dependency cluster
  SmallVector<Operation *, 16> loads;
  llvm::SetVector<Operation *> loadCluster;

  for (Operation *op : rangeOps) {
    if (isa<triton::LoadOp>(op))
      loads.push_back(op);
  }

  if (loads.empty()) {
    LDBG("No loads found in unrolled range, skipping reorder");
    return;
  }

  for (Operation *load : loads) {
    loadCluster.insert(load);
    collectDepsInBlock(load, block, loadCluster);
  }

  LDBG("Full-unroll reorder: load cluster size: "
       << loadCluster.size() << " (loads: " << loads.size() << ")");

  // Build reordered sequence:
  // 1. Load cluster ops (deps + loads) in original relative order
  // 2. Remaining ops in original relative order
  SmallVector<Operation *, 32> reordered;
  for (Operation *op : rangeOps) {
    if (loadCluster.contains(op))
      reordered.push_back(op);
  }
  for (Operation *op : rangeOps) {
    if (!loadCluster.contains(op))
      reordered.push_back(op);
  }

  // Apply the new ordering by moving ops before endOp
  for (Operation *op : reordered) {
    op->moveBefore(endOp);
  }

  LDBG("Full-unroll reorder complete");
}

} // namespace

void reorderLoopLoadsAfterUnroll(const UnrolledLoopInfo &info,
                                 Block *parentBlock, Operation *opBeforeLoop,
                                 scf::ForOp originalLoop, bool fullyUnrolls) {
  if (fullyUnrolls) {
    // Full unroll: the loop has been completely eliminated.
    // The unrolled ops are now in parentBlock between opBeforeLoop and
    // the block terminator.
    Operation *rangeBegin =
        opBeforeLoop ? opBeforeLoop->getNextNode() : &parentBlock->front();
    Operation *rangeEnd = parentBlock->getTerminator();
    reorderLoadsInRange(parentBlock, rangeBegin, rangeEnd);
  } else {
    // Partial unroll: the loop still exists, body has been replicated.
    // Reorder loads within the loop body.
    // After partial unroll the original loop is updated in-place, so
    // `originalLoop` is still valid but its body has been expanded.
    // However loopUnrollByFactor may return a new main loop op when
    // there's an epilogue. Use the main loop if available.
    scf::ForOp mainLoop = info.mainLoopOp ? *info.mainLoopOp : originalLoop;
    if (mainLoop) {
      Block *body = mainLoop.getBody();
      reorderLoadsInRange(body, &body->front(), body->getTerminator());
    }
  }
}

} // namespace mlir::triton::tle
