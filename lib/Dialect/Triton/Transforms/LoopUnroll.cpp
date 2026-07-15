#include "mlir/Dialect/SCF/Utils/Utils.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/Triton/Transforms/Passes.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/Support/Debug.h"

namespace mlir::triton {

#define GEN_PASS_DEF_TRITONLOOPUNROLL
#include "triton/Dialect/Triton/Transforms/Passes.h.inc"

#define DEBUG_TYPE "triton-loop-unroll"
#define DBGS() (llvm::dbgs() << "[" DEBUG_TYPE "]: ")
#define LDBG(X) LLVM_DEBUG(DBGS() << X << "\n")

class LoopUnrollPass : public impl::TritonLoopUnrollBase<LoopUnrollPass> {

  int getUnrollFactorOrDefault(scf::ForOp forOp) {
    // Use the attribute attached to the loop if it exists otherwise set the
    // factor to 1 to suppress the unrolling.
    if (auto factor =
            forOp->getAttrOfType<IntegerAttr>(loopUnrollFactorAttrName))
      return factor.getInt();
    return 1;
  }

#ifdef __FLAGTREE_REORDER_LOOP_LOADS__
  // Returns true if unrolling by the given factor will fully eliminate the
  // loop.
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
    // The main loop after partial unroll has tripCount / unrollFactor
    // iterations. If that is <= 1, MLIR will eliminate it by inlining the
    // body, so from our perspective the main loop is fully unrolled.
    return tripCount > 0 && tripCount / unrollFactor <= 1;
  }

  bool hasReorderAttr(scf::ForOp forOp) {
    if (auto reorderAttr = forOp->getAttrOfType<BoolAttr>(reorderAttrName))
      return reorderAttr.getValue();
    return false;
  }

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
#endif // __FLAGTREE_REORDER_LOOP_LOADS__

  const char *loopUnrollFactorAttrName = "tt.loop_unroll_factor";
  const char *pipelineStagesAttrName = "tt.num_stages";
#ifdef __FLAGTREE_REORDER_LOOP_LOADS__
  const char *reorderAttrName = "tt.reorder";
#endif // __FLAGTREE_REORDER_LOOP_LOADS__

public:
  void runOnOperation() override {
    LDBG("Loop unroll pass");
    SmallVector<scf::ForOp, 4> loops;
    getOperation()->walk([&](scf::ForOp forOp) {
      // Bail out for loops with unroll factor <= 1.
      if (getUnrollFactorOrDefault(forOp) > 1)
        loops.push_back(forOp);
    });

    auto ctx = getOperation()->getContext();
    for (auto loop : loops) {
      auto unrollFactor = getUnrollFactorOrDefault(loop);
#ifdef __FLAGTREE_REORDER_LOOP_LOADS__
      bool needsReorder = hasReorderAttr(loop);
      bool fullyUnrolls = willFullyUnroll(loop, unrollFactor);
#endif // __FLAGTREE_REORDER_LOOP_LOADS__

      loop->removeAttr(loopUnrollFactorAttrName);
#ifdef __FLAGTREE_REORDER_LOOP_LOADS__
      if (needsReorder)
        loop->removeAttr(reorderAttrName);

      // Record position before unrolling so we can identify unrolled ops
      // if the loop gets fully eliminated.
      Block *parentBlock = loop->getBlock();
      Operation *opBeforeLoop = loop->getPrevNode(); // may be nullptr
#endif // __FLAGTREE_REORDER_LOOP_LOADS__

      LDBG("Unrolling loop by " << unrollFactor << " times\n" << loop);
      auto resultLoops = loopUnrollByFactor(loop, unrollFactor);

      if (failed(resultLoops))
        continue;
#ifdef __FLAGTREE_REORDER_LOOP_LOADS__
      if (needsReorder) {
        if (fullyUnrolls) {
          // Full unroll: the loop has been completely eliminated.
          // The unrolled ops are now in parentBlock between opBeforeLoop and
          // the block terminator.
          Operation *rangeBegin = opBeforeLoop ? opBeforeLoop->getNextNode()
                                               : &parentBlock->front();
          Operation *rangeEnd = parentBlock->getTerminator();
          reorderLoadsInRange(parentBlock, rangeBegin, rangeEnd);
        } else {
          // Partial unroll: the loop still exists, body has been replicated.
          // Reorder loads within the loop body.
          // After partial unroll the original loop is updated in-place, so
          // `loop` is still valid but its body has been expanded.
          // However loopUnrollByFactor may return a new main loop op when
          // there's an epilogue. Use the main loop if available.
          scf::ForOp mainLoop =
              resultLoops->mainLoopOp ? *resultLoops->mainLoopOp : loop;
          if (mainLoop) {
            Block *body = mainLoop.getBody();
            reorderLoadsInRange(body, &body->front(), body->getTerminator());
          }
        }
      }
#endif // __FLAGTREE_REORDER_LOOP_LOADS__

      // Do not pipeline the epilog loop.
      if (succeeded(resultLoops) && resultLoops->epilogueLoopOp) {
        (*resultLoops->epilogueLoopOp)
            ->setAttr(pipelineStagesAttrName,
                      mlir::IntegerAttr::get(IntegerType::get(ctx, 32), 1));
      }
    }
  }
};

} // namespace mlir::triton
