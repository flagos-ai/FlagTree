#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/SCF/Utils/Utils.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/Triton/Transforms/Passes.h"
#include "llvm/Support/Debug.h"

namespace mlir::triton {

#define GEN_PASS_DEF_TRITONLOOPUNROLL
#include "triton/Dialect/Triton/Transforms/Passes.h.inc"

#define DEBUG_TYPE "triton-loop-unroll"
#define DBGS() (llvm::dbgs() << "[" DEBUG_TYPE "]: ")
#define LDBG(X) LLVM_DEBUG(DBGS() << X << "\n")

// The standard unroller constructs an epilogue with fewer than factor
// iterations. Peel that bounded remainder using its original scalar condition.
// Nested guards preserve the exact iter-arg values on the taken path, including
// async tokens and memdescs, instead of merging them between peeled stages.
static SmallVector<Value> peelGuardedRemainder(IRRewriter &rewriter,
                                              scf::ForOp loop, unsigned count,
                                              Value iv, ValueRange args) {
  auto loc = loop.getLoc();
  auto active = rewriter.create<arith::CmpIOp>(
      loc, arith::CmpIPredicate::slt, iv, loop.getUpperBound());
  auto guarded = rewriter.create<scf::IfOp>(loc, loop.getResultTypes(), active,
                                           /*withElseRegion=*/true);
  {
    OpBuilder::InsertionGuard guard(rewriter);
    rewriter.setInsertionPointToStart(&guarded.getThenRegion().front());
    IRMapping mapping;
    mapping.map(loop.getInductionVar(), iv);
    mapping.map(loop.getRegionIterArgs(), args);
    for (Operation &op : loop.getBody()->without_terminator())
      rewriter.clone(op, mapping);
    SmallVector<Value> yielded;
    for (Value value : loop.getBody()->getTerminator()->getOperands())
      yielded.push_back(mapping.lookupOrDefault(value));
    if (count > 1) {
      auto next = rewriter.create<arith::AddIOp>(loc, iv, loop.getStep());
      yielded = peelGuardedRemainder(rewriter, loop, count - 1, next, yielded);
    }
    rewriter.create<scf::YieldOp>(loc, yielded);
  }
  {
    OpBuilder::InsertionGuard guard(rewriter);
    rewriter.setInsertionPointToStart(&guarded.getElseRegion().front());
    rewriter.create<scf::YieldOp>(loc, args);
  }
  return SmallVector<Value>(guarded.getResults().begin(),
                            guarded.getResults().end());
}

static void peelGeneratedRemainder(scf::ForOp loop, unsigned factor) {
  // Opt-in experiment: keep code growth bounded and use the signed i32 loop
  // form exercised by HCU's late unroller. Other forms retain the original loop.
  if (factor < 2 || factor > 8 || loop.getNumResults() == 0 ||
      !loop.getInductionVar().getType().isInteger(32) ||
      loop->hasAttr("unsignedCmp"))
    return;
  llvm::APInt step;
  if (!matchPattern(loop.getStep(), m_ConstantInt(&step)) || step.isZero() ||
      step.isNegative())
    return;
  IRRewriter rewriter(loop.getContext());
  rewriter.setInsertionPoint(loop);
  // Check the bound in widened arithmetic and keep the original loop as a
  // fallback. This does not assume that an arbitrary i32 bound calculation or
  // the final induction increment cannot overflow.
  auto loc = loop.getLoc();
  auto i64 = rewriter.getI64Type();
  auto lower = rewriter.create<arith::ExtSIOp>(loc, i64, loop.getLowerBound());
  auto upper = rewriter.create<arith::ExtSIOp>(loc, i64, loop.getUpperBound());
  auto span = rewriter.create<arith::SubIOp>(loc, upper, lower);
  auto step64 = rewriter.create<arith::ConstantIntOp>(loc, step.getSExtValue(), 64);
  auto maxSpan = rewriter.create<arith::ConstantIntOp>(
      loc, step.getSExtValue() * (factor - 1), 64);
  auto bounded = rewriter.create<arith::CmpIOp>(
      loc, arith::CmpIPredicate::sle, span, maxSpan);
  auto incrementLimit = rewriter.create<arith::AddIOp>(loc, upper, step64);
  auto maxI32 = rewriter.create<arith::ConstantIntOp>(loc, 2147483647, 64);
  auto noOverflow = rewriter.create<arith::CmpIOp>(
      loc, arith::CmpIPredicate::sle, incrementLimit, maxI32);
  auto usePeeling = rewriter.create<arith::AndIOp>(loc, bounded, noOverflow);
  auto choice = rewriter.create<scf::IfOp>(loc, loop.getResultTypes(), usePeeling,
                                         /*withElseRegion=*/true);
  {
    OpBuilder::InsertionGuard guard(rewriter);
    rewriter.setInsertionPointToStart(&choice.getThenRegion().front());
    auto results = peelGuardedRemainder(rewriter, loop, factor - 1,
                                        loop.getLowerBound(), loop.getInitArgs());
    rewriter.create<scf::YieldOp>(loc, results);
  }
  {
    OpBuilder::InsertionGuard guard(rewriter);
    rewriter.setInsertionPointToStart(&choice.getElseRegion().front());
    auto fallback = cast<scf::ForOp>(rewriter.clone(*loop.getOperation()));
    rewriter.create<scf::YieldOp>(loc, fallback.getResults());
  }
  rewriter.replaceOp(loop, choice.getResults());
}

class LoopUnrollPass : public impl::TritonLoopUnrollBase<LoopUnrollPass> {

  int getUnrollFactorOrDefault(scf::ForOp forOp) {
    // Use the attribute attached to the loop if it exists otherwise set the
    // factor to 1 to suppress the unrolling.
    if (auto factor =
            forOp->getAttrOfType<IntegerAttr>(loopUnrollFactorAttrName))
      return factor.getInt();
    return 1;
  }

  const char *loopUnrollFactorAttrName = "tt.loop_unroll_factor";
  const char *pipelineStagesAttrName = "tt.num_stages";

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
    auto requested = getOperation()->getAttrOfType<IntegerAttr>(
        "ttg.hcu.peel_unroll_remainder");
    auto target = getOperation()->getAttrOfType<StringAttr>("ttg.target");
    bool peelRemainder = requested && requested.getInt() == 1 && target &&
                         target.getValue() == "hip:gfx936";
    for (auto loop : loops) {
      auto unrollFactor = getUnrollFactorOrDefault(loop);
      loop->removeAttr(loopUnrollFactorAttrName);
      LDBG("Unrolling loop by " << unrollFactor << " times\n" << loop);
      // The unroll utility computes ceilDivPositive(upper - lower, step).
      // Pipelining can produce an empty loop with upper < lower. Preserve its
      // zero iterations before the utility builds the remainder loop.
      OpBuilder builder(loop);
      Value upper = loop.getUnsignedCmp()
                        ? builder.createOrFold<arith::MaxUIOp>(
                              loop.getLoc(), loop.getUpperBound(), loop.getLowerBound())
                        : builder.createOrFold<arith::MaxSIOp>(
                              loop.getLoc(), loop.getUpperBound(), loop.getLowerBound());
      loop.getUpperBoundMutable().assign(upper);
      auto resultLoops = loopUnrollByFactor(loop, unrollFactor);
      // Do not pipeline the epilog loop.
      if (succeeded(resultLoops) && resultLoops->epilogueLoopOp) {
        (*resultLoops->epilogueLoopOp)
            ->setAttr(pipelineStagesAttrName,
                      mlir::IntegerAttr::get(IntegerType::get(ctx, 32), 1));
        if (peelRemainder)
          peelGeneratedRemainder(*resultLoops->epilogueLoopOp, unrollFactor);
      }
    }
  }
};

} // namespace mlir::triton
