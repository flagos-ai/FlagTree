// Copyright 2026 FlagOS Contributors
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.

#include "triton/Dialect/Triton/Transforms/LoopPeeling.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Pass/Pass.h"
#include "triton/Dialect/Triton/IR/Utility.h"

using namespace mlir;

namespace mlir {
namespace triton {

void peelLoopEpilogue(
    scf::ForOp forOp,
    function_ref<Operation *(RewriterBase &, Operation *, bool)>
        processPeeledOp) {
  SmallVector<Operation *> loopBodyOps;
  IRRewriter rewriter(forOp);
  Location loc = forOp.getLoc();
  Type type = forOp.getStep().getType();

  // Fetch loop bounds and step
  Value lowerBound = forOp.getLowerBound();
  Value upperBound = forOp.getUpperBound();
  Value step = forOp.getStep();
  Value newUpperBound = arith::SubIOp::create(rewriter, loc, upperBound, step);

  rewriter.setInsertionPointAfter(forOp);
  Value lastIV = getLastInductionValue(rewriter, forOp);

  auto cond = arith::CmpIOp::create(rewriter, loc, arith::CmpIPredicate::slt,
                                    lowerBound, upperBound);

  // Create an if op to execute the peeled iteration
  IRMapping map;
  map.map(forOp.getRegionIterArgs(), forOp.getResults());
  map.map(forOp.getInductionVar(), lastIV);
  auto ifOp = scf::IfOp::create(rewriter, loc, forOp.getResultTypes(), cond);
  forOp.getBodyRegion().cloneInto(&ifOp.getThenRegion(), map);
  auto newElseBlock = rewriter.createBlock(&ifOp.getElseRegion());
  rewriter.setInsertionPointToStart(newElseBlock);
  scf::YieldOp::create(rewriter, loc, forOp.getResults());

  forOp->replaceUsesWithIf(ifOp, [&](OpOperand &operand) {
    return !ifOp->isAncestor(operand.getOwner());
  });

  forOp.getUpperBoundMutable().assign(newUpperBound);

  if (processPeeledOp) {
    for (auto &op :
         llvm::make_early_inc_range(forOp.getBody()->without_terminator())) {
      Operation *newOp = processPeeledOp(rewriter, &op, /*isEpilogue=*/false);
      if (newOp && newOp != &op) {
        op.replaceAllUsesWith(newOp);
      }
    }
    for (auto &op : llvm::make_early_inc_range(
             ifOp.getThenRegion().front().without_terminator())) {
      Operation *newOp = processPeeledOp(rewriter, &op, /*isEpilogue=*/true);
      if (newOp && newOp != &op) {
        op.replaceAllUsesWith(newOp);
      }
    }
  }
}

} // namespace triton
} // namespace mlir
