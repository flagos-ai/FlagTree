/**
 * Copyright 2024-2026 Enflame. All Rights Reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *  http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
#include <utility>
#include <vector>

#include "Conversion/TritonToGCU/TritonToGCUPass.h"

#include "Conversion/TritonToGCU/Utils.h"
#include "Dialect/TritonGCU/IR/TritonGCUDialect.h"
#include "Dialect/TritonGCU/IR/TritonGCUTypes.h"

#include "PipelineExpander.h"
#include "PipeliningUtility.h"
#include "Schedule.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/TypeUtilities.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "triton/Analysis/Utility.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/Transforms/Passes.h"
#include "triton/Dialect/TritonGPU/Transforms/Utility.h"
#include "triton/Dialect/TritonNvidiaGPU/IR/Dialect.h"
#include "llvm/Support/Debug.h"

//===----------------------------------------------------------------------===//
// This file will create a schedule that will be handed over to the pipeline
// expander.
// Software pipeliners are usually separated into two pieces, one that create a
// modulo schedule and an expander that rewrites the loop and emits a prologue
// and epilogue. This pass first calls a helper that will pre-process the IR
// to create async operations and create a modulo schedule. Then we call the
// expander to generate the prologue and new loop.
//===----------------------------------------------------------------------===//

namespace mlir {
#define GEN_PASS_DEF_TRITONGCUPINGPONGPASS
#include "Conversion/Passes.h.inc"
} // namespace mlir
#define DEBUG_TYPE "triton-gcu-pingpong"

namespace {
using namespace mlir;
using namespace mlir::triton;
using namespace mlir::triton::gpu;
// Return true if the preconditions for pipelining the loop are met.
static bool preCondition(scf::ForOp forOp) {
  // Skip loop with distance > 1 for now.
  // TODO(triton): relax the constraint in the expander.
  if (llvm::any_of(forOp.getBody()->getTerminator()->getOperands(),
                   [](Value operand) {
                     Operation *def = operand.getDefiningOp();
                     return !def;
                   }))
    return false;
  // Don't pipeline outer loops.
  if (forOp
          ->walk([&](Operation *op) {
            if (forOp.getOperation() == op)
              return WalkResult::advance();
            if (isa<scf::ForOp, scf::WhileOp>(op))
              return WalkResult::interrupt();
            return WalkResult::advance();
          })
          .wasInterrupted())
    return false;
  return true;
}

// static void tryAndPipelineOuterLoop(scf::ForOp forOp) {
//   mlir::triton::gcu::PipeliningOption options;
//   bool foundSchedule = false;
//   // Limit 2 stages to not require extra shared memory.
//   foundSchedule = gcu::getOuterLoopSchedule(forOp, /*numStage=*/2, options);
//   if (!foundSchedule)
//     return;
//   IRRewriter rewriter(forOp->getContext());
//   rewriter.setInsertionPoint(forOp);
//   FailureOr<scf::ForOp> newForOp =
//       mlir::triton::gcu::pipelineForLoop(rewriter, forOp, options);
//   (void)newForOp;
// }

static bool pipelineLoop(scf::ForOp forOp, int numStages) {
  mlir::triton::gcu::PipeliningOption options;
  if (!preCondition(forOp))
    return false;

  bool foundSchedule = false;
  foundSchedule = gcu::preProcessLoopAndGetSchedule(forOp, numStages, options);

  // TODO(triton): add more pipelines strategy.
  if (!foundSchedule)
    return false;

  IRRewriter rewriter(forOp->getContext());
  rewriter.setInsertionPoint(forOp);
  FailureOr<scf::ForOp> newForOp =
      mlir::triton::gcu::pipelineForLoop(rewriter, forOp, options);

  if (failed(newForOp))
    return false;
  // gcu need post process last dte config
  auto newFor = *newForOp;
  LLVM_DEBUG({
    llvm::dbgs() << "post process \n";
    newFor.getOperation()->getParentOp()->dump();
  });
  auto step = newFor.getStep();
  auto upperBound = newFor.getUpperBound();
  auto forIdx = newFor.getInductionVar();
  std::vector<Operation *> eraseOps;
  OpBuilder builder(newFor.getContext());
  builder.setInsertionPoint(newFor.getOperation());
  auto twoStep = builder.create<arith::MulIOp>(
      step.getLoc(), step,
      builder.create<arith::ConstantOp>(
          step.getLoc(),
          builder.getIntegerAttr(step.getType(), numStages - 1)));
  for (Operation &op : newFor.getBody()->without_terminator()) {
    if (isa<triton::gcu::AsyncLoadGlobalToShareOp>(op)) {
      SmallVector<std::pair<Operation *, int>> queue;
      for (auto &use : op.getUses()) {
        queue.push_back({use.getOwner(), use.getOperandNumber()});
      }
      if (queue.size() == 1 && isa<scf::YieldOp>(queue[0].first)) {
        builder.setInsertionPoint(&op);
        auto loc = op.getLoc();
        auto lastLastLoad = builder.create<arith::CmpIOp>(
            loc, arith::CmpIPredicate::slt,
            builder.create<arith::AddIOp>(loc, forIdx, twoStep), upperBound);
        auto ifop = builder.create<scf::IfOp>(
            loc, lastLastLoad,
            [&](OpBuilder &builder, Location loc) {
              auto clone = builder.clone(op);
              llvm::SmallVector<Value> yieldOperands;
              yieldOperands.push_back(clone->getResult(0));
              builder.create<scf::YieldOp>(loc, yieldOperands);
            },
            [&](OpBuilder &builder, Location loc) {
              llvm::SmallVector<Value> yieldOperands;
              yieldOperands.push_back(newFor.getInitArgs()[queue[0].second]);
              builder.create<scf::YieldOp>(loc, yieldOperands);
            });
        op.getResult(0).replaceAllUsesWith(ifop.getResult(0));
        eraseOps.push_back(&op);
      }
    } else if (isa<triton::gcu::AsyncWaitOp>(op)) {
      SmallVector<std::pair<Operation *, int>> queue;
      for (auto &use : op.getUses()) {
        queue.push_back({use.getOwner(), use.getOperandNumber()});
      }
      if (queue.size() == 1 && isa<scf::YieldOp>(queue[0].first)) {
        builder.setInsertionPoint(&op);
        auto loc = op.getLoc();
        auto lastLastWait = builder.create<arith::CmpIOp>(
            loc, arith::CmpIPredicate::slt,
            builder.create<arith::AddIOp>(loc, forIdx, step), upperBound);
        auto ifop = builder.create<scf::IfOp>(
            loc, lastLastWait,
            [&](OpBuilder &builder, Location loc) {
              auto clone = builder.clone(op);
              llvm::SmallVector<Value> yieldOperands;
              yieldOperands.push_back(clone->getResult(0));
              builder.create<scf::YieldOp>(loc, yieldOperands);
            },
            [&](OpBuilder &builder, Location loc) {
              llvm::SmallVector<Value> yieldOperands;
              yieldOperands.push_back(newFor.getInitArgs()[queue[0].second]);
              builder.create<scf::YieldOp>(loc, yieldOperands);
            });
        op.getResult(0).replaceAllUsesWith(ifop.getResult(0));
        eraseOps.push_back(&op);
      }
    }
  }
  for (auto erase : eraseOps) {
    erase->erase();
  }
  return true;
}

struct TritonGCUPingpongPass
    : public mlir::impl::TritonGCUPingpongPassBase<TritonGCUPingpongPass> {
  using Base::Base;
  int getNumStagesOrDefault(scf::ForOp forOp) {
    // Use the attribute attached to the loop if it exists otherwise use the
    // global control.
    if (!forOp->hasAttr("tt.num_stages")) {
      return numStages > 6 ? 6 : numStages;
    }
    int stageNumber =
        mlir::cast<IntegerAttr>(forOp->getAttr("tt.num_stages")).getInt();
    if (stageNumber > 6) {
      stageNumber = 6;
    }
    return stageNumber;
  }
  void runOnOperation() override;
  void getDependentDialects(DialectRegistry &registry) const override {
    registry
        .insert<arith::ArithDialect, memref::MemRefDialect,
                triton::TritonDialect, mlir::triton::gcu::TritonGCUDialect>();
  }
};
} // namespace

using namespace mlir;
void TritonGCUPingpongPass::runOnOperation() {
  LLVM_DEBUG({ llvm::dbgs() << "enter  TritonGCUPingpongPass\n"; });
  SmallVector<scf::ForOp> loops;
  getOperation()->walk([&](scf::ForOp forOp) {
    // Bail out for loops with num_stage <= 1.
    if (getNumStagesOrDefault(forOp) > 2)
      loops.push_back(forOp);
  });

  if (loops.empty()) {
    LLVM_DEBUG({ llvm::dbgs() << "no loops \n"; });
    return;
  }
  llvm::SmallSetVector<scf::ForOp, 8> outerLoops;
  for (scf::ForOp forOp : loops) {
    auto outerLoop = dyn_cast<scf::ForOp>(forOp->getParentOp());
    int loopNumStages = getNumStagesOrDefault(forOp);
    bool pipelined = pipelineLoop(forOp, loopNumStages);
    if (pipelined && outerLoop && getNumStagesOrDefault(outerLoop) > 2)
      outerLoops.insert(outerLoop);
  }

  // Clean up arithmetic before applying the next level of pipelining to
  // simplify the IR.
  auto arithDialect =
      getOperation().getContext()->getLoadedDialect<arith::ArithDialect>();
  RewritePatternSet patterns(getOperation().getContext());
  arithDialect->getCanonicalizationPatterns(patterns);
  if (applyPatternsGreedily(getOperation(), std::move(patterns)).failed())
    return signalPassFailure();

  // Try to pipeline the outer loop to overlap the prologue and epilogue of
  // the inner loop.
  // for (scf::ForOp outerLoop : outerLoops)
  //   tryAndPipelineOuterLoop(outerLoop);

  // Re-collect loop ops todo support store latter
  // loops.clear();
  // getOperation()->walk([&](scf::ForOp forOp) {
  //   // Bail out for loops with num_stage <= 1.
  //   if (getNumStagesOrDefault(forOp) > 1)
  //     loops.push_back(forOp);
  // });

  // for (scf::ForOp forOp : loops) {
  //   mlir::triton::pipelineTMAStores(forOp);
  // }
}
