/**
 * Copyright 2025-2026 Enflame. All Rights Reserved.
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
#include <algorithm>

#include "Dialect/TritonGCU/IR/TritonGCUDialect.h"
#include "Dialect/TritonGCU/IR/TritonGCUTypes.h"
#include "TaskIdPropagation.h"
#include "Transforms/Passes.h"
#include "Utility.h"

#include "mlir/Analysis/DataFlow/ConstantPropagationAnalysis.h"
#include "mlir/Analysis/DataFlow/DeadCodeAnalysis.h"
#include "mlir/Analysis/SliceAnalysis.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/Passes.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/Transforms/Utility.h"
#include "llvm/Support/DebugLog.h"

#define DEBUG_TYPE "gcu-ws-task-id-propagate"

using namespace mlir;
using namespace mlir::dataflow;

namespace ttgcu = mlir::triton::gcu;

namespace mlir::triton::gcu {

int doTaskIdPropagate(triton::FuncOp &funcOp) {
  SymbolTableCollection symbolTable;
  Operation *op = funcOp.getOperation();
  DataFlowSolver solver;

  solver.load<DeadCodeAnalysis>();
  solver.load<SparseConstantPropagation>();
  solver.load<ttgcu::TaskIdBackwardPropagation>(symbolTable);
  if (failed(solver.initializeAndRun(op))) {
    LDBG() << "Failed to doTaskIdPropagate initializeAndRun";
    return -1;
  }

  LDBG() << "Start walking";
  funcOp.walk([&](mlir::Operation *op) {
    auto taskIds = ttgcu::TaskId::getUninitialized();
    // Get the union of the results
    for (auto result : op->getResults()) {
      auto *lattice = solver.lookupState<ttgcu::TaskIdLattice>(result);
      if (!lattice)
        llvm_unreachable("Lattice not found.");
      taskIds = taskIds.meet(taskIds, lattice->getValue());
    }
    // Get the union of the operands
    if (op->getNumResults() == 0) {
      for (auto operand : op->getOperands()) {
        auto *lattice = solver.lookupState<ttgcu::TaskIdLattice>(operand);
        if (!lattice)
          llvm_unreachable("Lattice not found.");
        taskIds = taskIds.meet(taskIds, lattice->getValue());
      }
    }
    // TODO(Arda): Ideally front-end should not allow constant ops to be
    // annotated. Anchor constants cause problems.
    if (!taskIds.isUninitialized() &&
        (isa<arith::ConstantOp>(op) || !op->hasAttr("async_task_id"))) {
      op->setAttr("async_task_id", taskIds.getTaskIds());
      labelParentOps(op);
    }
  });
  return 0;
}

} // namespace mlir::triton::gcu

namespace mlir {
#define GEN_PASS_DEF_GCUTESTWSTASKIDPROPAGATE
#include "Transforms/Passes.h.inc"
} // namespace mlir

namespace {

class GCUTestWSTaskIdPropagatePass
    : public impl::GCUTestWSTaskIdPropagateBase<GCUTestWSTaskIdPropagatePass> {
public:
  using impl::GCUTestWSTaskIdPropagateBase<
      GCUTestWSTaskIdPropagatePass>::GCUTestWSTaskIdPropagateBase;

  void runOnFuncOp(triton::FuncOp funcOp) {
    llvm::DenseSet<Operation *> anchorOps;
    funcOp.walk([&](mlir::Operation *op) {
      auto asyncTasks = ttgcu::getAsyncTaskIds(op);
      if (!asyncTasks.empty()) {
        std::sort(asyncTasks.begin(), asyncTasks.end());
        ttgcu::setAsyncTaskIds(op, asyncTasks);
        if (!isa<arith::ConstantOp, arith::ConstantIntOp>(op))
          anchorOps.insert(op);
        if (numWarpGroups == 0)
          op->removeAttr("async_task_id");
      }
    });
    if (numWarpGroups == 0 || anchorOps.empty())
      return;
    int retCode = ttgcu::doTaskIdPropagate(funcOp);
    if (retCode != 0)
      signalPassFailure();
  }

  void runOnOperation() override {
    getOperation()->walk([&](triton::FuncOp funcOp) { runOnFuncOp(funcOp); });
  }
};

} // namespace
