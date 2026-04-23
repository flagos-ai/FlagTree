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
#include "Dialect/TritonGCU/IR/TritonGCUDialect.h"
#include "Dialect/TritonGCU/IR/TritonGCUTypes.h"
#include "Transforms/Passes.h"
#include "Utility.h"

#include "mlir/Analysis/SliceAnalysis.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/Passes.h"
#include "triton/Dialect/TritonGPU/Transforms/Utility.h"
#include "llvm/Support/DebugLog.h"

#define DEBUG_TYPE "gcu-ws-task-partition"

using namespace mlir;

namespace tt = mlir::triton;
namespace ttgcu = mlir::triton::gcu;

namespace mlir::triton::gcu {

// Compute a partition schedule for later passes to actually partition the
// program into async tasks.
int doTaskPartition(tt::FuncOp &funcOp, unsigned numWarpGroups) {
  if (numWarpGroups <= 1) {
    LDBG() << "Failed to doTaskPartition: numWarpGroups <= 1";
    return -1;
  }

  // Bail out in the presence of user annotations.
  DenseSet<int> allAsyncTasks;
  funcOp->walk([&](Operation *op) {
    auto asyncTasks = getAsyncTaskIds(op);
    allAsyncTasks.insert_range(getAsyncTaskIds(op));
  });
  if (!allAsyncTasks.empty()) {
    LDBG() << "Failed to doTaskPartition: allAsyncTasks is not empty";
    return -1;
  }

  SmallVector<scf::ForOp> loops;
  SmallVector<Operation *> loads;
  SmallVector<Operation *> stores;
  SmallVector<Operation *> dots;

  funcOp.walk([&](Operation *op) {
    if (scf::ForOp forOp = dyn_cast<scf::ForOp>(op))
      loops.push_back(forOp);
    else if (isa<ttgcu::WarpGroupDotOp>(op))
      dots.push_back(op);
    else if (isa<tt::LoadOp>(op))
      loads.push_back(op);
    else if (isa<tt::StoreOp>(op))
      stores.push_back(op);
  });

  if (loops.empty() || loads.empty() || dots.empty()) {
    LDBG() << "Failed to doTaskPartition: not find ForOp, LoadOp, or "
              "WarpGroupDotOp";
    return -1;
  }

  // Step 1. Select loads into the first task, which is the producer task by
  // default. Place dots into the second task, which is the consumer.
  // Only consider loads that are connected to a dot op in a loop.
  DenseSet<Operation *> producerOps;
  SmallVector<Operation *> consumerOps;
  BackwardSliceOptions opt;
  opt.omitBlockArguments = true;
  opt.omitUsesFromAbove = true;
  opt.inclusive = true;

  for (auto op : dots) {
    consumerOps.push_back(op);
    auto dotOp = dyn_cast<ttgcu::WarpGroupDotOp>(op);
    if (!dotOp)
      continue;
    SetVector<Operation *> backwardSlice;
    (void)getBackwardSlice(dotOp.getA(), &backwardSlice, opt);
    (void)getBackwardSlice(dotOp.getB(), &backwardSlice, opt);
    for (auto depOp : backwardSlice) {
      if (isa<tt::LoadOp>(depOp) && isExpensiveLoadOrStore(depOp)) {
        producerOps.insert(depOp);
      }
    }
  }

  LLVM_DEBUG({
    LDBG() << "Producer ops:";
    for (auto op : producerOps) {
      op->dump();
    }

    LDBG() << "Consumer ops:";
    for (auto op : consumerOps) {
      op->dump();
    }
  });

  if (consumerOps.empty() || producerOps.empty()) {
    LDBG() << "Failed to doTaskPartition: consumerOps.empty() || "
              "producerOps.empty()";
    return -1;
  }

  // Annotate the program with task ids
  SmallVector<AsyncTaskId, 1> producerTaskIds{0};
  SmallVector<AsyncTaskId, 1> consumerTaskIds;
  for (unsigned i = 0; i < numWarpGroups - 1; ++i) {
    consumerTaskIds.push_back(i + producerTaskIds.size());
  }

  for (auto op : producerOps) {
    setAsyncTaskIds(op, producerTaskIds);
  }

  for (auto op : consumerOps) {
    setAsyncTaskIds(op, consumerTaskIds);
  }

  // All stores go with the consumers.
  for (auto op : stores) {
    setAsyncTaskIds(op, consumerTaskIds);
  }

  LLVM_DEBUG({
    LDBG() << "After WS task partition";
    funcOp.dump();
  });
  return 0;
}

} // namespace mlir::triton::gcu

namespace mlir {
#define GEN_PASS_DEF_GCUTESTWSTASKPARTITION
#include "Transforms/Passes.h.inc"
} // namespace mlir

namespace {

class GCUTestWSTaskPartitionPass
    : public impl::GCUTestWSTaskPartitionBase<GCUTestWSTaskPartitionPass> {
public:
  using impl::GCUTestWSTaskPartitionBase<
      GCUTestWSTaskPartitionPass>::GCUTestWSTaskPartitionBase;

  void runOnFuncOp(triton::FuncOp funcOp) {
    if (numWarpGroups > 1)
      ttgcu::doTaskPartition(funcOp, numWarpGroups);
  }

  void runOnOperation() override {
    getOperation()->walk([&](triton::FuncOp funcOp) { runOnFuncOp(funcOp); });
  }
};

} // namespace
