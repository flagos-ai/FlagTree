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

#include "Dialect/TritonGCU/IR/TritonGCUDialect.h"
#include "Dialect/TritonGCU/IR/TritonGCUTypes.h"
#include "PipeliningUtility.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/TypeUtilities.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "mlir/Support/LLVM.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/Transforms/Passes.h"
#include "triton/Dialect/TritonGPU/Transforms/Utility.h"

using namespace mlir;
namespace tt = mlir::triton;
namespace ttg = mlir::triton::gpu;
namespace ttng = mlir::triton::nvidia_gpu;

// Combine the current mask with the given predicate.
static Value getPredMask(RewriterBase &rewriter, Type typeLike,
                         Value currentMask, Value pred) {
  Type maskType = tt::getI1SameShape(typeLike);
  Location loc = pred.getLoc();
  Value mask = pred;
  if (isa<RankedTensorType>(maskType)) {
    mask = rewriter.create<tt::SplatOp>(loc, maskType, pred);
  }
  if (currentMask) {
    mask = rewriter.create<arith::AndIOp>(loc, mask, currentMask);
  }
  return mask;
}

// Function to mask operations during scheduling.
Operation *mlir::triton::gcu::predicateOp(RewriterBase &rewriter, Operation *op,
                                          Value pred) {
  OpBuilder::InsertionGuard guard(rewriter);
  if (mlir::isMemoryEffectFree(op))
    return op;
  if (isa<mlir::triton::gcu::AsyncWaitOp>(op))
    return op;
  if (isa<ttg::LocalLoadOp>(op))
    return op;
  if (auto ifOp = dyn_cast<scf::IfOp>(op)) {
    rewriter.setInsertionPoint(op);
    Value cnd = getPredMask(rewriter, ifOp.getCondition().getType(),
                            ifOp.getCondition(), pred);
    ifOp.getConditionMutable().assign(cnd);
    return op;
  }
  if (auto asyncCopyOp =
          dyn_cast<mlir::triton::gcu::AsyncLoadGlobalToShareOp>(op)) {
    return op;
  }
  if (auto loadOp = dyn_cast<triton::gcu::LoadOp>(op)) {
    return op;
  }
  if (auto loadOp = dyn_cast<triton::LoadOp>(op)) {
    return op;
  }
  assert("don't know how to predicate this op" && false);
  return op;
}

/// Helper to recursively add dependencies to the same stage.
void mlir::triton::gcu::addDep(Operation *op, DenseSet<Operation *> &deps,
                               bool includeArg, DenseSet<Operation *> *filter) {
  if (filter && filter->count(op))
    return;
  if (!deps.insert(op).second)
    return;
  for (Value operand : op->getOperands()) {
    Value v = operand;
    llvm::DenseSet<Value> seen;
    seen.reserve(4);
    while (auto arg = mlir::dyn_cast<BlockArgument>(v)) {
      if (!includeArg)
        break;
      if (!seen.insert(v).second)
        break;
      if (arg.getArgNumber() > 0 && arg.getOwner() == op->getBlock()) {
        auto yieldOp = op->getBlock()->getTerminator();
        v = yieldOp->getOperand(arg.getArgNumber() - 1);
        continue;
      }
      break;
    }
    Operation *defOp = v.getDefiningOp();
    if (defOp && defOp->getBlock() == op->getBlock()) {
      triton::gcu::addDep(defOp, deps, includeArg, filter);
    }
  }
}

// Add operations to the schedule with the given stage based on the filter
// function.
void mlir::triton::gcu::addOps(
    scf::ForOp forOp, int stage,
    std::vector<std::pair<Operation *, unsigned>> &schedule,
    std::function<bool(Operation *)> filter) {
  for (Operation &op : forOp.getBody()->without_terminator()) {
    if (!filter(&op))
      continue;
    schedule.emplace_back(&op, stage);
  }
}
