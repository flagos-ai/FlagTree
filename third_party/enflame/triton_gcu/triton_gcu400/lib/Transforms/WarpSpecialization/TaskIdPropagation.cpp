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
#include <utility>

#include "TaskIdPropagation.h"
#include "Utility.h"

#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Support/LLVM.h"
#include "triton/Dialect/TritonGPU/Transforms/Utility.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/DebugLog.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/raw_ostream.h"

#define DEBUG_TYPE "gcu-ws-task-id-propagation"

using namespace mlir;
using namespace mlir::dataflow;

namespace mlir::triton::gcu {

//===----------------------------------------------------------------------===//
// TaskId
//===----------------------------------------------------------------------===//

void TaskId::print(raw_ostream &os) const {
  if (isUninitialized()) {
    os << "<UNINITIALIZED>";
    return;
  }
  if (isUnknown()) {
    os << "<UNKNOWN>";
    return;
  }
  return getTaskIds().print(os);
}

TaskId TaskId::join(const TaskId &lhs, const TaskId &rhs) {
  return TaskId::getUnknownTaskId();
}

TaskId TaskId::meet(const TaskId &lhs, const TaskId &rhs) {
  if (lhs.isUnknown() || rhs.isUnknown())
    return TaskId::getUnknownTaskId();
  if (lhs.isUninitialized())
    return rhs;
  if (rhs.isUninitialized())
    return lhs;
  if (lhs == rhs)
    return lhs;

  auto context = lhs.getTaskIds().getContext();
  auto lhsTasks = lhs.getTaskIds().asArrayRef();
  auto rhsTasks = rhs.getTaskIds().asArrayRef();
  // Meet the task ids by merging and deduplicating them
  SmallVector<AsyncTaskId> result(lhsTasks.begin(), lhsTasks.end());
  result.insert(result.end(), rhsTasks.begin(), rhsTasks.end());
  std::sort(result.begin(), result.end());
  result.erase(std::unique(result.begin(), result.end()), result.end());
  auto mergedAndDedupedTaskIds =
      TaskId(DenseI32ArrayAttr::get(context, ArrayRef<AsyncTaskId>(result)));
  return mergedAndDedupedTaskIds;
}

//===----------------------------------------------------------------------===//
// TaskIdBackwardPropagation
//===----------------------------------------------------------------------===//

void TaskIdBackwardPropagation::propagateToYield(
    scf::YieldOp yieldOp, SmallVector<TaskId> &lattices) {
  LDBG() << "TaskIdBackwardPropagation propagateToYield";
  for (auto [lattice, yieldOperand] :
       llvm::zip_equal(lattices, yieldOp->getOperands())) {
    auto yieldLattice = getLatticeElement(yieldOperand);
    ChangeResult changed = yieldLattice->meet(lattice);
    propagateIfChanged(yieldLattice, changed);
  }
}

void TaskIdBackwardPropagation::propagateToTerminator(
    Operation *op, ArrayRef<const TaskIdLattice *> &lattices) {
  LDBG() << "TaskIdBackwardPropagation propagateToTerminator";
  for (auto [lattice, terminatorOperand] :
       llvm::zip_equal(lattices, op->getOperands())) {
    auto terminatorLattice = getLatticeElement(terminatorOperand);
    ChangeResult changed = terminatorLattice->meet(lattice->getValue());
    propagateIfChanged(terminatorLattice, changed);
  }
}

void TaskIdBackwardPropagation::propagateToParent(Operation *op,
                                                  const TaskId &taskId) {
  LDBG() << "TaskIdBackwardPropagation propagateToParent";
  auto parentOp = op->getParentOp();
  while (parentOp && !isa<triton::FuncOp>(parentOp)) {
    if (auto forOp = dyn_cast<scf::ForOp>(parentOp)) {
      // Propagate to the control operands of the for op.
      for (auto controlOperand :
           forOp.getOperands().take_front(forOp.getNumControlOperands())) {
        auto controlLattice = getLatticeElement(controlOperand);
        ChangeResult changed = controlLattice->meet(taskId);
        propagateIfChanged(controlLattice, changed);
      }
    } else if (auto whileOp = dyn_cast<scf::WhileOp>(parentOp)) {
      // scf.while has no separate control operands: its loop-carried inits
      // feed the loop condition, so they are computed by the same task as the
      // loop body (the analogue of the for op's bounds above).
      for (auto init : whileOp.getInits()) {
        auto initLattice = getLatticeElement(init);
        ChangeResult changed = initLattice->meet(taskId);
        propagateIfChanged(initLattice, changed);
      }
    } else if (auto ifOp = dyn_cast<scf::IfOp>(parentOp)) {
      auto cond = ifOp.getCondition();
      auto condLattice = getLatticeElement(cond);
      ChangeResult changed = condLattice->meet(taskId);
      propagateIfChanged(condLattice, changed);
    } else if (!isa<triton::FuncOp, triton::ReduceOp>(parentOp)) {
      LDBG() << "propagateToParent: unsupported parent op " << *parentOp;
      unsupportedControlFlow = true;
      return;
    }
    parentOp = parentOp->getParentOp();
  }
}

LogicalResult TaskIdBackwardPropagation::visitOperation(
    Operation *op, ArrayRef<TaskIdLattice *> operands,
    ArrayRef<const TaskIdLattice *> results) {
  LDBG() << "TaskIdBackwardPropagation visitOperation";
  // Already annotated
  // TODO(Arda): Replace the following with getAsyncTaskIds when we no longer
  // need to dump the task ids into the IR.
  auto taskIdAttr = op->getAttrOfType<DenseI32ArrayAttr>("async_task_id");
  if (taskIdAttr) {
    LDBG() << "TaskIdBackwardPropagation annotated taskid";
    const auto annotated = TaskId(taskIdAttr);
    for (auto operandLattice : operands) {
      ChangeResult changed = operandLattice->meet(annotated);
      propagateIfChanged(operandLattice, changed);
    }
    // Propagate to the parent ops such as control flows
    propagateToParent(op, annotated);

    if (op->getNumRegions() == 1) {
      if (auto reduceOp = dyn_cast<triton::ReduceOp>(op)) {
        propagateToTerminator(reduceOp.getCombineOp().front().getTerminator(),
                              results);
      }
    }

    return success();
  }
  // If it is not annotated by the user, propagate from results to the
  // operands
  for (const auto resultLattice : results) {
    for (auto operandLattice : operands) {
      ChangeResult changed = operandLattice->meet(resultLattice->getValue());
      propagateIfChanged(operandLattice, changed);
    }
  }

  for (const auto resultLattice : results)
    propagateToParent(op, resultLattice->getValue());

  if (op->getNumRegions() == 1) {
    if (auto reduceOp = dyn_cast<triton::ReduceOp>(op)) {
      propagateToTerminator(reduceOp.getCombineOp().front().getTerminator(),
                            results);
    }
  }

  return success();
}

void TaskIdBackwardPropagation::visitBranchOperand(OpOperand &operand) {
  LDBG() << "TaskIdBackwardPropagation visitBranchOperand";
  auto defOp = operand.getOwner();

  // scf.condition terminates the "before" region of an scf.while. The only
  // operand of it that the framework does not forward to a successor region is
  // the continuation flag itself. That flag decides whether the loop body runs
  // again, so it must be evaluated by every task that observes the loop -- and
  // nothing else can give it a task id: the flag is consumed by the terminator
  // alone, so without meeting the loop's task ids into it its producers stay
  // unannotated and a per-task clone of the loop is left with a dangling
  // condition. Meet the loop's result task ids, plus the loop state the
  // condition is derived from; the framework registers a dependency on the
  // loop's results, so this visit is re-run once they are known.
  if (auto condOp = dyn_cast<scf::ConditionOp>(defOp)) {
    auto *condLattice = getLatticeElement(operand.get());
    if (auto whileOp = dyn_cast<scf::WhileOp>(condOp->getParentOp())) {
      for (auto result : whileOp->getResults()) {
        ChangeResult changed =
            condLattice->meet(getLatticeElement(result)->getValue());
        propagateIfChanged(condLattice, changed);
      }
    }
    for (auto arg : condOp.getArgs()) {
      ChangeResult changed =
          condLattice->meet(getLatticeElement(arg)->getValue());
      propagateIfChanged(condLattice, changed);
    }
    return;
  }

  // The other owners we can be called with are the branch ops themselves: their
  // operands that are consumed by the op instead of forwarded to a region (the
  // bounds of a for op, the condition of an if op, the inits of a while op).
  if (!isa<scf::ForOp, scf::IfOp, scf::WhileOp>(defOp)) {
    LDBG() << "visitBranchOperand: unsupported branch op " << *defOp;
    unsupportedControlFlow = true;
    return;
  }

  SmallVector<TaskId> lattices(defOp->getNumResults(),
                               TaskId::getUninitialized());
  for (auto [i, result] : llvm::enumerate(defOp->getResults())) {
    auto resultLattice = getLatticeElement(result);
    // Wait for all the results to be initialized.
    if (resultLattice->getValue().isUninitialized())
      return;
    lattices[i] =
        resultLattice->getValue().meet(lattices[i], resultLattice->getValue());
  }

  // Propagate to the yield ops
  if (auto forOp = dyn_cast<scf::ForOp>(defOp)) {
    auto yieldOp = cast<scf::YieldOp>(forOp.getBody()->getTerminator());
    propagateToYield(yieldOp, lattices);
  } else if (auto ifOp = dyn_cast<scf::IfOp>(defOp)) {
    propagateToYield(ifOp.thenYield(), lattices);
    if (!ifOp.getElseRegion().empty())
      propagateToYield(ifOp.elseYield(), lattices);
  } else if (auto whileOp = dyn_cast<scf::WhileOp>(defOp)) {
    // The loop's results are the values yielded by the "after" region.
    auto yieldOp = cast<scf::YieldOp>(whileOp.getAfterBody()->getTerminator());
    propagateToYield(yieldOp, lattices);
  }
  return;

  // TODO(Arda): Address what happens when loop is annotated
}

void TaskIdBackwardPropagation::visitCallOperand(OpOperand &operand) {
  llvm_unreachable(
      "Should not have any call operands in the IR after inlining.");
}

void TaskIdBackwardPropagation::setToExitState(TaskIdLattice *lattice) {}

} // namespace mlir::triton::gcu
