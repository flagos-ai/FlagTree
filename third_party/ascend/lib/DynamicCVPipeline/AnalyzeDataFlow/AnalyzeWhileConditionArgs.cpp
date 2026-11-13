/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
 * THE SOFTWARE.
 */

#include "ascend/include/DynamicCVPipeline/AnalyzeDataFlow.h"
#include "ascend/include/DynamicCVPipeline/Common/Utils.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Value.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Debug.h"

static constexpr const char *DEBUG_TYPE = "analyze-while-condition-args";
#define DBGS() (llvm::dbgs() << '[' << DEBUG_TYPE << "] ")
#define LDBG(...)                                                              \
  LLVM_DEBUG({                                                                 \
    DBGS();                                                                    \
    llvm::dbgs() << __VA_ARGS__;                                               \
    llvm::dbgs() << "\n";                                                      \
  })

using namespace llvm;
using namespace mlir;
using namespace triton;
using namespace CVPipeline;

// Expand a block argument reached by the backward trace into the values that
// update it per iteration:
//   - a before-region arg is updated by the after-region scf.yield operand of
//     the same index (the init value enters once from outside and is not
//     traced);
//   - an after-region arg is the value forwarded by scf.condition at the same
//     index.
// Args of enclosing ops (outer loops, function) are outside the while's
// update process and are not expanded.
static void expandBlockArgument(scf::WhileOp whileOp, BlockArgument blockArg,
                                SmallVectorImpl<Value> &worklist) {
  Block *owner = blockArg.getOwner();
  unsigned argIdx = blockArg.getArgNumber();

  if (owner == whileOp.getBeforeBody()) {
    worklist.push_back(whileOp.getYieldOp()->getOperand(argIdx));
    return;
  }

  if (owner == whileOp.getAfterBody()) {
    worklist.push_back(whileOp.getConditionOp().getArgs()[argIdx]);
    return;
  }
}

// Expand an op result reached by the backward trace into the op's operands.
// Computation outside the while op is not traced through. For region-holding
// ops (scf.if/for/while nested in the loop), the results are produced by the
// region terminators, so their operands are conservatively traced too.
static void expandOpResult(scf::WhileOp whileOp, OpResult result,
                           SmallVectorImpl<Value> &worklist) {
  Operation *op = result.getOwner();

  if (!whileOp->isProperAncestor(op)) {
    return;
  }

  worklist.append(op->operand_begin(), op->operand_end());

  for (Region &region : op->getRegions()) {
    for (Block &block : region) {
      if (Operation *terminator = block.getTerminator()) {
        worklist.append(terminator->operand_begin(), terminator->operand_end());
      }
    }
  }
}

// Returns true when `value` is a non-scalar (tensor) value produced inside
// `whileOp`, either by an op nested in one of its regions or as a loop-carried
// arg of the while itself or of a nested loop.
//
// A tensor defined outside the loop is not a problem: its computation stays
// out of the scope that has to be cloned, and only the scalar extracted from
// it takes part in the condition update chain.
static bool isNonScalarProducedInside(scf::WhileOp whileOp, Value value) {
  if (!isa<TensorType>(value.getType())) {
    return false;
  }

  if (auto blockArg = dyn_cast<BlockArgument>(value)) {
    Operation *parentOp = blockArg.getOwner()->getParentOp();
    return parentOp && (parentOp == whileOp.getOperation() ||
                        whileOp->isProperAncestor(parentOp));
  }

  return whileOp->isProperAncestor(value.getDefiningOp());
}

// Returns true when the condition of `whileOp` transitively depends on a
// non-scalar (tensor) value produced inside the loop.
//
// The trace starts from the condition operand of scf.condition and walks the
// use-def chain backwards, following loop-carried args across both regions.
static bool conditionDependsOnTensor(scf::WhileOp whileOp) {
  SmallVector<Value> worklist{whileOp.getConditionOp().getCondition()};
  llvm::DenseSet<Value> visited;

  while (!worklist.empty()) {
    Value value = worklist.pop_back_val();
    if (!visited.insert(value).second) {
      continue;
    }

    if (isNonScalarProducedInside(whileOp, value)) {
      LDBG("condition of " << *whileOp
                           << " depends on tensor value: " << value);
      return true;
    }

    if (auto blockArg = dyn_cast<BlockArgument>(value)) {
      expandBlockArgument(whileOp, blockArg, worklist);
    } else {
      expandOpResult(whileOp, cast<OpResult>(value), worklist);
    }
  }
  return false;
}

void AnalyzeWhileConditionArgsPass::runOnOperation() {
  ModuleOp module = getOperation();

  if (CVPipeline::hasFallbackAttr(module)) {
    return;
  }

  WalkResult walkResult = module.walk([&](scf::WhileOp whileOp) -> WalkResult {
    if (CVPipeline::isMainLoopOp(whileOp) &&
        conditionDependsOnTensor(whileOp)) {
      return WalkResult::interrupt();
    }
    return WalkResult::advance();
  });

  if (walkResult.wasInterrupted()) {
    LDBG("scf.while condition depends on non-scalar data, the "
         "DynamicCVPipeline pass will be interrupted.");
    CVPipeline::setFallbackAttr(module, CVPipeline::ERRCODE_IGNORED);
  }
}

namespace mlir {
namespace triton {

std::unique_ptr<OperationPass<ModuleOp>> createAnalyzeWhileConditionArgsPass() {
  return std::make_unique<AnalyzeWhileConditionArgsPass>();
}

} // namespace triton
} // namespace mlir
