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

#include "ascend/include/DynamicCVPipeline/SplitDataflow/RefineArgsBlockId.h"
#include "DynamicCVPipeline/Common/MemoryEffectsTracker.h"
#include "ascend/include/DynamicCVPipeline/Common/Utils.h"
#include "ascend/include/DynamicCVPipeline/ComputeBlockOpt/Common.h"
#include "ascend/include/DynamicCVPipeline/PlanComputeBlock/Common.h"
#include "ascend/include/DynamicCVPipeline/PlanComputeBlock/ComputeBlockIdManager.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/Value.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/Support/Debug.h"

using namespace mlir;

static constexpr const char *DEBUG_TYPE = "refine-args-block-id";
#define LOG_DEBUG(...)                                                         \
  LLVM_DEBUG(llvm::dbgs() << " [" << DEBUG_TYPE << "] " << __VA_ARGS__)

using namespace mlir::triton;

static void eraseOpsWithUnusedUsers(Operation *op, Block *loopBlock,
                                    CVPipeline::ComputeBlockIdManager &bm) {
  llvm::SetVector<Operation *> worklist;

  worklist.insert(op);

  while (!worklist.empty()) {
    Operation *cur = worklist.pop_back_val();
    // Stay inside the loop body the caller handed us.
    if (cur->getBlock() != loopBlock) {
      continue;
    }
    // No users left and no side effects to lose: the only condition under which
    // erasing is legal.
    if (!isOpTriviallyDead(cur)) {
      continue;
    }
    // Operands have to be collected before erase(): afterwards `cur` is gone.
    SmallVector<Operation *> producers;
    for (Value operand : cur->getOperands()) {
      if (Operation *defOp = operand.getDefiningOp()) {
        producers.push_back(defOp);
      }
    }
    bm.forgetOp(cur);
    cur->erase();
    // Re-examine the producers now that their uses have actually been dropped.
    worklist.insert(producers.begin(), producers.end());
  }
}

static bool cloneDepSubgraph(Operation *yieldDefOp, Block *loopBlock,
                             SmallVector<Operation *> &clonedOps,
                             IRMapping &mapping, int targetBlockId,
                             CVPipeline::ComputeBlockIdManager &bm) {
  SmallVector<Operation *> worklist;
  llvm::SetVector<Operation *> visited;
  SmallVector<Operation *> toClone;

  // First pass: BFS to collect ops to clone
  visited.insert(yieldDefOp);
  toClone.push_back(yieldDefOp);
  worklist.push_back(yieldDefOp);

  while (!worklist.empty()) {
    Operation *cur = worklist.pop_back_val();

    for (Value operand : cur->getOperands()) {
      Operation *defOp = operand.getDefiningOp();
      if (!defOp || defOp->getBlock() != loopBlock || visited.count(defOp)) {
        continue;
      }
      if (defOp != yieldDefOp &&
          (defOp->getNumResults() != 1 ||
           !CVPipeline::isScalarLike(defOp->getResult(0)))) {
        return false;
      }
      visited.insert(defOp);
      toClone.push_back(defOp);
      worklist.push_back(defOp);
    }
  }

  // Second pass: clone in code order and update mapping
  llvm::sort(toClone,
             [](Operation *l, Operation *r) { return l->isBeforeInBlock(r); });

  for (Operation *cur : toClone) {
    Operation *cloned = cur->clone(mapping);
    loopBlock->push_back(cloned);
    bm.updateBlockId(cloned, -1);
    cloned->moveAfter(cur);
    clonedOps.push_back(cloned);
    mapping.map(cur->getResult(0), cloned->getResult(0));
  }

  // Move cloned ops to after the last op in target block
  auto targetOps = bm.getOpsByBlockId(targetBlockId);
  Operation *lastOpInTargetBlock = nullptr;
  for (Operation *op : targetOps) {
    if (op->getBlock() != loopBlock)
      continue;
    if (!lastOpInTargetBlock || lastOpInTargetBlock->isBeforeInBlock(op)) {
      lastOpInTargetBlock = op;
    }
  }
  if (lastOpInTargetBlock) {
    for (Operation *cloned : llvm::reverse(clonedOps)) {
      cloned->moveAfter(lastOpInTargetBlock);
      bm.updateBlockId(cloned, targetBlockId);
    }
  }

  return true;
}

static int findFirstUser(Value iterArg, Block *loopBlock,
                         CVPipeline::ComputeBlockIdManager &bm) {
  llvm::SetVector<Value> visited;
  SmallVector<std::pair<Value, int>> worklist;
  int firstUserBlockId = -1;
  Operation *firstUserOp = nullptr;

  for (OpOperand &use : iterArg.getUses()) {
    Operation *user = use.getOwner();
    auto userInblock = CVPipeline::getAncestorInBlock(user, loopBlock);
    if (!userInblock) {
      continue;
    }
    if (isa<scf::YieldOp>(userInblock)) {
      continue;
    }
    if (firstUserOp == nullptr || userInblock->isBeforeInBlock(firstUserOp)) {
      firstUserOp = userInblock;
      firstUserBlockId = bm.getBlockIdByOp(userInblock);
    }
  }
  return firstUserBlockId;
}

static bool
isDependenceOther(Operation *yieldDefOp, Block *loopBlock, int argsId,
                  const CVPipeline::MemoryDependenceGraph &memGraph) {
  // Now we only filter the different iter_arg dependency.
  for (Value operand : yieldDefOp->getOperands()) {
    if (operand.getDefiningOp())
      continue;
    // if have block argument from for block. Skip;
    if (CVPipeline::getLoopCarriedArgIndex(operand, loopBlock) != argsId) {
      LOG_DEBUG("Yield def op depends on other arg:"
                << CVPipeline::getLoopCarriedArgIndex(operand, loopBlock)
                << "\n");
      return true;
    }
  }

  for (auto memDep : memGraph.getExecBefore(yieldDefOp)) {
    auto userInBlock = CVPipeline::getAncestorInBlock(memDep, loopBlock);
    if (userInBlock) {
      LOG_DEBUG("Yield def op depends on other memory in for block: " << *memDep
                                                                      << "\n");
      return true;
    }
  }
  return false;
}

static void processOneLoop(Operation *loopOp,
                           CVPipeline::ComputeBlockIdManager &bm,
                           const CVPipeline::MemoryDependenceGraph &memGraph) {
  auto ml = CVPipeline::MainLoop(loopOp);
  Block *loopBlock = nullptr;
  for (Region &region : loopOp->getRegions()) {
    if (auto *terminator = region.front().getTerminator()) {
      if (isa<scf::YieldOp>(terminator)) {
        loopBlock = &region.front();
        break;
      }
    }
  }
  if (!loopBlock) {
    LOG_DEBUG("No yield op found in loop block\n");
    return;
  }
  auto yieldOp = cast<scf::YieldOp>(loopBlock->getTerminator());

  SmallVector<Value> iterArgs = ml.getIterArgs();
  for (size_t i = 0; i < iterArgs.size(); ++i) {
    auto argsi = iterArgs[i];

    Value yieldOperand = yieldOp.getOperand(i);
    Operation *yieldDefOp = yieldOperand.getDefiningOp();
    if (!yieldDefOp) {
      LOG_DEBUG("Yield operand is a block argument, skip. Operand: "
                << yieldOperand << "\n");
      continue;
    }
    if (llvm::isa<scf::IfOp>(yieldDefOp)) {
      LOG_DEBUG("Yield operand is a if result, skip. Operand: " << yieldOperand
                                                                << "\n");
      continue;
    }
    LOG_DEBUG("yieldDefOp: " << *yieldDefOp << "\n"
                             << "idx: " << i << "\n");
    if (isDependenceOther(yieldDefOp, loopBlock, i, memGraph)) {
      continue;
    }

    int updateBlockId = bm.getBlockIdByOp(yieldDefOp);
    LOG_DEBUG("Update block id for yield def op: " << updateBlockId << "\n");

    int firstUserBlockId = findFirstUser(argsi, loopBlock, bm);
    LOG_DEBUG("First user block id: " << firstUserBlockId << "\n");

    if (firstUserBlockId == -1 || updateBlockId == firstUserBlockId) {
      continue;
    }

    IRMapping mapping;
    SmallVector<Operation *> clonedOps;

    // Clone entire depSubgraph (yieldDefOp + upstream deps) to target block
    if (!cloneDepSubgraph(yieldDefOp, loopBlock, clonedOps, mapping,
                          firstUserBlockId, bm)) {
      LOG_DEBUG("Skip iter_arg " << i << " due to non-scalar dependency\n");
      continue;
    }

    // Check if clonedOps would create a cycle
    if (willCreateCycle(clonedOps, memGraph, firstUserBlockId, bm)) {
      LOG_DEBUG("Skip iter_arg " << i << " due to cycle detected\n");
      continue;
    }

    // Update yield to use cloned yieldDefOp result
    yieldOp.setOperand(i, mapping.lookup(yieldDefOp->getResult(0)));

    // Erase original yieldDefOp and its upstream ops that have no more users
    eraseOpsWithUnusedUsers(yieldDefOp, loopBlock, bm);
    LOG_DEBUG("Successfully moved iter_arg " << i << " from block "
                                             << updateBlockId << " to block "
                                             << firstUserBlockId << "\n");
  }
}

void RefineArgsBlockIdPass::runOnOperation() {
  LOG_DEBUG("\n--- enter RefineArgsBlockIdPass --->\n");
  ModuleOp moduleOp = getOperation();

  if (CVPipeline::hasFallbackAttr(moduleOp)) {
    return;
  }

  CVPipeline::ComputeBlockIdManager bm(moduleOp);
  auto &aa = getAnalysis<AliasAnalysis>();
  LOG_DEBUG(*moduleOp << "\n");
  moduleOp.walk([&](Operation *op) {
    if (!op->hasAttr(CVPipeline::kMainLoop) ||
        !isa<scf::ForOp, scf::WhileOp>(op)) {
      return;
    }
    auto memDepGraph = CVPipeline::MemoryDependenceGraph(op, aa);
    processOneLoop(op, bm, memDepGraph);
  });

  LOG_DEBUG(*moduleOp << "\n");
  LOG_DEBUG("--- exit RefineArgsBlockIdPass --->\n");
}

namespace mlir {
namespace triton {

std::unique_ptr<OperationPass<ModuleOp>> createRefineArgsBlockIdPass() {
  return std::make_unique<RefineArgsBlockIdPass>();
}

void registerRefineArgsBlockIdPasses() {
  registerPass([]() -> std::unique_ptr<mlir::Pass> {
    return createRefineArgsBlockIdPass();
  });
}

} // namespace triton
} // namespace mlir
