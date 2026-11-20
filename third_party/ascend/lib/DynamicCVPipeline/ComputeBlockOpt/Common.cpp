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

#include "ascend/include/DynamicCVPipeline/ComputeBlockOpt/Common.h"
#include "DynamicCVPipeline/Common/CycleDetector.h"
#include "DynamicCVPipeline/Common/DependencyHelper.h"
#include "ascend/include/DynamicCVPipeline/Common/SyncWall.h"
#include "ascend/include/DynamicCVPipeline/Common/Utils.h"
#include "ascend/include/DynamicCVPipeline/PlanComputeBlock/Common.h"
#include "mlir/Analysis/TopologicalSortUtils.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/Operation.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "mlir/Interfaces/ViewLikeInterface.h"
#include "triton/Analysis/Utility.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Debug.h"

static constexpr const char *DEBUG_TYPE = "compute-block-opt-common";
#define LOG_DEBUG(...)                                                         \
  LLVM_DEBUG(llvm::dbgs() << " [" << DEBUG_TYPE << "] " << __VA_ARGS__ << "\n")

using namespace mlir;

namespace mlir {
namespace CVPipeline {

static bool groupWouldStraddleSync(llvm::ArrayRef<Operation *> opsToUnify,
                                   int targetBlockId,
                                   ComputeBlockIdManager &bm) {
  llvm::DenseMap<Block *, llvm::SmallVector<Operation *>> opsByBlock;
  for (auto *op : opsToUnify) {
    if (op && op->getBlock()) {
      opsByBlock[op->getBlock()].push_back(op);
    }
  }
  for (auto *op : bm.getOpsByBlockId(targetBlockId)) {
    if (op && op->getBlock()) {
      opsByBlock[op->getBlock()].push_back(op);
    }
  }
  for (auto &blockOps : opsByBlock) {
    Block *blk = blockOps.first;
    SyncWall wall(blk);
    unsigned refSeg = 0;
    bool refSet = false;
    for (auto *op : blockOps.second) {
      if (getAncestorInBlock(op, blk) == nullptr) {
        continue;
      }
      unsigned seg = wall.segmentOf(op);
      if (!refSet) {
        refSeg = seg;
        refSet = true;
        continue;
      }
      if (seg != refSeg) {
        LOG_DEBUG("Reject merge: group would straddle a sync in block "
                  << blk << "\n");
        return true;
      }
    }
  }
  return false;
}

bool willCreateCycle(llvm::ArrayRef<Operation *> opsToUnify,
                     const MemoryDependenceGraph &memGraph, int targetBlockId,
                     ComputeBlockIdManager &bm) {
  if (opsToUnify.empty()) {
    return false;
  }

  auto *block = opsToUnify.front()->getBlock();

  // A block_id group must never straddle a synchronization op
  if (groupWouldStraddleSync(opsToUnify, targetBlockId, bm)) {
    return true;
  }

  llvm::DenseSet<Operation *> okSet;
  for (auto *op : bm.getOpsByBlockId(targetBlockId)) {
    okSet.insert(op);
  }
  for (auto *op : opsToUnify) {
    okSet.insert(op);
  }

  DenseMap<Operation *, int> origBlockIdMap;
  for (auto *op : opsToUnify) {
    auto optBlockId = getOpBlockId(op);
    origBlockIdMap[op] = optBlockId ? *optBlockId : -1;
    bm.updateBlockId(op, targetBlockId);
  }

  // Initialize DFS detector
  DependencyHelper depHelper{memGraph};
  DependencyCycleDetector dfs(block, depHelper, okSet, bm);
  auto hasCycle = dfs.detectCycle();

  for (auto &[op, origBlockId] : origBlockIdMap) {
    bm.updateBlockId(op, origBlockId);
  }

  return hasCycle;
}

void cloneScalarOpsForCrossBlockUses(ComputeBlockIdManager &bm,
                                     SetVector<Operation *> &matchedOps,
                                     int targetBlockId) {

  // This means move matchedOps into targetBlockId
  auto sorted = mlir::topologicalSort(matchedOps);
  for (Operation *op : llvm::reverse(sorted)) {
    if (op->getNumResults() == 1 &&
        CVPipeline::isScalarLike(op->getResult(0))) {
      // replace op not in matchedOps with cloned op, and keep original op for
      // other pattern.
      SmallVector<OpOperand *> otherUses;
      for (auto &use : op->getResult(0).getUses()) {
        Operation *userOp = use.getOwner();
        auto userInBlock =
            CVPipeline::getAncestorInBlock(userOp, op->getBlock());
        if (!userInBlock)
          continue;
        if (llvm::find(matchedOps, userInBlock) == matchedOps.end() &&
            bm.getBlockIdByOp(userInBlock) != targetBlockId) {
          otherUses.push_back(&use);
        }
      }
      if (otherUses.size() > 0) {
        LOG_DEBUG("now cloned: " << *op);
        OpBuilder builder(op);
        auto clonedOp = builder.clone(*op);
        bm.updateBlockId(clonedOp, bm.getBlockIdByOp(op));
        for (auto use : otherUses) {
          (*use).set(clonedOp->getResult(0));
        }
      }
    }
  }
}

bool collectViewOpsAndCheckGlobalMemory(Value viewValue,
                                        SetVector<Operation *> &matchedOps) {
  // Subview ops may be nested many layers deep through reinterpretation or
  // other subviews. like, subview (subview (reinterpret_cast (subview
  // (reinterpret_cast (arg0))))) so we need Search and only keep same block
  // view-like op.
  auto isFuncArg = [&](Value v) {
    if (auto blockArg = dyn_cast<BlockArgument>(v)) {
      Operation *parentOp = blockArg.getOwner()->getParentOp();
      if (isa<func::FuncOp>(parentOp)) {
        return true;
      } else {
        LOG_DEBUG(
            "Subview source block argument is not from func entry block.");
        return false;
      }
    }
    return false;
  };
  if (isFuncArg(viewValue)) {
    return true;
  }

  auto viewOp = viewValue.getDefiningOp<ViewLikeOpInterface>();
  if (!viewOp) {
    return false;
  }
  if (!viewOp->hasOneUse()) {
    return false;
  }
  auto block = viewOp->getBlock();
  LOG_DEBUG("Check view source: " << viewValue);
  while (true) {

    if (isFuncArg(viewValue)) {
      return true;
    }
    if (!viewValue.getDefiningOp()) {
      return false;
    }
    // From other view-like op
    if (auto viewLike =
            dyn_cast<ViewLikeOpInterface>(viewValue.getDefiningOp())) {
      if (viewLike->getBlock() == block) {
        matchedOps.insert(viewLike.getOperation());
      }
      if (!viewLike->hasOneUse()) {
        return false;
      }
      viewValue = viewLike.getViewSource();
      continue;
    }
    LOG_DEBUG(
        "Subview source defining op is not ViewLikeOpInterface: " << viewValue);
    return false;
  }
  return false;
}

} // namespace CVPipeline
} // namespace mlir
