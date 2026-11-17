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

#include <optional>

#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/LogicalResult.h"

#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"

#include "ascend/include/DynamicCVPipeline/Common/Utils.h"
#include "ascend/include/DynamicCVPipeline/PlanComputeBlock/ComputeBlockIdManager.h"
#include "mlir/Support/WalkResult.h"

namespace mlir {
namespace CVPipeline {

ComputeBlockIdManager::ComputeBlockIdManager(Operation *root) {
  blockIdToOps.clear();
  opToBlockId.clear();
  root->walk([&](Operation *op) {
    if (auto blockIdAttr = op->getAttrOfType<IntegerAttr>(kBlockId)) {
      auto blockId = blockIdAttr.getInt();
      if (blockId <= 0) {
        return;
      }
      opToBlockId[op] = blockId;
      blockIdToOps[blockId].push_back(op);
      cntComputeBlockId =
          std::max(cntComputeBlockId, static_cast<int>(blockId));
    }
  });
  cntComputeBlockId++; // ensure new id is unique
}

bool ComputeBlockIdManager::isWholeCubeReady(
    Operation *seedOp, llvm::DenseMap<Operation *, int> &indegree) {
  for (auto *op : getOpsInSameBlock(seedOp)) {
    if (!indegree.contains(op)) {
      continue;
    }
    if (indegree[op] != 0) {
      return false;
    }
  }
  return true;
}

bool ComputeBlockIdManager::isSameBlock(Operation *a, Operation *b) {
  if (getBlockIdByOp(a) == getBlockIdByOp(b) && getBlockIdByOp(a) != -1) {
    return true;
  }
  return false;
}

int ComputeBlockIdManager::getNextId() { return cntComputeBlockId++; }

void ComputeBlockIdManager::updateBlockIdWithInner(Operation *parentOp,
                                                   int targetId) {
  // If the parentOp's blockId is the same as every op's id in each of its
  // blocks, we need to change the ops inside its blocks to targetId as well.
  int parentBlockId = getBlockIdByOp(parentOp);
  if (parentBlockId == -1) {
    updateBlockId(parentOp, targetId);
    return;
  }
  // LinalgDialect should have targetId only in prarent, the inner op shouldn't
  // have blockId.
  if (isa<linalg::LinalgDialect>(parentOp->getDialect())) {
    updateBlockId(parentOp, targetId);
    return;
  }

  auto ret = parentOp->walk([&](Operation *op) {
    auto innerBlockId = getBlockIdByOp(op);
    if (innerBlockId != -1 && innerBlockId != parentBlockId) {
      return WalkResult::interrupt();
    }
    return WalkResult::advance();
  });

  if (ret != WalkResult::interrupt()) {
    parentOp->walk([&](Operation *op) {
      // Never fold a sync into a compute block: it must keep its own unique
      // block id so the fence between before/after ops survives.
      if (CVPipeline::isSyncOp(op)) {
        return;
      }
      updateBlockId(op, targetId);
    });
  }
  updateBlockId(parentOp, targetId);
}

void ComputeBlockIdManager::updateBlockId(Operation *op, int blockId) {
  if (!op) {
    return;
  }
  // Force Update.
  MLIRContext *ctx = op->getContext();
  if (blockId == -1) {
    op->removeAttr(kBlockId);
  } else {
    op->setAttr(kBlockId, IntegerAttr::get(IntegerType::get(ctx, kBlockIdWidth),
                                           blockId));
  }

  auto it = opToBlockId.find(op);
  if (it != opToBlockId.end()) {
    int preBlockId = it->second;
    if (preBlockId != -1) {
      auto &vec = blockIdToOps[preBlockId];
      auto vecIt = llvm::find(vec, op);
      if (vecIt != vec.end()) {
        vec.erase(vecIt);
      }
    }
  }

  opToBlockId[op] = blockId;
  blockIdToOps[blockId].push_back(op);
}

void ComputeBlockIdManager::forgetOp(Operation *op) {
  if (!op) {
    return;
  }

  auto it = opToBlockId.find(op);
  if (it == opToBlockId.end()) {
    return; // never recorded, nothing to drop
  }

  const int blockId = it->second;
  if (auto vecIt = blockIdToOps.find(blockId); vecIt != blockIdToOps.end()) {
    auto &vec = vecIt->second;
    auto found = llvm::find(vec, op);
    if (found != vec.end()) {
      vec.erase(found);
    }
  }
  opToBlockId.erase(it);
}

llvm::ArrayRef<Operation *>
ComputeBlockIdManager::getOpsRefByBlockId(int blockId) const {
  if (blockId == -1) {
    return {};
  }

  auto it = blockIdToOps.find(blockId);
  if (it == blockIdToOps.end()) {
    return {};
  }
  return it->second;
}

llvm::SmallVector<Operation *>
ComputeBlockIdManager::getOpsByBlockId(int blockId) const {
  auto ref = getOpsRefByBlockId(blockId);
  return {ref.begin(), ref.end()};
}

llvm::SmallVector<Operation *>
ComputeBlockIdManager::getOpsInSameBlock(Operation *op) const {
  auto blockIdOpt = getBlockIdByOpOpt(op);
  if (!blockIdOpt.has_value()) {
    return {op};
  }
  auto blockId = blockIdOpt.value();
  auto *block = op->getBlock();
  if (auto it = blockIdToOps.find(blockId); it != blockIdToOps.end()) {
    auto filtered =
        llvm::make_filter_range(it->second, [block](Operation *opInBlock) {
          return opInBlock->getBlock() == block;
        });
    return {filtered.begin(), filtered.end()};
  }
  return {op};
}

std::optional<int>
ComputeBlockIdManager::getBlockIdByOpOpt(Operation *op) const {
  auto it = opToBlockId.find(op);
  if (it != opToBlockId.end()) {
    return it->second;
  }
  return std::nullopt;
}

int ComputeBlockIdManager::getBlockIdByOp(Operation *op) {
  return getBlockIdByOpOpt(op).value_or(-1);
}

llvm::LogicalResult ComputeBlockIdManager::markAndRecord(Operation *op,
                                                         int blockId) {
  // When we call mark, we assume the op have no record in manager.
  MLIRContext *ctx = op->getContext();
  op->setAttr(kBlockId,
              IntegerAttr::get(IntegerType::get(ctx, kBlockIdWidth), blockId));
  auto itOld = opToBlockId.find(op);
  if (itOld != opToBlockId.end() && itOld->second != -1) {
    llvm::errs() << "Error: Operation already has a block id. Op: " << *op
                 << ", old block id: " << itOld->second
                 << ", new block id: " << blockId << "\n";
    return llvm::failure();
  }

  opToBlockId[op] = blockId;
  blockIdToOps[blockId].push_back(op);
  return llvm::success();
}

llvm::LogicalResult ComputeBlockIdManager::markOpBlockId(Operation *op) {
  int blockId = getNextId();
  return markAndRecord(op, blockId);
}

llvm::LogicalResult
ComputeBlockIdManager::markOpsWithNewId(llvm::ArrayRef<Operation *> ops) {
  if (ops.empty()) {
    return llvm::success();
  }
  int id = getNextId();
  for (Operation *op : ops) {
    if (llvm::failed(markAndRecord(op, id))) {
      return llvm::failure();
    }
  }
  return llvm::success();
}

bool ComputeBlockIdManager::shouldInheritFromParent(
    Block *block, CoreType requiredCoreType) const {
  auto *parentOp = block->getParentOp();
  if (!parentOp || !isScfOp(parentOp) ||
      getCoreTypeOfSimpleOpOrCf(parentOp) != requiredCoreType) {
    return false;
  }

  auto blockIdOpt = getBlockIdByOpOpt(parentOp);
  return blockIdOpt.has_value();
}

llvm::LogicalResult ComputeBlockIdManager::inheritFromParent(Block *block) {
  auto *parentOp = block->getParentOp();
  if (!parentOp) {
    return llvm::failure();
  }

  auto blockIdOpt = getBlockIdByOpOpt(parentOp);
  if (!blockIdOpt.has_value()) {
    return llvm::failure();
  }

  // no need to and should not walk inside nested blocks:
  // 1. the caller is from walk already
  // 2. if we have marked nested ops with this block id, it could be correct,
  // since they will be re-marked with the same id, so no failures will be
  // returned, but this is less robust
  auto blockId = blockIdOpt.value();
  for (auto &op : *block) {
    if (llvm::failed(markAndRecord(&op, blockId))) {
      return llvm::failure();
    }
  }
  return llvm::success();
}

} // namespace CVPipeline
} // namespace mlir
