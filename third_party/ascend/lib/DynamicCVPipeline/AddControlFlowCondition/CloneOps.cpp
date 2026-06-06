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

#include "ascend/include/DynamicCVPipeline/AddControlFlowCondition/CloneOps.h"
#include "ascend/include/DynamicCVPipeline/AddControlFlowCondition/Utils.h"
#include "bishengir/Dialect/HIVM/IR/HIVM.h"
#include "bishengir/Dialect/Scope/IR/Scope.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/Debug.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/IRMapping.h"

static constexpr const char *DEBUG_TYPE = "CloneOps";
#define DBGS() (llvm::dbgs() << '[' << DEBUG_TYPE << "] ")
#define LDBG(...) \
LLVM_DEBUG({ \
  DBGS(); \
  llvm::outs() << __VA_ARGS__; \
  llvm::outs() << "\n"; \
})

using namespace mlir;
using namespace triton;
using namespace hivm;

// Update op operands using value mapping, skip yield values of forOp
static LogicalResult updateCloneMapping(Operation *op,
    llvm::DenseMap<Value, Value> &valueMap,
    const llvm::DenseSet<Value> &yieldValues)
{
  if (!op) {
    return failure();
  }

  for (OpOperand &operand : op->getOpOperands()) {
    // Only skip if this operand is a yield value from the main_loop forOp
    // Nested yield ops (in scf.if/scf.for) should have their operands updated
    Value v = operand.get();
    if (yieldValues.contains(v)) {
      continue;
    }

    auto it = valueMap.find(v);
    if (it != valueMap.end()) {
      if (it->second.getType() != v.getType()) {
        LDBG("[Error]: type mismatch in value mapping: " << v.getType() << " vs " << it->second.getType() << "\n");
        return failure();
      }
      operand.set(it->second);
    }
  }

  for (Region &region : op->getRegions()) {
    for (Block &block : region) {
      for (Operation &nestedOp : block) {
        if (failed(updateCloneMapping(&nestedOp, valueMap, yieldValues))) {
          return failure();
        }
      }
    }
  }

  return success();
}

// Clone a single op with IRMapping
static Operation *cloneOpWithMapping(Operation *op,
    OpBuilder &builder,
    llvm::DenseMap<Value, Value> &valueMap)
{
  IRMapping mapper;
  for (auto result : op->getResults()) {
    if (valueMap.count(result)) {
      mapper.map(result, valueMap[result]);
    }
  }

  Operation *cloned = builder.clone(*op, mapper);
  for (auto it : llvm::zip(op->getResults(), cloned->getResults())) {
    valueMap[std::get<0>(it)] = std::get<1>(it);
  }

  return cloned;
}

// Clone ops for a single block in vector/cube mode
static LogicalResult cloneOpsForBlock(int curId, SmallVector<Operation *> &curOps,
    const SmallVector<int> &earlierIds,
    const llvm::DenseMap<int, SmallVector<Operation *>> &blockOps,
    scf::ForOp forOp)
{
  if (curOps.empty() || earlierIds.empty()) {
    return success();
  }

  // Collect all ops from earlier blocks to clone
  SmallVector<Operation *> toClone;
  for (int eid : earlierIds) {
    llvm::append_range(toClone, blockOps.lookup(eid));
  }
  if (toClone.empty()) {
    return success();
  }

  llvm::DenseMap<Value, Value> valueMap;
  SmallVector<Operation *> clonedOps;
  OpBuilder builder(curOps.front());

  for (Operation *op : toClone) {
    Operation *cloned = cloneOpWithMapping(op, builder, valueMap);
    cloned->setAttr("ssbuffer.block_id", builder.getI32IntegerAttr(curId));
    if (auto origBlockIdAttr = op->getAttrOfType<IntegerAttr>("ssbuffer.block_id")) {
      cloned->setAttr("ssbuffer.clone", origBlockIdAttr);
    }
    clonedOps.push_back(cloned);
  }

  curOps.insert(curOps.begin(), clonedOps.begin(), clonedOps.end());

  llvm::DenseSet<Value> yieldValues;
  if (auto yieldOp = dyn_cast<scf::YieldOp>(forOp.getBody()->getTerminator())) {
    for (Value operand : yieldOp.getOperands()) {
      yieldValues.insert(operand);
    }
  }

  for (Operation *op : curOps) {
    if (failed(updateCloneMapping(op, valueMap, yieldValues))) {
      return failure();
    }
  }

  if (failed(topologicalSort(curOps))) {
    return failure();
  }

  return success();
}

// Clone ops for all blocks in main loop
LogicalResult CloneOpsPass::cloneOpsInMainLoop(scf::ForOp forOp)
{
  llvm::DenseMap<int, SmallVector<Operation *>> blockOps;
  if (failed(collectOpsByBlockId(forOp, blockOps))) {
    return failure();
  }

  SmallVector<int> idsInOrder = getBlockIdsInOrder(forOp);

  for (int i = idsInOrder.size() - 1; i >= 0; --i) {
    int curId = idsInOrder[i];
    SmallVector<int> earlierIds(idsInOrder.begin(), idsInOrder.begin() + i);

    if (failed(cloneOpsForBlock(curId, blockOps[curId], earlierIds, blockOps, forOp))) {
      return failure();
    }
  }

  return success();
}

// Check if an op should be erased during cleanup (for cube)
static bool shouldEraseOpForCube(Operation *op)
{
  if (auto ifOp = dyn_cast<scf::IfOp>(op)) {
    if (ifOp->hasAttr("ssbuffer.cross_buffer") || ifOp->hasAttr("ssbuffer.intra_buffer") ||
        ifOp->hasAttr("ssbuffer.load_store")) {
      return true;
    }
  }

  if (isa<SyncBlockWaitOp>(op) || isa<SyncBlockSetOp>(op) ||
      isa<hivm::FixpipeOp>(op)) {
    return true;
  }

  // Copy and fill ops: erase if operand not used elsewhere
  if (isa<memref::CopyOp>(op) || (isa<linalg::FillOp>(op) && op->getNumResults() == 0)) {
    if (op->getNumOperands() >  1) {
      Value secondOperand = op->getOperand(1);
      bool usedByOtherOp = llvm::any_of(secondOperand.getUsers(), [&](Operation *user) {
        return user != op;
      });
      if (!usedByOtherOp) {
        return true;
      }
    }
  }

  // Erase ops with no used results
  return llvm::none_of(op->getResults(), [](auto result) { return !result.use_empty(); });
}

// Check if an op should be erased (for vector)
static bool shouldEraseOpForVector(Operation *op)
{
  if (auto ifOp = dyn_cast<scf::IfOp>(op)) {
    if (ifOp->hasAttr("ssbuffer.cross_buffer") || ifOp->hasAttr("ssbuffer.intra_buffer") ||
        ifOp->hasAttr("ssbuffer.load_store")) {
      return true;
    }
  }

  return llvm::none_of(op->getResults(), [](auto result) { return !result.use_empty(); });
}

// Cleanup for cloned ops in a forOp
static LogicalResult cleanupClonedOps(scf::ForOp forOp,
    llvm::DenseMap<int, SmallVector<Operation *>> &blockOps,
    const SmallVector<int> &idsInOrder,
    bool isCube)
{
  for (int i = idsInOrder.size() - 1; i >= 0; --i) {
    auto &curOps = blockOps[idsInOrder[i]];
    if (curOps.empty()) {
      continue;
    }

    // Find last index of cloned ops
    int startIdx = -1;
    for (int j = curOps.size() - 1; j >= 0; --j) {
      if (curOps[j]->hasAttr("ssbuffer.clone")) {
        startIdx = j;
        break;
      }
    }
    if (startIdx < 0) {
      continue;
    }

    // Erase cloned ops that are not needed
    for (int j = startIdx; j >= 0; --j) {
      Operation *op = curOps[j];
      if (!op->hasAttr("ssbuffer.clone")) {
        break;
      }
      bool shouldErase = isCube ? shouldEraseOpForCube(op) : shouldEraseOpForVector(op);
      if (shouldErase) {
        op->erase();
      }
    }
  }

  // Check whether new forOp is valid after cleanup
  for (Operation &op : forOp.getBody()->without_terminator()) {
    if (op.hasAttr("ssbuffer.clone")) {
      if (isa<SyncBlockWaitOp>(op) || isa<SyncBlockSetOp>(op) ||
          isa<hivm::FixpipeOp>(op)) {
        LDBG("[ERROR]: Cloned sync/fixpipe op should have been erased: " << op.getName() << "\n");
        return failure();
      }
    }
  }

  return success();
}

// Cleanup cloned ops for a single main loop
LogicalResult CloneOpsPass::cleanupClonedOpsInMainLoop(scf::ForOp forOp)
{
  ModuleOp module = getOperation();
  scope::ScopeOp scopeOp = forOp->getParentOfType<scope::ScopeOp>();
  if (!scopeOp) {
    return success();
  }

  auto attr = scopeOp->getAttrOfType<hivm::TCoreTypeAttr>("hivm.tcore_type");
  if (!attr) {
    return success();
  }

  bool isCube = (attr == hivm::TCoreTypeAttr::get(module.getContext(), hivm::TCoreType::CUBE));

  llvm::DenseMap<int, SmallVector<Operation *>> blockOps;
  if (failed(collectOpsByBlockId(forOp, blockOps))) {
    return failure();
  }

  SmallVector<int> idsInOrder = getBlockIdsInOrder(forOp);
  if (failed(cleanupClonedOps(forOp, blockOps, idsInOrder, isCube))) {
    return failure();
  }

  return success();
}

// Validate that each block_id's ops form contiguous ranges (not interleaved with other ids)
// e.g., [1,1,2,2] is valid, but [1,2,1,2] is invalid
static bool areBlockIdsConsecutive(scf::ForOp forOp)
{
  SmallVector<int> idsInOrder;
  for (Operation &op : forOp.getBody()->without_terminator()) {
    auto blockIdAttr = op.getAttrOfType<IntegerAttr>("ssbuffer.block_id");
    if (!blockIdAttr) {
      LDBG("[ERROR]: Op missing ssbuffer.block_id: " << op.getName() << "\n");
      return false;
    }

    idsInOrder.push_back(blockIdAttr.getInt());
  }

  // Check that each block_id forms a contiguous range
  for (size_t i = 0; i < idsInOrder.size();) {
    int currentId = idsInOrder[i];
    size_t j = i;

    while (j < idsInOrder.size() && idsInOrder[j] == currentId) {
      ++j;
    }

    for (size_t k = j; k < idsInOrder.size(); ++k) {
      if (idsInOrder[k] == currentId) {
        LDBG("[ERROR]: block_id: " << currentId << " is interleaved\n");
        return false;
      }
    }

    i = j;
  }

  return true;
}

LogicalResult CloneOpsPass::validateBlockIdsConsecutive(ModuleOp module)
{
  WalkResult result = module.walk([&](Operation *op) -> WalkResult {
    if (!op->hasAttr("ssbuffer.main_loop")) {
      return WalkResult::advance();
    }
    auto forOp = dyn_cast<scf::ForOp>(op);
    if (!forOp) {
      LDBG("[Error]: op with ssbuffer.main_loop is not a scf::ForOp\n");
      return WalkResult::interrupt();
    }
    if (!areBlockIdsConsecutive(forOp)) {
      return WalkResult::interrupt();
    }
    return WalkResult::advance();
  });
  if (result.wasInterrupted())
    return failure();

  return success();
}

void CloneOpsPass::runOnOperation()
{
  ModuleOp module = getOperation();

  LDBG("before cloneOps:\n" << module << "\n");

  // Validate block_ids are consecutive before cloning
  if (failed(validateBlockIdsConsecutive(module)))
    return;

  // Clone ops in vector/cube to ensure that each block_id has its own
  // ops without sharing
  module.walk([&](Operation *op) -> WalkResult {
    if (!op->hasAttr("ssbuffer.main_loop")) {
      return WalkResult::advance();
    }
    auto forOp = dyn_cast<scf::ForOp>(op);
    if (!forOp) {
      LDBG("[Error]: op with ssbuffer.main_loop is not a scf::ForOp\n");
      signalPassFailure();
      return WalkResult::interrupt();
    }

    if (failed(cloneOpsInMainLoop(forOp))) {
      signalPassFailure();
      return WalkResult::interrupt();
    }

    if (failed(cleanupClonedOpsInMainLoop(forOp))) {
      signalPassFailure();
      return WalkResult::interrupt();
    }
    return WalkResult::advance();
  });

  LDBG("after cloneOps:\n" << module << "\n");
}

namespace mlir {
namespace triton {

std::unique_ptr<OperationPass<ModuleOp>> createCloneOpsPass()
{
  return std::make_unique<CloneOpsPass>();
}

} // namespace triton
} // namespace mlir
