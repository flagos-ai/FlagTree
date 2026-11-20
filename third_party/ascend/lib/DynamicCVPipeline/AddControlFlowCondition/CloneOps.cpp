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
#include "ascend/include/DynamicCVPipeline/Common/MemoryEffectsTracker.h"
#include "ascend/include/DynamicCVPipeline/Common/Utils.h"
#include "bishengir/Dialect/HIVM/IR/HIVM.h"
#include "bishengir/Dialect/Scope/IR/Scope.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/IRMapping.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/Debug.h"

static constexpr const char *DEBUG_TYPE = "CloneOps";
#define DBGS() (llvm::dbgs() << '[' << DEBUG_TYPE << "] ")
#define LDBG(...)                                                              \
  LLVM_DEBUG({                                                                 \
    DBGS();                                                                    \
    llvm::dbgs() << __VA_ARGS__;                                               \
    llvm::dbgs() << "\n";                                                      \
  })

using namespace mlir;
using namespace triton;
using namespace CVPipeline;
using namespace hivm;

using MemDepGraph = std::unique_ptr<CVPipeline::MemoryDependenceGraph>;
using MemDepGraphT = CVPipeline::MemoryDependenceGraph;

// Updates op operands via value mapping; skips main-loop body yield values
// (yieldValues collected once per main-loop from the body's terminator).
static LogicalResult
updateCloneMapping(Operation *op, llvm::DenseMap<Value, Value> &valueMap,
                   const llvm::DenseSet<Value> &yieldValues) {
  if (!op) {
    return failure();
  }

  for (OpOperand &operand : op->getOpOperands()) {
    // Only skip if this operand is a yield value from the main_loop body.
    // Nested yield ops (in scf.if/scf.for) should have their operands updated.
    Value v = operand.get();
    if (yieldValues.contains(v)) {
      continue;
    }

    auto it = valueMap.find(v);
    if (it != valueMap.end()) {
      if (it->second.getType() != v.getType()) {
        LDBG("[Error]: type mismatch in value mapping: "
             << v.getType() << " vs " << it->second.getType());
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
static Operation *cloneOpWithMapping(Operation *op, OpBuilder &builder,
                                     llvm::DenseMap<Value, Value> &valueMap) {
  IRMapping mapper;
  // Populate mapper with ALL previously cloned values (not just the current
  // op's results).
  for (const auto &entry : valueMap) {
    mapper.map(entry.first, entry.second);
  }

  Operation *cloned = builder.clone(*op, mapper);
  for (auto it : llvm::zip(op->getResults(), cloned->getResults())) {
    valueMap[std::get<0>(it)] = std::get<1>(it);
  }

  return cloned;
}

// Clones ops for a single block in vector/cube mode. `bodyBlock` (forOp body
// or whileOp after-region) only supplies iter-arg yields not to be remapped.
static LogicalResult
cloneOpsForBlock(int curId, SmallVector<Operation *> &curOps,
                 const SmallVector<int> &earlierIds,
                 const llvm::DenseMap<int, SmallVector<Operation *>> &blockOps,
                 Block *bodyBlock) {
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
    cloned->setAttr(CVPipeline::kBlockId, builder.getI32IntegerAttr(curId));
    if (auto origBlockIdOpt = CVPipeline::getOpBlockId(op)) {
      cloned->setAttr(
          CVPipeline::kClone,
          builder.getI32IntegerAttr(static_cast<int32_t>(*origBlockIdOpt)));
    }
    clonedOps.push_back(cloned);
  }

  curOps.insert(curOps.begin(), clonedOps.begin(), clonedOps.end());

  llvm::DenseSet<Value> yieldValues;
  if (bodyBlock) {
    if (auto yieldOp = dyn_cast<scf::YieldOp>(bodyBlock->getTerminator())) {
      for (Value operand : yieldOp.getOperands()) {
        yieldValues.insert(operand);
      }
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

// Clones ops for all blocks in the main loop op (scf.for or scf.while).
LogicalResult CloneOpsPass::cloneOpsInMainLoop(Operation *op) {
  Block *bodyBlock = MainLoop(op).getBody();
  if (!bodyBlock) {
    LDBG("[Error]: op with ssbuffer.main_loop is not a scf::ForOp or "
         "scf::WhileOp");
    return failure();
  }

  llvm::DenseMap<int, SmallVector<Operation *>> blockOps;
  if (failed(collectOpsByBlockId(op, blockOps))) {
    return failure();
  }
  SmallVector<int> idsInOrder = getBlockIdsInOrder(op);

  for (int i = idsInOrder.size() - 1; i >= 0; --i) {
    int curId = idsInOrder[i];
    SmallVector<int> earlierIds(idsInOrder.begin(), idsInOrder.begin() + i);

    if (failed(cloneOpsForBlock(curId, blockOps[curId], earlierIds, blockOps,
                                bodyBlock))) {
      return failure();
    }
  }

  return success();
}

// Should an op be erased during cube cleanup? sameBlockIdExecAfter: per-block
// same-block-id exec-after ops (sync filtered); erasedOps no longer pin ops.
static bool shouldEraseOpForCube(
    Operation *op,
    const llvm::DenseMap<Operation *, llvm::SmallPtrSet<Operation *, 4>>
        &sameBlockIdExecAfter,
    const llvm::DenseSet<Operation *> &erasedOps) {
  // Rule 1: SyncBlockWaitOp, SyncBlockSetOp, FixpipeOp -> directly erase,
  // including when inter-core multi-buffering has already wrapped them in an
  // scf.if picking the flag for this iteration's buffer half. Only scf.ifs
  // that yield nothing qualify: the wrapper around a transfer op yields its
  // result and is data, not a handshake.
  if (isa<SyncBlockWaitOp>(op) || isa<SyncBlockSetOp>(op) ||
      isa<hivm::FixpipeOp>(op) ||
      (op->getNumResults() == 0 && isIfOpWithOnlySyncOps(op))) {
    return true;
  }

  auto opBlockId = getLoopDirectChildBlockId(op);

  // Rule 2: if op has results, check via SSA whether any result is used by a
  // later op with the same loop-direct-child block_id (main-loop's own child).
  if (op->getNumResults() > 0) {
    for (auto result : op->getResults()) {
      if (result.use_empty()) {
        // Result not used by anyone, can erase
        continue;
      }
      if (!opBlockId) {
        // No block_id but result is used, be conservative and keep
        return false;
      }
      // Check if any user is in the same loop-direct-child block_id
      bool usedInSameBlockId =
          llvm::any_of(result.getUsers(), [&](Operation *user) {
            auto userBlockId = getLoopDirectChildBlockId(user);
            return userBlockId && *userBlockId == *opBlockId;
          });
      if (usedInSameBlockId) {
        // Result used in same loop-direct-child block, cannot erase
        return false;
      }
    }
    // All results are either unused or not used in same loop-direct-child
    // block, can erase
    return true;
  }

  // Rule 3: no results -> consult precomputed same-block-id exec-after set; ops
  // already erased in this pass are not live (mirrors per-iteration rebuild).
  auto it = sameBlockIdExecAfter.find(op);
  if (it != sameBlockIdExecAfter.end()) {
    for (Operation *execOp : it->second) {
      // Already erased in this pass → no longer pins the current op.
      if (erasedOps.contains(execOp)) {
        continue;
      }
      // scf.if bodies holding only sync_block_wait/set are cleaned up in their
      // own turn; skip so they don't block this op's erasure.
      if (isIfOpWithOnlySyncOps(execOp)) {
        continue;
      }
      auto execBlockId = CVPipeline::getOpBlockId(execOp);
      if (execBlockId && opBlockId && *execBlockId == *opBlockId) {
        return false;
      }
    }
  }

  // Can erase if no results and no live exec-after dependencies in same block
  return true;
}

// Check if an op should be erased (for vector)
static bool shouldEraseOpForVector(Operation *op) {
  return llvm::none_of(op->getResults(),
                       [](auto result) { return !result.use_empty(); });
}

// Verify cloned sync/fixpipe ops were erased after cleanup; nullptr bodyBlock
// means nothing to check.
static LogicalResult validateClonedSyncOpsErased(Block *bodyBlock) {
  if (!bodyBlock) {
    return success();
  }
  for (Operation &op : bodyBlock->without_terminator()) {
    if (!op.hasAttr(CVPipeline::kClone)) {
      continue;
    }
    if (isa<SyncBlockWaitOp>(&op) || isa<SyncBlockSetOp>(&op) ||
        isa<hivm::FixpipeOp>(&op)) {
      LDBG("[ERROR]: Cloned sync/fixpipe op should have been erased: "
           << op.getName());
      return failure();
    }
    // Look inside as well
    WalkResult nested = op.walk([](Operation *inner) {
      return isa<SyncBlockWaitOp>(inner) || isa<SyncBlockSetOp>(inner)
                 ? WalkResult::interrupt()
                 : WalkResult::advance();
    });
    if (nested.wasInterrupted()) {
      LDBG("[ERROR]: Cloned op still holds a sync op that should have been "
           "erased: "
           << op.getName());
      return failure();
    }
  }

  return success();
}

// Cleans up cloned ops in a main-loop op. `bodyBlock` is only scanned after
// cleanup; `mainLoopOp` goes to `memGraphFactory` to rebuild the mem graph.
static LogicalResult
cleanupClonedOps(Operation *mainLoopOp, Block *bodyBlock,
                 llvm::DenseMap<int, SmallVector<Operation *>> &blockOps,
                 const SmallVector<int> &idsInOrder, bool isCube,
                 std::function<MemDepGraph(Operation *)> memGraphFactory) {
  for (int i = idsInOrder.size() - 1; i >= 0; --i) {
    auto &curOps = blockOps[idsInOrder[i]];
    if (curOps.empty()) {
      continue;
    }

    // Find last index of cloned ops
    int startIdx = -1;
    for (int j = curOps.size() - 1; j >= 0; --j) {
      if (curOps[j]->hasAttr(CVPipeline::kClone)) {
        startIdx = j;
        break;
      }
    }
    if (startIdx < 0) {
      continue;
    }

    // Locate the start of the contiguous cloned-op suffix; the original
    // cleanup broke at the first non-cloned op, so keep that same set.
    int firstClonedIdx = 0;
    for (int j = startIdx - 1; j >= 0; --j) {
      if (!curOps[j]->hasAttr(CVPipeline::kClone)) {
        firstClonedIdx = j + 1;
        break;
      }
    }

    // Collect the cloned ops in the contiguous suffix (in execution order).
    SmallVector<Operation *> clonedOps;
    clonedOps.reserve(startIdx - firstClonedIdx + 1);
    for (int j = firstClonedIdx; j <= startIdx; ++j) {
      clonedOps.push_back(curOps[j]);
    }

    // Build the MemoryDependenceGraph at most once per block, only if a cloned
    // op has no results (only Rule 3 needs it; Rule 2 SSA, Rule 1 type-based).
    MemDepGraph memGraph;
    if (isCube) {
      bool needsMemGraph = llvm::any_of(
          clonedOps, [](Operation *o) { return o->getNumResults() == 0; });
      if (needsMemGraph) {
        memGraph = memGraphFactory(mainLoopOp);
      }
    }

    // Precompute each cloned op's exec-after ops sharing its block_id once and
    // reuse it; erased ops are filtered at check time (as if rebuilt).
    llvm::DenseMap<Operation *, llvm::SmallPtrSet<Operation *, 4>>
        sameBlockIdExecAfter;
    if (memGraph) {
      for (Operation *op : clonedOps) {
        if (op->getNumResults() > 0) {
          // Rule 2 path: memgraph is not consulted.
          continue;
        }
        auto opBlockId = getLoopDirectChildBlockId(op);
        if (!opBlockId) {
          continue;
        }
        for (Operation *execOp : memGraph->getExecAfter(op)) {
          // sync_block_wait/sync_block_set are not memory side effects for
          // cleanup analysis, so skip them.
          if (isa<SyncBlockWaitOp>(execOp) || isa<SyncBlockSetOp>(execOp)) {
            continue;
          }
          auto execBlockId = CVPipeline::getOpBlockId(execOp);
          if (execBlockId && *execBlockId == *opBlockId) {
            sameBlockIdExecAfter[op].insert(execOp);
          }
        }
      }
    }

    // Erase cloned ops bottom-to-top (original ordering); erasedOps makes an
    // already-erased op count as absent from the precomputed exec-after sets.
    llvm::DenseSet<Operation *> erasedOps;
    for (int j = startIdx; j >= firstClonedIdx; --j) {
      Operation *op = curOps[j];
      bool shouldErase =
          isCube ? shouldEraseOpForCube(op, sameBlockIdExecAfter, erasedOps)
                 : shouldEraseOpForVector(op);
      if (shouldErase) {
        op->erase();
        erasedOps.insert(op);
      }
    }
  }

  return validateClonedSyncOpsErased(bodyBlock);
}

// Cleans up cloned ops for one main loop (scf.for or scf.while); body block
// comes from MainLoop::getBody, the op feeds collect/order helpers+factory.
LogicalResult CloneOpsPass::cleanupClonedOpsInMainLoop(Operation *op) {
  Block *bodyBlock = MainLoop(op).getBody();
  if (!bodyBlock) {
    LDBG("[Error]: op with ssbuffer.main_loop is not a scf::ForOp or "
         "scf::WhileOp");
    return failure();
  }

  ModuleOp module = getOperation();
  scope::ScopeOp scopeOp = op->getParentOfType<scope::ScopeOp>();
  if (!scopeOp) {
    return success();
  }

  auto attr =
      scopeOp->getAttrOfType<hivm::TCoreTypeAttr>(CVPipeline::kTcoreType);
  if (!attr) {
    return success();
  }

  bool isCube = (attr == hivm::TCoreTypeAttr::get(module.getContext(),
                                                  hivm::TCoreType::CUBE));

  llvm::DenseMap<int, SmallVector<Operation *>> blockOps;
  if (failed(collectOpsByBlockId(op, blockOps))) {
    return failure();
  }
  SmallVector<int> idsInOrder = getBlockIdsInOrder(op);

  if (failed(cleanupClonedOps(op, bodyBlock, blockOps, idsInOrder, isCube,
                              [&](Operation *loopOp) -> MemDepGraph {
                                if (!isCube) {
                                  return nullptr;
                                }
                                auto &aliasAnalysis =
                                    getAnalysis<mlir::AliasAnalysis>();
                                return std::make_unique<MemDepGraphT>(
                                    loopOp, aliasAnalysis);
                              }))) {
    return failure();
  }

  return success();
}

// Validate each block_id's ops form contiguous ranges, not interleaved (e.g.
// [1,1,2,2] valid, [1,2,1,2] invalid), on a main-loop body block.
static bool areBlockIdsConsecutive(Block *bodyBlock) {
  SmallVector<int> idsInOrder;
  for (Operation &op : bodyBlock->without_terminator()) {
    auto blockIdOpt = CVPipeline::getOpBlockId(&op);
    if (!blockIdOpt) {
      LDBG("[ERROR]: Op missing ssbuffer.block_id: " << op.getName());
      return false;
    }

    idsInOrder.push_back(static_cast<int>(*blockIdOpt));
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
        LDBG("[ERROR]: block_id: " << currentId << " is interleaved");
        return false;
      }
    }

    i = j;
  }

  return true;
}

LogicalResult CloneOpsPass::validateBlockIdsConsecutive(ModuleOp module) {
  WalkResult result = module.walk([&](Operation *op) -> WalkResult {
    if (!isMainLoopOp(op)) {
      return WalkResult::advance();
    }
    Block *bodyBlock = MainLoop(op).getBody();
    if (!bodyBlock) {
      LDBG("[Error]: op with ssbuffer.main_loop is not a scf::ForOp or "
           "scf::WhileOp");
      return WalkResult::interrupt();
    }
    if (!areBlockIdsConsecutive(bodyBlock)) {
      return WalkResult::interrupt();
    }
    return WalkResult::advance();
  });
  if (result.wasInterrupted())
    return failure();

  return success();
}

// Checks that no op in a VECTOR scope's main_loop op (scf.for or scf.while)
// has a tensor result carrying ssbuffer.clone.
LogicalResult CloneOpsPass::validateClonedOpsInVector(ModuleOp module) {
  WalkResult result = module.walk([&](Operation *op) -> WalkResult {
    if (!isMainLoopOp(op)) {
      return WalkResult::advance();
    }
    Block *bodyBlock = MainLoop(op).getBody();
    if (!bodyBlock) {
      LDBG("[Error]: op with ssbuffer.main_loop is not a scf::ForOp or "
           "scf::WhileOp");
      return WalkResult::interrupt();
    }

    scope::ScopeOp scopeOp = op->getParentOfType<scope::ScopeOp>();
    if (!scopeOp) {
      return WalkResult::advance();
    }

    auto attr =
        scopeOp->getAttrOfType<hivm::TCoreTypeAttr>(CVPipeline::kTcoreType);
    if (!attr) {
      return WalkResult::advance();
    }

    if (attr != hivm::TCoreTypeAttr::get(module.getContext(),
                                         hivm::TCoreType::VECTOR)) {
      return WalkResult::advance();
    }

    for (Operation &bodyOp : bodyBlock->without_terminator()) {
      if (!bodyOp.hasAttr(CVPipeline::kClone)) {
        continue;
      }
      if (isa<tensor::EmptyOp>(&bodyOp)) {
        continue;
      }
      bool hasTensorDep = llvm::any_of(bodyOp.getResults(), [](Value result) {
        return isa<RankedTensorType>(result.getType());
      });
      if (hasTensorDep) {
        LDBG("[Error]: VECTOR main_loop contains cloned op with tensor type: "
             << bodyOp.getName());
        return WalkResult::interrupt();
      }
    }

    return WalkResult::advance();
  });

  if (result.wasInterrupted())
    return failure();

  return success();
}

void CloneOpsPass::runOnOperation() {
  ModuleOp module = getOperation();

  if (CVPipeline::hasFallbackAttr(module)) {
    return;
  }

  LDBG("before cloneOps:\n" << module);

  // Validate block_ids are consecutive before cloning
  if (failed(validateBlockIdsConsecutive(module))) {
    CVPipeline::setFallbackAttr(module, CVPipeline::ERRCODE_FAILED);
    return;
  }

  // Clone ops in vector/cube so each block_id owns its ops (no sharing); entry
  // points take any main-loop op (scf.for/scf.while) and dispatch internally.
  auto walkResult = module.walk([&](Operation *op) -> WalkResult {
    if (!isMainLoopOp(op)) {
      return WalkResult::advance();
    }

    if (failed(cloneOpsInMainLoop(op))) {
      CVPipeline::setFallbackAttr(module, CVPipeline::ERRCODE_FAILED);
      return WalkResult::interrupt();
    }

    if (failed(cleanupClonedOpsInMainLoop(op))) {
      CVPipeline::setFallbackAttr(module, CVPipeline::ERRCODE_FAILED);
      return WalkResult::interrupt();
    }
    return WalkResult::advance();
  });
  if (walkResult.wasInterrupted()) {
    return;
  }

  LDBG("after cloneOps:\n" << module);

  // Validate no cloned tensor ops remaining in VECTOR main_loop op
  if (failed(validateClonedOpsInVector(module))) {
    CVPipeline::setFallbackAttr(module, CVPipeline::ERRCODE_FAILED);
    return;
  }
}

namespace mlir {
namespace triton {

std::unique_ptr<OperationPass<ModuleOp>> createCloneOpsPass() {
  return std::make_unique<CloneOpsPass>();
}

} // namespace triton
} // namespace mlir
