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

#include "ascend/include/DynamicCVPipeline/AddControlFlowCondition/Utils.h"
#include "ascend/include/DynamicCVPipeline/Common/Utils.h"
#include "bishengir/Dialect/HIVM/IR/HIVM.h"
#include "llvm/Support/Debug.h"
#include <optional>

static constexpr const char *DEBUG_TYPE = "AddControlFlowConditionUtils";
#define DBGS() (llvm::dbgs() << '[' << DEBUG_TYPE << "] ")
#define LDBG(...)                                                              \
  LLVM_DEBUG({                                                                 \
    DBGS();                                                                    \
    llvm::dbgs() << __VA_ARGS__;                                               \
    llvm::dbgs() << "\n";                                                      \
  })

namespace mlir {
using namespace llvm;
using namespace CVPipeline;

// Collect all nested ops within an operation's regions
LogicalResult collectAllNestedOps(Operation *op,
                                  llvm::DenseSet<Operation *> &regionOps) {
  if (!op) {
    return failure();
  }

  if (regionOps.contains(op)) {
    return success();
  }

  regionOps.insert(op);
  for (Region &region : op->getRegions()) {
    for (Block &block : region) {
      for (Operation &nestedOp : block) {
        if (failed(collectAllNestedOps(&nestedOp, regionOps))) {
          return failure();
        }
      }
    }
  }

  return success();
}

// Group operations by their block_id attribute. `op` must be scf.for or
// scf.while (see MainLoop::getBody).
LogicalResult
collectOpsByBlockId(Operation *op,
                    llvm::DenseMap<int, SmallVector<Operation *>> &blockOps) {
  Block *bodyBlock = MainLoop(op).getBody();
  if (!bodyBlock) {
    return failure();
  }

  for (Operation &op : bodyBlock->without_terminator()) {
    if (auto attr = op.getAttrOfType<IntegerAttr>(CVPipeline::kBlockId)) {
      blockOps[attr.getInt()].push_back(&op);
    } else {
      return failure();
    }
  }

  return success();
}

// DFS for topological sort - returns failure if cycle detected
static LogicalResult
dfsTopologicalSort(Operation *op, llvm::DenseSet<Operation *> &visited,
                   llvm::DenseSet<Operation *> &inStack,
                   const llvm::DenseSet<Operation *> &ops,
                   llvm::DenseMap<Operation *, int> *opOrder,
                   SmallVectorImpl<Operation *> &sorted) {
  if (!op) {
    return success();
  }
  if (inStack.contains(op)) {
    return failure();
  }
  if (visited.contains(op)) {
    return success();
  }

  visited.insert(op);
  inStack.insert(op);

  SmallVector<Operation *> deps;
  for (Value operand : op->getOperands()) {
    if (Operation *def = operand.getDefiningOp()) {
      if (ops.contains(def)) {
        deps.push_back(def);
      }
    }
  }

  if (opOrder && !opOrder->empty()) {
    llvm::sort(deps, [&](Operation *a, Operation *b) {
      auto itA = opOrder->find(a);
      auto itB = opOrder->find(b);
      if (itA == opOrder->end() || itB == opOrder->end()) {
        return false;
      }
      return itA->second < itB->second;
    });
  }

  for (Operation *dep : deps) {
    if (failed(
            dfsTopologicalSort(dep, visited, inStack, ops, opOrder, sorted))) {
      return failure();
    }
  }

  inStack.erase(op);
  sorted.push_back(op);
  return success();
}

// Topological sort of operations based on operand dependencies
LogicalResult topologicalSort(llvm::DenseSet<Operation *> &ops,
                              llvm::DenseMap<Operation *, int> *opOrder,
                              SmallVectorImpl<Operation *> &sorted) {
  llvm::DenseSet<Operation *> visited;
  llvm::DenseSet<Operation *> inStack;

  SmallVector<Operation *> opList(ops.begin(), ops.end());
  if (opOrder && !opOrder->empty()) {
    llvm::sort(opList, [&](Operation *a, Operation *b) {
      return (*opOrder)[a] < (*opOrder)[b];
    });
  }

  for (Operation *op : opList) {
    if (failed(
            dfsTopologicalSort(op, visited, inStack, ops, opOrder, sorted))) {
      return failure();
    }
  }
  return success();
}

LogicalResult topologicalSort(SmallVector<Operation *> &ops) {
  llvm::DenseSet<Operation *> opSet(ops.begin(), ops.end());
  SmallVector<Operation *> sorted;

  if (succeeded(topologicalSort(opSet, nullptr, sorted))) {
    ops.assign(sorted.begin(), sorted.end());
    return success();
  }
  return failure();
}

// Get block_ids in order of appearance in the main-loop body (forOp body
// or whileOp after-region body). Returns empty if `op` is neither.
SmallVector<int> getBlockIdsInOrder(Operation *op) {
  Block *bodyBlock = MainLoop(op).getBody();
  if (!bodyBlock) {
    return {};
  }

  SmallVector<int> idsInOrder;
  llvm::DenseSet<int> seenIds;

  for (Operation &op : bodyBlock->without_terminator()) {
    if (auto blockIdAttr =
            op.getAttrOfType<IntegerAttr>(CVPipeline::kBlockId)) {
      int id = blockIdAttr.getInt();
      if (seenIds.insert(id).second) {
        idsInOrder.push_back(id);
      }
    }
  }
  return idsInOrder;
}

// Get block_id of the immediate child of main-loop op (scf.for or scf.while
// carrying ssbuffer.main_loop) containing op. For scf.while "body" is the
// after-region block.
std::optional<int> getLoopDirectChildBlockId(Operation *op) {
  if (!op) {
    return std::nullopt;
  }
  Operation *parent = op->getParentOp();
  while (parent) {
    if (CVPipeline::isMainLoopOp(parent)) {
      return CVPipeline::getOpBlockId(op);
    }
    op = parent;
    parent = parent->getParentOp();
  }
  return std::nullopt;
}

// Counts unique ssbuffer.if values inside a main-loop op (scf.for or
// scf.while), walking all nested ops. Returns 0 if none.
int countUniqueIfBlockIds(Operation *loopOp) {
  llvm::DenseSet<int> ifBlockIds;
  loopOp->walk([&](Operation *innerOp) {
    if (auto ifAttr = innerOp->getAttrOfType<IntegerAttr>(CVPipeline::kIf)) {
      ifBlockIds.insert(ifAttr.getInt());
    }
  });
  return static_cast<int>(ifBlockIds.size());
}

// Find the tcb group id that contains value v
int findTcbGroupId(
    Value v,
    llvm::DenseMap<int, SmallVector<Value>> &tightlyCoupledBufferGroups) {
  for (auto &tcbEntry : tightlyCoupledBufferGroups) {
    if (llvm::is_contained(tcbEntry.second, v)) {
      return tcbEntry.first;
    }
  }
  return -1;
}

// Get isCube/isVector from scopeOp's tcore_type attribute.
// Returns failure if attribute is missing or not CUBE/VECTOR.
LogicalResult getScopeType(Operation *scopeOp, bool &isCube, bool &isVector) {
  isCube = false;
  isVector = false;

  if (!scopeOp->hasAttr(CVPipeline::kTcoreType)) {
    return failure();
  }

  auto attr = scopeOp->getAttr(CVPipeline::kTcoreType);
  auto aiCAttr =
      hivm::TCoreTypeAttr::get(scopeOp->getContext(), hivm::TCoreType::CUBE);
  auto aiVAttr =
      hivm::TCoreTypeAttr::get(scopeOp->getContext(), hivm::TCoreType::VECTOR);
  if (attr == aiCAttr) {
    isCube = true;
  } else if (attr == aiVAttr) {
    isVector = true;
  }

  if (!isCube && !isVector) {
    return failure();
  }

  return success();
}

// Check if op is a scf.if whose body only contains hivm.hir.sync_block_wait,
// hivm.hir.sync_block_set and hivm.fixpipe ops (excluding terminators).
bool isIfOpWithOnlySyncOps(Operation *op) {
  auto ifOp = dyn_cast<scf::IfOp>(op);
  if (!ifOp) {
    return false;
  }

  WalkResult result = ifOp->walk([&](Operation *innerOp) -> WalkResult {
    if (innerOp == op) {
      return WalkResult::advance();
    }
    if (innerOp->hasTrait<OpTrait::IsTerminator>()) {
      return WalkResult::advance();
    }
    if (!isa<hivm::SyncBlockWaitOp>(innerOp) &&
        !isa<hivm::SyncBlockSetOp>(innerOp) && !isa<hivm::FixpipeOp>(innerOp)) {
      return WalkResult::interrupt();
    }
    return WalkResult::advance();
  });

  return !result.wasInterrupted();
}

// Migrate ops from oldBlock to newBlock; replaceAllUsesWith on oldBlock's args
// to newBlock's args (same index).
void migrateBody(Block *oldBlock, Block *newBlock) {
  for (unsigned i = 0; i < oldBlock->getNumArguments(); ++i) {
    oldBlock->getArgument(i).replaceAllUsesWith(newBlock->getArgument(i));
  }

  for (Operation &op :
       llvm::make_early_inc_range(oldBlock->without_terminator())) {
    op.moveBefore(newBlock, newBlock->end());
  }
}

// Migrate both before and after regions of a scf.while op. Does not touch
// terminators (the caller builds new condition/yield in the new regions).
void migrateWhileBodies(scf::WhileOp oldWhileOp, scf::WhileOp newWhileOp) {
  migrateBody(oldWhileOp.getBeforeBody(), newWhileOp.getBeforeBody());
  migrateBody(oldWhileOp.getAfterBody(), newWhileOp.getAfterBody());
}

// Build new scf.yield at end of `newBlock`: copies oldBlock's yield operands,
// appends `extraYieldValues`, creates new scf::YieldOp, erases old yield.
LogicalResult buildNewYieldOp(Block *oldBlock, Block *newBlock,
                              Operation *newOp,
                              ArrayRef<Value> extraYieldValues) {
  auto oldYield = cast<scf::YieldOp>(oldBlock->getTerminator());
  SmallVector<Value> yieldOperands;
  for (unsigned i = 0; i < oldYield.getNumOperands(); ++i) {
    yieldOperands.push_back(oldYield.getOperand(i));
  }
  for (Value v : extraYieldValues) {
    yieldOperands.push_back(v);
  }
  OpBuilder builder = OpBuilder::atBlockEnd(newBlock);
  builder.create<scf::YieldOp>(newOp->getLoc(), yieldOperands);
  oldYield.erase();
  return success();
}

// Replace all uses of `oldOp`'s results with `newOp`'s matching results.
// No-op when `oldOp` is result-less.
void replaceOpResultUses(Operation *oldOp, Operation *newOp) {
  if (oldOp->getNumResults() == 0)
    return;

  SmallVector<Value> newResults;
  for (unsigned i = 0; i < oldOp->getNumResults(); ++i) {
    newResults.push_back(newOp->getResult(i));
  }
  oldOp->replaceAllUsesWith(newResults);
}

// Build new scf.condition in `newWhileOp`'s before region. Condition preserved
// from `whileOp`; forwarded values = new before-block args (incl. extras).
void buildNewWhileCondition(scf::WhileOp whileOp, scf::WhileOp newWhileOp) {
  auto oldCond = whileOp.getConditionOp();
  Value origCond = oldCond.getCondition();

  OpBuilder beforeBuilder(newWhileOp.getBeforeBody(),
                          newWhileOp.getBeforeBody()->end());
  SmallVector<Value> forwardedValues;
  for (BlockArgument arg : newWhileOp.getBeforeArguments()) {
    forwardedValues.push_back(arg);
  }
  beforeBuilder.create<scf::ConditionOp>(whileOp.getLoc(), origCond,
                                         forwardedValues);
  oldCond.erase();
}

// Creates a new scf.for with `extraInitArgs` appended to the original init
// args. Returns `oldForOp` unchanged when `extraInitArgs` is empty.
scf::ForOp createNewForOpWithExtras(scf::ForOp oldForOp,
                                    ArrayRef<Value> extraInitArgs) {
  if (extraInitArgs.empty()) {
    return oldForOp;
  }

  OpBuilder builder(oldForOp);
  SmallVector<Value> newInitArgs(oldForOp.getInitArgs().begin(),
                                 oldForOp.getInitArgs().end());
  llvm::append_range(newInitArgs, extraInitArgs);

  scf::ForOp newForOp = builder.create<scf::ForOp>(
      oldForOp.getLoc(), oldForOp.getLowerBound(), oldForOp.getUpperBound(),
      oldForOp.getStep(), newInitArgs);

  for (auto &attr : oldForOp->getAttrs()) {
    newForOp->setAttr(attr.getName(), attr.getValue());
  }
  return newForOp;
}

// Creates a new scf.while with `extraInitArgs` appended to the original inits
// and empty before/after blocks. Returns `oldWhileOp` unchanged when empty.
scf::WhileOp createNewWhileOpWithExtras(scf::WhileOp oldWhileOp,
                                        ArrayRef<Value> extraInitArgs) {
  if (extraInitArgs.empty()) {
    return oldWhileOp;
  }

  OpBuilder builder(oldWhileOp);

  SmallVector<Value> newInits(oldWhileOp.getInits().begin(),
                              oldWhileOp.getInits().end());
  llvm::append_range(newInits, extraInitArgs);

  SmallVector<Type> newResultTypes(oldWhileOp->getResultTypes().begin(),
                                   oldWhileOp->getResultTypes().end());
  for (Value v : extraInitArgs) {
    newResultTypes.push_back(v.getType());
  }

  scf::WhileOp newWhileOp = builder.create<scf::WhileOp>(
      oldWhileOp.getLoc(), newResultTypes, newInits);

  for (auto &attr : oldWhileOp->getAttrs()) {
    newWhileOp->setAttr(attr.getName(), attr.getValue());
  }

  SmallVector<Type> argTypes;
  argTypes.reserve(newInits.size());
  for (Value v : newInits) {
    argTypes.push_back(v.getType());
  }
  SmallVector<Location> argLocs(newInits.size(), oldWhileOp.getLoc());

  builder.createBlock(&newWhileOp.getBefore(), {}, argTypes, argLocs);
  builder.createBlock(&newWhileOp.getAfter(), {}, argTypes, argLocs);

  return newWhileOp;
}

// Dispatches createNewForOpWithExtras / createNewWhileOpWithExtras by op type.
// Returns nullptr if `oldOp` is neither scf.for nor scf.while.
Operation *createMainLoopOpWithExtras(Operation *oldOp,
                                      ArrayRef<Value> extraInitArgs) {
  if (auto forOp = dyn_cast<scf::ForOp>(oldOp)) {
    return createNewForOpWithExtras(forOp, extraInitArgs);
  }
  if (auto whileOp = dyn_cast<scf::WhileOp>(oldOp)) {
    return createNewWhileOpWithExtras(whileOp, extraInitArgs);
  }
  return nullptr;
}

// Prints whileBlockArgMap (whileOp -> block_id -> (new_arg_idx ->
// old_arg_idx)) to the debug stream, gated by LLVM_DEBUG.
void dumpWhileBlockArgMap(const triton::WhileBlockArgMap &map,
                          llvm::StringRef header) {
  LLVM_DEBUG({
    LDBG("[INFO]: " << header);
    for (auto &[whileOp, blockArgMap] : map) {
      LDBG("  whileOp @" << whileOp->getLoc());
      for (auto &[blockId, argIdxMap] : blockArgMap) {
        for (auto &[newArgIdx, oldArgIdx] : argIdxMap) {
          LDBG("    block_id=" << blockId << " new_arg_idx=" << newArgIdx
                               << " -> old_arg_idx=" << oldArgIdx);
        }
      }
    }
  });
}

} // namespace mlir
