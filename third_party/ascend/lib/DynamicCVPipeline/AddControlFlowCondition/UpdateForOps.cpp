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

#include "ascend/include/DynamicCVPipeline/AddControlFlowCondition/UpdateForOps.h"
#include "bishengir/Dialect/HIVM/IR/HIVM.h"
#include "bishengir/Dialect/Scope/IR/Scope.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/Support/Debug.h"
#include "mlir/Dialect/SCF/IR/SCF.h"

static constexpr const char *DEBUG_TYPE = "UpdateForOps";
static constexpr int kPipeSFlagId = 15;
#define DBGS() (llvm::dbgs() << '[' << DEBUG_TYPE << "] ")
#define LDBG(...) \
LLVM_DEBUG({ \
  DBGS(); \
  llvm::outs() << __VA_ARGS__; \
  llvm::outs() << "\n"; \
})

using namespace llvm;
using namespace mlir;
using namespace triton;
using namespace hivm;

// Replace old block arguments with new ones
static LogicalResult replaceBlockArguments(Block *oldBlock, Block *newBlock)
{
  if (!oldBlock || !newBlock) {
    LDBG("[Error]: oldBlock or newBlock is null\n");
    return failure();
  }

  unsigned totalArgs = oldBlock->getNumArguments();

  for (unsigned i = 0; i < totalArgs; ++i) {
    oldBlock->getArgument(i).replaceAllUsesWith(newBlock->getArgument(i));
  }

  return success();
}

// Collect forOps that need processing based on info
static SmallVector<scf::ForOp> collectForOpsToProcess(
    ModuleOp module, const llvm::DenseMap<scf::ForOp, int> &numInfo)
{
  SmallVector<scf::ForOp> forOps;

  module.walk([&](scf::ForOp forOp) {
    if (numInfo.count(forOp)) {
      forOps.push_back(forOp);
    }
  });

  return forOps;
}

// Create new yield operands: original yield ops + extra args from new block
static SmallVector<Value> createNewYieldOperands(
    scf::YieldOp oldYield, unsigned oldNumArgs,
    Block *newBlock, int numExtraArgs)
{
  SmallVector<Value> newYieldOperands;

  for (unsigned i = 0; i < oldNumArgs; ++i) {
    newYieldOperands.push_back(oldYield.getOperand(i));
  }

  for (int i = 0; i < numExtraArgs; ++i) {
    newYieldOperands.push_back(newBlock->getArgument(1 + oldNumArgs + i));
  }

  return newYieldOperands;
}

// Derive block counters from ssbuffer.if attributes when info is not pre-populated
LogicalResult UpdateForOpsPass::deriveBlockCountersFromIfOps(ModuleOp module, ControlFlowConditionInfo *info)
{
  if (!info) {
    LDBG("[Error]: info is null\n");
    return failure();
  }

  module.walk([&](Operation *op) -> WalkResult {
    if (!op->hasAttr("ssbuffer.main_loop")) {
      return WalkResult::advance();
    }
    auto forOp = dyn_cast<scf::ForOp>(op);
    if (!forOp) {
      LDBG("[Error]: op with ssbuffer.main_loop is not a scf::ForOp\n");
      return WalkResult::interrupt();
    }

    // Collect unique ssbuffer.if values in this for op's body
    llvm::DenseSet<int> ifBlockIds;

    forOp.walk([&](Operation *innerOp) {
      if (auto ifAttr = innerOp->getAttrOfType<IntegerAttr>("ssbuffer.if")) {
        ifBlockIds.insert(ifAttr.getInt());
      }
    });

    if (!ifBlockIds.empty()) {
      info->blockCounterNums[forOp] = ifBlockIds.size();
    }
    return WalkResult::advance();
  });

  return success();
}

// Create new for op with extra iter args and migrate body
static scf::ForOp createForOpAndMigrateBody(
    scf::ForOp oldForOp, int numExtraArgs,
    const SmallVector<Value> &extraInitArgs)
{
  if (numExtraArgs < 0) {
    LDBG("[Error]: invalid numExtraArgs " << numExtraArgs << "\n");
    return scf::ForOp();
  }
  if (numExtraArgs == 0)
    return oldForOp;
  if ((int)extraInitArgs.size() != numExtraArgs) {
    LDBG("[Error]: extraInitArgs size " << extraInitArgs.size() << " != numExtraArgs " << numExtraArgs << "\n");
    return scf::ForOp();
  }

  OpBuilder builder(oldForOp);
  // Create new for op with extra iter args
  SmallVector<Value> newInitArgs(oldForOp.getInitArgs().begin(),
                                 oldForOp.getInitArgs().end());
  llvm::append_range(newInitArgs, extraInitArgs);

  scf::ForOp newForOp = builder.create<scf::ForOp>(
      oldForOp.getLoc(), oldForOp.getLowerBound(), oldForOp.getUpperBound(),
      oldForOp.getStep(), newInitArgs);

  for (auto &attr : oldForOp->getAttrs())
    newForOp->setAttr(attr.getName(), attr.getValue());

  // Migrate body
  Block *oldBlock = oldForOp.getBody();
  Block *newBlock = newForOp.getBody();

  if (failed(replaceBlockArguments(oldBlock, newBlock)))
    return scf::ForOp();

  for (Operation &op : llvm::make_early_inc_range(oldBlock->without_terminator()))
    op.moveBefore(newBlock, newBlock->end());

  auto oldYield = cast<scf::YieldOp>(oldBlock->getTerminator());
  SmallVector<Value> newYieldOperands = createNewYieldOperands(
      oldYield, oldForOp.getNumRegionIterArgs(), newBlock, numExtraArgs);

  builder.setInsertionPointToEnd(newBlock);
  builder.create<scf::YieldOp>(newForOp.getLoc(), newYieldOperands);
  oldYield.erase();

  return newForOp;
}

static LogicalResult replaceForOpUsesAndErase(scf::ForOp oldForOp, scf::ForOp newForOp)
{
  if (oldForOp.getNumResults() > 0) {
    SmallVector<Value> newResults;
    for (unsigned i = 0; i < oldForOp.getNumResults(); ++i) {
      if (oldForOp.getResult(i).getType() != newForOp.getResult(i).getType()) {
        LDBG("[Error]: result type mismatch at index " << i << "\n");
        return failure();
      }
      newResults.push_back(newForOp.getResult(i));
    }
    oldForOp.replaceAllUsesWith(newResults);
  }

  oldForOp.erase();
  return success();
}

LogicalResult extendForOpWithExtraArgs(scf::ForOp oldForOp, ControlFlowConditionInfo *info)
{
  int numBlockCounters = info->blockCounterNums[oldForOp];
  int numInnerDepConds = info->intraCoreDependentMap[oldForOp].size();
  int totalExtraArgs = numBlockCounters + numInnerDepConds;
  if (totalExtraArgs == 0) {
    return success();
  }

  OpBuilder builder(oldForOp);
  SmallVector<Value> extraInitArgs;
  for (int i = 0; i < numBlockCounters; ++i)
    extraInitArgs.push_back(oldForOp.getLowerBound());
  for (int i = 0; i < numInnerDepConds; ++i)
    extraInitArgs.push_back(builder.create<arith::ConstantOp>(
        oldForOp.getLoc(), builder.getI32Type(), builder.getI32IntegerAttr(0)));

  scf::ForOp newForOp = createForOpAndMigrateBody(oldForOp, totalExtraArgs, extraInitArgs);
  if (!newForOp) {
    return failure();
  }

  unsigned baseIdx = oldForOp.getNumRegionIterArgs();
  if (numBlockCounters > 0) {
    SmallVector<int> indices;
    for (int j = 0; j < numBlockCounters; ++j)
      indices.push_back(baseIdx + j);
    info->blockCounters.erase(oldForOp);
    info->blockCounters[newForOp] = indices;
  }

  if (numInnerDepConds > 0) {
    SmallVector<int> indices;
    for (int j = 0; j < numInnerDepConds; ++j)
      indices.push_back(baseIdx + numBlockCounters + j);
    info->innerDepConds.erase(oldForOp);
    info->innerDepConds[newForOp] = indices;
  }

  if (info->intraCoreDependentMap.count(oldForOp)) {
    info->intraCoreDependentMap[newForOp] = info->intraCoreDependentMap[oldForOp];
    info->intraCoreDependentMap.erase(oldForOp);
  }

  return replaceForOpUsesAndErase(oldForOp, newForOp);
}

// Add block counter and inner dependency condition iter args to for ops
LogicalResult UpdateForOpsPass::addBlockCountersAndInnerDepConds(ModuleOp module, ControlFlowConditionInfo *info)
{
  llvm::DenseSet<scf::ForOp> allForOps;
  for (auto &p : info->blockCounterNums) {
    if (p.second < 0) {
      LDBG("[Error]: invalid blockCounterNum " << p.second << "\n");
      return failure();
    }
    allForOps.insert(p.first);
  }
  for (auto &entry : info->intraCoreDependentMap)
    allForOps.insert(entry.first);

  for (scf::ForOp oldForOp : allForOps) {
    if (failed(extendForOpWithExtraArgs(oldForOp, info)))
      return failure();
  }
  return success();
}

// Insert sync ops for a single forOp
static WalkResult insertSyncOpsForForOp(scf::ForOp forOp, Block *forBody,
                                        hivm::TCoreTypeAttr coreType,
                                        PipeAttr setPipe, PipeAttr waitPipe,
                                        int waitFlagId, int setFlagId)
{
  Operation *forTerminator = forBody->getTerminator();
  if (!forTerminator) {
    return WalkResult::interrupt();
  }

  Location loc = forOp.getLoc();

  // Insert wait at for loop start
  OpBuilder insertionBuilder(&forBody->front());
  auto waitFlagAttr = insertionBuilder.getIntegerAttr(insertionBuilder.getI64Type(), waitFlagId);
  insertionBuilder.create<SyncBlockWaitOp>(loc, coreType, setPipe, waitPipe, waitFlagAttr);

  // Insert set before yield
  OpBuilder setBuilder(forTerminator);
  auto setFlagAttr = setBuilder.getIntegerAttr(setBuilder.getI64Type(), setFlagId);
  setBuilder.setInsertionPoint(forTerminator);
  setBuilder.create<SyncBlockSetOp>(loc, coreType, setPipe, waitPipe, setFlagAttr);

  return WalkResult::advance();
}

// Insert sync ops for a single scopeOp
static WalkResult insertSyncOpsForCube(scope::ScopeOp scopeOp,
                                       hivm::TCoreTypeAttr coreType,
                                       PipeAttr setPipe, PipeAttr waitPipe,
                                       int waitFlagId, int setFlagId)
{
  Block &scopeBlock = scopeOp.getRegion().front();
  Operation *scopeTerminator = scopeBlock.getTerminator();
  if (!scopeTerminator) {
    return WalkResult::interrupt();
  }

  OpBuilder scopeBuilder(scopeTerminator);
  auto scopeFlagAttr = scopeBuilder.getIntegerAttr(scopeBuilder.getI64Type(), waitFlagId);
  scopeBuilder.setInsertionPoint(scopeTerminator);
  scopeBuilder.create<SyncBlockWaitOp>(scopeTerminator->getLoc(), coreType,
                                        setPipe, waitPipe, scopeFlagAttr);

  WalkResult innerResult = scopeOp.walk([&](scf::ForOp forOp) -> WalkResult {
    if (!forOp->hasAttr("ssbuffer.main_loop")) {
      return WalkResult::advance();
    }
    Block &forBody = forOp.getRegion().front();
    return insertSyncOpsForForOp(forOp, &forBody, coreType, setPipe, waitPipe,
                                 waitFlagId, setFlagId);
  });
  if (innerResult.wasInterrupted()) {
    return WalkResult::interrupt();
  }

  return WalkResult::advance();
}

// Insert sync ops for a single scopeOp (vector variant: inserts SyncBlockSetOp at scope start)
static WalkResult insertSyncOpsForVector(scope::ScopeOp scopeOp,
                                         hivm::TCoreTypeAttr coreType,
                                         PipeAttr setPipe, PipeAttr waitPipe,
                                         int waitFlagId, int setFlagId)
{
  Block &scopeBlock = scopeOp.getRegion().front();
  OpBuilder builder(&scopeBlock, scopeBlock.begin());
  auto scopeFlagAttr = builder.getIntegerAttr(builder.getI64Type(), setFlagId);
  builder.create<SyncBlockSetOp>(scopeOp.getLoc(), coreType, setPipe, waitPipe, scopeFlagAttr);

  WalkResult innerResult = scopeOp.walk([&](scf::ForOp forOp) -> WalkResult {
    if (!forOp->hasAttr("ssbuffer.main_loop")) {
      return WalkResult::advance();
    }
    Block &forBody = forOp.getRegion().front();
    return insertSyncOpsForForOp(forOp, &forBody, coreType, setPipe, waitPipe,
                                 waitFlagId, setFlagId);
  });
  if (innerResult.wasInterrupted()) {
    return WalkResult::interrupt();
  }

  return WalkResult::advance();
}

// Insert inter-core PIPE_S synchronization for cube cores
static LogicalResult insertInterCorePipeSForCube(ModuleOp module)
{
  auto cubeCoreType = hivm::TCoreTypeAttr::get(module.getContext(), hivm::TCoreType::CUBE);
  auto setPipeType = PipeAttr::get(module.getContext(), hivm::PIPE::PIPE_S);
  auto waitPipeType = PipeAttr::get(module.getContext(), hivm::PIPE::PIPE_S);

  WalkResult result = module.walk([&](scope::ScopeOp scopeOp) -> WalkResult {
    auto attr = scopeOp->getAttrOfType<hivm::TCoreTypeAttr>("hivm.tcore_type");
    if (!attr || attr != cubeCoreType) {
      return WalkResult::advance();
    }

    return insertSyncOpsForCube(scopeOp, cubeCoreType, setPipeType, waitPipeType,
                                kPipeSFlagId, kPipeSFlagId);
  });

  return result.wasInterrupted() ? failure() : success();
}

// Insert inter-core PIPE_S synchronization for vector cores
static LogicalResult insertInterCorePipeSForVector(ModuleOp module)
{
  auto vectorCoreType = hivm::TCoreTypeAttr::get(module.getContext(), hivm::TCoreType::VECTOR);
  auto setPipeType = PipeAttr::get(module.getContext(), hivm::PIPE::PIPE_S);
  auto waitPipeType = PipeAttr::get(module.getContext(), hivm::PIPE::PIPE_S);

  WalkResult result = module.walk([&](scope::ScopeOp scopeOp) -> WalkResult {
    auto attr = scopeOp->getAttrOfType<hivm::TCoreTypeAttr>("hivm.tcore_type");
    if (!attr || attr != vectorCoreType) {
      return WalkResult::advance();
    }

    return insertSyncOpsForVector(scopeOp, vectorCoreType, setPipeType, waitPipeType,
                                  kPipeSFlagId, kPipeSFlagId);
  });

  return result.wasInterrupted() ? failure() : success();
}

LogicalResult UpdateForOpsPass::insertInterCorePipeS(ModuleOp module)
{
  if (failed(insertInterCorePipeSForCube(module))) {
    return failure();
  }

  if (failed(insertInterCorePipeSForVector(module))) {
    return failure();
  }

  return success();
}

void UpdateForOpsPass::runOnOperation()
{
  ModuleOp module = getOperation();

  LDBG("before updateForOps:\n" << module << "\n");

  // Use provided info, or create a local one if not available
  ControlFlowConditionInfo localInfo;
  ControlFlowConditionInfo *infoToUse = info ? info : &localInfo;

  // Derive block counters from ssbuffer.if if blockCounterNums is empty
  if (infoToUse->blockCounterNums.empty()) {
    if (failed(deriveBlockCountersFromIfOps(module, infoToUse))) {
      signalPassFailure();
      return;
    }
  }

  // Update for ops iter_args for block counters and inner dependency conditions
  if (infoToUse && (!infoToUse->blockCounterNums.empty() || !infoToUse->intraCoreDependentMap.empty()))
    if (failed(addBlockCountersAndInnerDepConds(module, infoToUse))) {
      signalPassFailure();
      return;
    }

  // Insert PIPE_S inter-core synchronization
  if (failed(insertInterCorePipeS(module))) {
    signalPassFailure();
    return;
  }

  LDBG("after updateForOps:\n" << module << "\n");
}

namespace mlir {
namespace triton {

std::unique_ptr<OperationPass<ModuleOp>> createUpdateForOpsPass()
{
  return std::make_unique<UpdateForOpsPass>();
}

} // namespace triton
} // namespace mlir
