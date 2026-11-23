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

#include "ascend/include/DynamicCVPipeline/AddControlFlowCondition/UpdateLoopOps.h"
#include "ascend/include/DynamicCVPipeline/AddControlFlowCondition/Utils.h"
#include "ascend/include/DynamicCVPipeline/Common/Utils.h"
#include "bishengir/Dialect/HIVM/IR/HIVM.h"
#include "bishengir/Dialect/Scope/IR/Scope.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/BuiltinTypes.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/Debug.h"

static constexpr const char *DEBUG_TYPE = "UpdateLoopOps";
static constexpr int kPipeSFlagId = 15;
#define DBGS() (llvm::dbgs() << '[' << DEBUG_TYPE << "] ")
#define LDBG(...)                                                              \
  LLVM_DEBUG({                                                                 \
    DBGS();                                                                    \
    llvm::dbgs() << __VA_ARGS__;                                               \
    llvm::dbgs() << "\n";                                                      \
  })

using namespace mlir;
using namespace triton;
using namespace hivm;
using namespace CVPipeline;

// Derive block counters from ssbuffer.if for the main-loop op (scf.for or
// scf.while with ssbuffer.main_loop). blockCounterNums keyed on Operation*.
LogicalResult UpdateLoopOpsPass::deriveBlockCountersFromIfOps(
    ModuleOp module, ControlFlowConditionInfo *info) {
  if (!info) {
    LDBG("[Error]: info is null");
    return failure();
  }

  module.walk([&](Operation *op) -> WalkResult {
    if (!isMainLoopOp(op)) {
      return WalkResult::advance();
    }

    int numIfBlockIds = countUniqueIfBlockIds(op);
    if (numIfBlockIds > 0) {
      info->blockCounterNums[op] = numIfBlockIds;
    }
    return WalkResult::advance();
  });

  return success();
}

// Creates new scf.for with extras and migrates old body. Yields each new
// iter_arg forward (counts on themselves, deps/tensors via buildNewYieldOp).
static scf::ForOp
createForOpAndMigrateBody(scf::ForOp oldForOp,
                          const llvm::SmallVector<Value> &extraInitArgs) {
  scf::ForOp newForOp = createNewForOpWithExtras(oldForOp, extraInitArgs);
  if (newForOp == oldForOp) {
    return oldForOp;
  }

  Block *oldBlock = oldForOp.getBody();
  Block *newBlock = newForOp.getBody();
  migrateBody(oldBlock, newBlock);

  // Append extra iter_arg yields (new block args at [1+oldNumArgs, +numExtra)).
  unsigned oldNumArgs = oldForOp.getNumRegionIterArgs();
  llvm::SmallVector<Value> extras;
  extras.reserve(extraInitArgs.size());
  for (size_t i = 0; i < extraInitArgs.size(); ++i) {
    extras.push_back(newBlock->getArgument(1 + oldNumArgs + i));
  }
  if (failed(buildNewYieldOp(oldBlock, newBlock, newForOp, extras))) {
    newForOp.erase();
    return scf::ForOp();
  }

  return newForOp;
}

// Splices the new main-loop op in place of the old one: type-checks results,
// redirects external uses to new results, erases old op.
static LogicalResult replaceMainLoopOpUsesAndErase(Operation *oldOp,
                                                   Operation *newOp) {
  if (oldOp->getNumResults() > 0) {
    for (unsigned i = 0; i < oldOp->getNumResults(); ++i) {
      if (oldOp->getResult(i).getType() != newOp->getResult(i).getType()) {
        LDBG("[Error]: main_loop op result type mismatch at index " << i);
        return failure();
      }
    }
  }

  replaceOpResultUses(oldOp, newOp);
  oldOp->erase();
  return success();
}

// Appends the new iter_args (new before-block args at
// [numOriginal..numOriginal+numExtra-1]) to the scf.condition's forwarded
// values.
static void extendWhileCondition(scf::ConditionOp oldCond, Block *newBefore,
                                 unsigned numOriginalIterArgs,
                                 unsigned numExtraArgs) {
  llvm::SmallVector<Value> newCondValues(oldCond.getArgs().begin(),
                                         oldCond.getArgs().end());
  for (unsigned i = 0; i < numExtraArgs; ++i) {
    newCondValues.push_back(newBefore->getArgument(numOriginalIterArgs + i));
  }
  oldCond.getArgsMutable().assign(newCondValues);
}

// Appends new iter_args (new after-block args at
// [numOriginal..numOriginal+numExtra-1]) to scf.yield. Counters forward
// themselves.
static void extendWhileYield(scf::YieldOp oldYield, Block *newAfter,
                             unsigned numOriginalIterArgs,
                             unsigned numExtraArgs) {
  llvm::SmallVector<Value> newYieldOperands(oldYield.getOperands().begin(),
                                            oldYield.getOperands().end());
  for (unsigned i = 0; i < numExtraArgs; ++i) {
    newYieldOperands.push_back(newAfter->getArgument(numOriginalIterArgs + i));
  }
  oldYield.getResultsMutable().assign(newYieldOperands);
}

// Migrates both before/after regions: migrateBody args+ops, move terminator,
// extend with new iter_args.
static LogicalResult migrateWhileRegions(scf::WhileOp oldWhileOp,
                                         scf::WhileOp newWhileOp,
                                         unsigned numOriginalIterArgs,
                                         unsigned numExtraArgs) {
  Block *oldBefore = oldWhileOp.getBeforeBody();
  Block *newBefore = newWhileOp.getBeforeBody();
  migrateBody(oldBefore, newBefore);
  auto oldCond = cast<scf::ConditionOp>(oldBefore->getTerminator());
  oldCond->moveBefore(newBefore, newBefore->end());
  extendWhileCondition(oldCond, newBefore, numOriginalIterArgs, numExtraArgs);

  Block *oldAfter = oldWhileOp.getAfterBody();
  Block *newAfter = newWhileOp.getAfterBody();
  migrateBody(oldAfter, newAfter);
  auto oldYield = cast<scf::YieldOp>(oldAfter->getTerminator());
  oldYield->moveBefore(newAfter, newAfter->end());
  extendWhileYield(oldYield, newAfter, numOriginalIterArgs, numExtraArgs);

  return success();
}

// Creates new scf.while with extras, then migrates old before/after regions.
// condition/yield forward new iter_args (mirroring forOp layout).
static scf::WhileOp
createWhileOpAndMigrateBody(scf::WhileOp oldWhileOp,
                            const llvm::SmallVector<Value> &extraInitArgs) {
  scf::WhileOp newWhileOp =
      createNewWhileOpWithExtras(oldWhileOp, extraInitArgs);
  if (newWhileOp == oldWhileOp) {
    return oldWhileOp;
  }

  unsigned numOriginalIterArgs = oldWhileOp.getInits().size();
  if (failed(migrateWhileRegions(oldWhileOp, newWhileOp, numOriginalIterArgs,
                                 extraInitArgs.size()))) {
    newWhileOp.erase();
    return scf::WhileOp();
  }

  return newWhileOp;
}

// Dispatches createForOpAndMigrateBody / createWhileOpAndMigrateBody by op
// type. Returns nullptr if op is neither scf.for nor scf.while.
static Operation *
createMainLoopOpAndMigrateBody(Operation *oldLoopOp,
                               const llvm::SmallVector<Value> &extraInitArgs) {
  if (auto forOp = dyn_cast<scf::ForOp>(oldLoopOp))
    return createForOpAndMigrateBody(forOp, extraInitArgs);
  if (auto whileOp = dyn_cast<scf::WhileOp>(oldLoopOp))
    return createWhileOpAndMigrateBody(whileOp, extraInitArgs);
  return nullptr;
}

// Computes extra-arg counts (block counters / inner dep conds / tensor
// iter_args) for `oldLoopOp`. Returns depsVecCopy so the caller can move it
// into info later.
static void computeMainLoopExtraArgs(
    Operation *oldLoopOp, ControlFlowConditionInfo *info, int &numBlockCounters,
    int &numInnerDepConds, int &numTensorIterArgs,
    llvm::SmallVector<TensorIterArgIfOpRelation> &depsVecCopy) {
  numBlockCounters = info->blockCounterNums.lookup(oldLoopOp);
  numInnerDepConds = info->intraCoreDependentMap.count(oldLoopOp)
                         ? (int)info->intraCoreDependentMap[oldLoopOp].size()
                         : 0;

  numTensorIterArgs = 0;
  auto tensorIterArgDepsIt = info->tensorIterArgDepsMap.find(oldLoopOp);
  if (tensorIterArgDepsIt != info->tensorIterArgDepsMap.end()) {
    depsVecCopy = tensorIterArgDepsIt->second;
    for (auto &entry : depsVecCopy) {
      numTensorIterArgs += entry.consumers.size();
    }
  }
}

// Returns the index of the first extra iter_arg in the new op. ForOp excludes
// IV; WhileOp shares arg layout between before/after.
static unsigned getMainLoopBaseIdx(Operation *oldLoopOp, bool isWhile) {
  return isWhile ? (unsigned)cast<scf::WhileOp>(oldLoopOp).getInits().size()
                 : cast<scf::ForOp>(oldLoopOp).getNumRegionIterArgs();
}

// Appends initial values for extra iter_args:
// - block counters: forOp reuses lowerBound; whileOp creates a fresh i32(0)
//   per counter (distinct SSA values)
// - inner dep conds: i32(0)
// - tensor iter_args: i32(1)
static void buildMainLoopExtraInitArgs(
    OpBuilder &builder, Location loc, Value forOpLowerBound, bool isWhile,
    int numBlockCounters, int numInnerDepConds, int numTensorIterArgs,
    llvm::SmallVector<Value> &extraInitArgs) {
  for (int i = 0; i < numBlockCounters; ++i) {
    if (isWhile) {
      extraInitArgs.push_back(builder.create<arith::ConstantOp>(
          loc, builder.getI32Type(), builder.getI32IntegerAttr(0)));
    } else {
      extraInitArgs.push_back(forOpLowerBound);
    }
  }
  for (int i = 0; i < numInnerDepConds; ++i) {
    extraInitArgs.push_back(builder.create<arith::ConstantOp>(
        loc, builder.getI32Type(), builder.getI32IntegerAttr(0)));
  }
  for (int i = 0; i < numTensorIterArgs; ++i) {
    extraInitArgs.push_back(builder.create<arith::ConstantOp>(
        loc, builder.getI32Type(), builder.getI32IntegerAttr(1)));
  }
}

// Migrates block-counter / inner-dep-cond index ranges from oldLoopOp to newOp.
static void recordMainLoopBlockCountersAndConds(
    Operation *oldLoopOp, Operation *newOp, ControlFlowConditionInfo *info,
    unsigned baseIdx, int numBlockCounters, int numInnerDepConds) {
  if (numBlockCounters > 0) {
    llvm::SmallVector<int> indices;
    for (int j = 0; j < numBlockCounters; ++j)
      indices.push_back(baseIdx + j);
    info->blockCounters.erase(oldLoopOp);
    info->blockCounters[newOp] = indices;
  }

  if (numInnerDepConds > 0) {
    llvm::SmallVector<int> indices;
    for (int j = 0; j < numInnerDepConds; ++j)
      indices.push_back(baseIdx + numBlockCounters + j);
    info->innerDepConds.erase(oldLoopOp);
    info->innerDepConds[newOp] = indices;
  }
}

// Migrates tensor iter_arg index ranges and moves depsVecCopy into
// info->tensorIterArgDepsMap.
static void recordMainLoopTensorIterArgs(
    Operation *oldLoopOp, Operation *newOp, ControlFlowConditionInfo *info,
    unsigned baseIdx, int numBlockCounters, int numInnerDepConds,
    int numTensorIterArgs,
    llvm::SmallVector<TensorIterArgIfOpRelation> &depsVecCopy) {
  if (numTensorIterArgs == 0) {
    return;
  }
  unsigned tensorBaseIdx = baseIdx + numBlockCounters + numInnerDepConds;
  auto &newIndicesMap = info->tensorIterArgIndicesMap[newOp];

  unsigned currentIdx = tensorBaseIdx;
  for (auto &entry : depsVecCopy) {
    llvm::SmallVector<int> indices;
    for (int j = 0; j < (int)entry.consumers.size(); ++j) {
      indices.push_back(currentIdx++);
    }
    newIndicesMap[entry.iterArg] = indices;
  }

  info->tensorIterArgIndicesMap.erase(oldLoopOp);
  info->tensorIterArgDepsMap[newOp] = std::move(depsVecCopy);
  info->tensorIterArgDepsMap.erase(oldLoopOp);
}

// Transfers intraCoreDependentMap (and WhileOp-only whileBlockArgMap) from
// oldLoopOp to newOp so the maps survive the old op being erased.
static void transferMainLoopInfoMaps(Operation *oldLoopOp, Operation *newOp,
                                     ControlFlowConditionInfo *info,
                                     bool isWhile) {
  if (info->intraCoreDependentMap.count(oldLoopOp)) {
    info->intraCoreDependentMap[newOp] = info->intraCoreDependentMap[oldLoopOp];
    info->intraCoreDependentMap.erase(oldLoopOp);
  }
  if (isWhile) {
    auto oldWhileOp = cast<scf::WhileOp>(oldLoopOp);
    auto newWhileOp = cast<scf::WhileOp>(newOp);
    if (info->whileBlockArgMap.count(oldWhileOp)) {
      info->whileBlockArgMap[newWhileOp] =
          std::move(info->whileBlockArgMap[oldWhileOp]);
      info->whileBlockArgMap.erase(oldWhileOp);
    }
  }
}

// Extends scf.for / scf.while with iter_args for block counters, inner dep
// conds, and tensor iter_args (seeded from maps).
static LogicalResult
extendMainLoopOpWithExtraArgs(Operation *oldLoopOp,
                              ControlFlowConditionInfo *info) {
  int numBlockCounters, numInnerDepConds, numTensorIterArgs;
  llvm::SmallVector<TensorIterArgIfOpRelation> depsVecCopy;
  computeMainLoopExtraArgs(oldLoopOp, info, numBlockCounters, numInnerDepConds,
                           numTensorIterArgs, depsVecCopy);

  int totalExtraArgs = numBlockCounters + numInnerDepConds + numTensorIterArgs;
  if (totalExtraArgs == 0) {
    return success();
  }

  // ForOp block-counter init uses lowerBound (single Value reused for every
  // counter); WhileOp creates a fresh arith.constant per counter so each new
  // iter_arg has a distinct SSA value (expected by tests and downstream
  // passes).
  OpBuilder builder(oldLoopOp);
  Value forOpLowerBound;
  bool isWhile = false;
  if (auto forOp = dyn_cast<scf::ForOp>(oldLoopOp)) {
    forOpLowerBound = forOp.getLowerBound();
  } else if (isa<scf::WhileOp>(oldLoopOp)) {
    isWhile = true;
  } else {
    LDBG("[Error]: main_loop op is neither scf::ForOp nor scf::WhileOp");
    return failure();
  }

  llvm::SmallVector<Value> extraInitArgs;
  buildMainLoopExtraInitArgs(builder, oldLoopOp->getLoc(), forOpLowerBound,
                             isWhile, numBlockCounters, numInnerDepConds,
                             numTensorIterArgs, extraInitArgs);

  Operation *newOp = createMainLoopOpAndMigrateBody(oldLoopOp, extraInitArgs);
  if (!newOp) {
    return failure();
  }

  unsigned baseIdx = getMainLoopBaseIdx(oldLoopOp, isWhile);
  recordMainLoopBlockCountersAndConds(oldLoopOp, newOp, info, baseIdx,
                                      numBlockCounters, numInnerDepConds);
  recordMainLoopTensorIterArgs(oldLoopOp, newOp, info, baseIdx,
                               numBlockCounters, numInnerDepConds,
                               numTensorIterArgs, depsVecCopy);
  transferMainLoopInfoMaps(oldLoopOp, newOp, info, isWhile);

  return replaceMainLoopOpUsesAndErase(oldLoopOp, newOp);
}

// Add block counter and inner dep cond iter args to for ops (and whileOps).
// Processed from info-driven `mainLoopOpsToProcess` set (keyed on Operation*).
LogicalResult UpdateLoopOpsPass::addBlockCountersAndInnerDepConds(
    ModuleOp module, ControlFlowConditionInfo *info) {
  llvm::DenseSet<Operation *> mainLoopOpsToProcess;

  for (auto &p : info->blockCounterNums) {
    if (p.second < 0) {
      LDBG("[Error]: invalid blockCounterNum " << p.second);
      return failure();
    }
    mainLoopOpsToProcess.insert(p.first);
  }
  for (auto &p : info->tensorIterArgDepsMap) {
    mainLoopOpsToProcess.insert(p.first);
  }

  for (Operation *loopOp : mainLoopOpsToProcess) {
    if (failed(extendMainLoopOpWithExtraArgs(loopOp, info)))
      return failure();
  }

  return success();
}

// Insert sync ops inside a main-loop body (scf.for or scf.while after-region):
// wait at start, set before yield. Body block is resolved via getMainLoopBody.
static LogicalResult
insertSyncOpsInsideMainLoop(Block *loopBody, Location loc,
                            hivm::TCoreTypeAttr coreType, PipeAttr setPipe,
                            PipeAttr waitPipe, int waitFlagId, int setFlagId) {
  Operation *forTerminator = loopBody->getTerminator();
  if (!forTerminator) {
    return failure();
  }

  // Insert wait at loop body start
  OpBuilder insertionBuilder(&loopBody->front());
  auto waitFlagAttr = insertionBuilder.getIntegerAttr(
      insertionBuilder.getI64Type(), waitFlagId);
  insertionBuilder.create<SyncBlockWaitOp>(loc, coreType, setPipe, waitPipe,
                                           waitFlagAttr);

  // Insert set before yield
  OpBuilder setBuilder(forTerminator);
  auto setFlagAttr =
      setBuilder.getIntegerAttr(setBuilder.getI64Type(), setFlagId);
  setBuilder.setInsertionPoint(forTerminator);
  setBuilder.create<SyncBlockSetOp>(loc, coreType, setPipe, waitPipe,
                                    setFlagAttr);

  return success();
}

// Insert a sync op (SET before, or WAIT after) outside a main-loop op.
// op-agnostic. `isBefore`: true -> SET, false -> WAIT.
static LogicalResult
insertSyncOpsOutsideMainLoop(Operation *loopOp, Location loc,
                             hivm::TCoreTypeAttr coreType, PipeAttr setPipe,
                             PipeAttr waitPipe, int flagId, bool isBefore) {
  OpBuilder builder(loopOp);
  auto flagAttr = builder.getIntegerAttr(builder.getI64Type(), flagId);
  if (isBefore) {
    builder.create<SyncBlockSetOp>(loc, coreType, setPipe, waitPipe, flagAttr);
  } else {
    builder.setInsertionPointAfter(loopOp);
    builder.create<SyncBlockWaitOp>(loc, coreType, setPipe, waitPipe, flagAttr);
  }
  return success();
}

// Returns the loop body block for a main-loop op. scf.for has the single
// region; scf.while uses the after region (where the actual loop body lives).
static Block *getMainLoopBody(Operation *loopOp) {
  if (auto whileOp = dyn_cast<scf::WhileOp>(loopOp)) {
    return &whileOp.getAfter().front();
  }
  return &loopOp->getRegion(0).front();
}

// Insert PIPE_S for a main_loop op (scf.for or scf.while) by loop/scope type.
// Body resolved via getMainLoopBody — scf.while uses the after region block.
static LogicalResult
insertPipeSForMainLoopOp(Operation *loopOp, scope::ScopeOp scopeOp,
                         bool isScopeCube, bool isScopeVector, PipeAttr setPipe,
                         PipeAttr waitPipe, int flagId) {
  Block *loopBody = getMainLoopBody(loopOp);
  Location loc = loopOp->getLoc();
  bool isVectorFirst = loopOp->hasAttr(CVPipeline::kVectorFirst);
  auto cubeType =
      hivm::TCoreTypeAttr::get(loopOp->getContext(), hivm::TCoreType::CUBE);
  auto vectorType =
      hivm::TCoreTypeAttr::get(loopOp->getContext(), hivm::TCoreType::VECTOR);

  if (isVectorFirst) {
    if (isScopeCube) {
      // vector_first + CUBE: before loop op (SET), inside (WAIT/SET)
      if (failed(insertSyncOpsOutsideMainLoop(loopOp, loc, cubeType, setPipe,
                                              waitPipe, flagId, true))) {
        return failure();
      }
      if (failed(insertSyncOpsInsideMainLoop(loopBody, loc, cubeType, setPipe,
                                             waitPipe, flagId, flagId))) {
        return failure();
      }
    } else if (isScopeVector) {
      // vector_first + VECTOR: inside (WAIT/SET), after loop op (WAIT)
      if (failed(insertSyncOpsInsideMainLoop(loopBody, loc, vectorType, setPipe,
                                             waitPipe, flagId, flagId))) {
        return failure();
      }
      if (failed(insertSyncOpsOutsideMainLoop(loopOp, loc, vectorType, setPipe,
                                              waitPipe, flagId, false))) {
        return failure();
      }
    }
  } else {
    // cube_first (including default when neither attribute is present)
    if (isScopeCube) {
      // cube_first + CUBE: inside (WAIT/SET), after loop op (WAIT)
      if (failed(insertSyncOpsInsideMainLoop(loopBody, loc, cubeType, setPipe,
                                             waitPipe, flagId, flagId))) {
        return failure();
      }
      if (failed(insertSyncOpsOutsideMainLoop(loopOp, loc, cubeType, setPipe,
                                              waitPipe, flagId, false))) {
        return failure();
      }
    } else if (isScopeVector) {
      // cube_first + VECTOR: before loop op (SET), inside (WAIT/SET)
      if (failed(insertSyncOpsOutsideMainLoop(loopOp, loc, vectorType, setPipe,
                                              waitPipe, flagId, true))) {
        return failure();
      }
      if (failed(insertSyncOpsInsideMainLoop(loopBody, loc, vectorType, setPipe,
                                             waitPipe, flagId, flagId))) {
        return failure();
      }
    }
  }
  return success();
}

LogicalResult UpdateLoopOpsPass::insertInterCorePipeS(ModuleOp module) {
  auto cubeCoreType =
      hivm::TCoreTypeAttr::get(module.getContext(), hivm::TCoreType::CUBE);
  auto vectorCoreType =
      hivm::TCoreTypeAttr::get(module.getContext(), hivm::TCoreType::VECTOR);
  auto setPipeType = PipeAttr::get(module.getContext(), hivm::PIPE::PIPE_S);
  auto waitPipeType = PipeAttr::get(module.getContext(), hivm::PIPE::PIPE_S);

  WalkResult result = module.walk([&](scope::ScopeOp scopeOp) -> WalkResult {
    auto scopeTypeAttr =
        scopeOp->getAttrOfType<hivm::TCoreTypeAttr>("hivm.tcore_type");
    if (!scopeTypeAttr) {
      return WalkResult::advance();
    }

    bool isScopeCube = (scopeTypeAttr == cubeCoreType);
    bool isScopeVector = (scopeTypeAttr == vectorCoreType);

    // Walk both scf.for and scf.while with ssbuffer.main_loop; both need
    // same PIPE_S sync. insertPipeSForMainLoopOp dispatches via
    // getMainLoopBody.
    WalkResult innerResult = scopeOp.walk([&](Operation *op) -> WalkResult {
      if (!isMainLoopOp(op)) {
        return WalkResult::advance();
      }
      if (failed(insertPipeSForMainLoopOp(op, scopeOp, isScopeCube,
                                          isScopeVector, setPipeType,
                                          waitPipeType, kPipeSFlagId))) {
        return WalkResult::interrupt();
      }
      return WalkResult::advance();
    });

    if (innerResult.wasInterrupted()) {
      return WalkResult::interrupt();
    }
    return WalkResult::advance();
  });

  return result.wasInterrupted() ? failure() : success();
}

// Analyzes producer/consumer relationship between tensor iter_args in
// main_loop and ssbuffer.if. Downstream only acts on forOp (whileOp future
// use).
LogicalResult UpdateLoopOpsPass::analyzeTensorIterArgDependencies(
    ModuleOp module, ControlFlowConditionInfo *info) {
  bool failed = false;
  module.walk([&](Operation *op) -> WalkResult {
    if (!isMainLoopOp(op)) {
      return WalkResult::advance();
    }

    LDBG("Analyzing main_loop op: " << op);

    // Get iter_args to analyze. forOp: getRegionIterArgs() (body args - IV).
    // whileOp: getAfterArguments() (before-block args aren't visible in body).
    llvm::SmallVector<Value> iterArgsVec = MainLoop(op).getIterArgs();

    for (auto iterArg : iterArgsVec) {
      if (!mlir::isa<TensorType>(iterArg.getType())) {
        continue;
      }

      LDBG("Found tensor type iter_arg: " << iterArg);
      scf::IfOp producerIfOp = nullptr;
      llvm::SmallVector<scf::IfOp> consumerIfOps;

      for (auto &use : iterArg.getUses()) {
        Operation *user = use.getOwner();
        scf::IfOp ifOp = nullptr;
        Operation *curr = user;
        while (curr && curr != op) {
          if (auto currIf = dyn_cast<scf::IfOp>(curr)) {
            if (currIf->hasAttr(CVPipeline::kIf)) {
              ifOp = currIf;
              break;
            }
          }
          curr = curr->getParentOp();
        }

        if (!ifOp) {
          LDBG("Use of tensor iter_arg "
               << iterArg << " is not inside any ssbuffer.if op.");
          continue;
        }

        // Only the direct terminator of this ssbuffer.if is a producer.
        // Nested if/for/while yields that forward iter_arg are consumers.
        bool isProducer = isa<scf::YieldOp>(user) &&
                          user->getParentOp() == ifOp.getOperation();

        // Check and update status of ifOp
        if (isProducer) {
          if (producerIfOp && producerIfOp != ifOp) {
            // Found a different producer ifOp! This is an error.
            LDBG("[Error]: tensor iter_arg "
                 << iterArg << " has multiple different producers!");
            LDBG("Existing producer: " << producerIfOp);
            LDBG("New producer: " << ifOp);
            failed = true;
            return WalkResult::interrupt();
          }
          if (!producerIfOp) {
            // First producer, or upgrade from consumer
            auto it = llvm::find(consumerIfOps, ifOp);
            if (it != consumerIfOps.end()) {
              consumerIfOps.erase(it);
              LDBG("This ifOp was consumer, now updated to producer: " << ifOp);
            } else {
              LDBG("Found producer ifOp (first time): " << ifOp);
            }
            producerIfOp = ifOp;
          }
          // Else: already is this producer, do nothing
        } else {
          // isConsumer
          if (producerIfOp == ifOp) {
            // Already a producer, even if current use is consumer, do nothing
            continue;
          }
          if (!llvm::is_contained(consumerIfOps, ifOp)) {
            consumerIfOps.push_back(ifOp);
            LDBG("Found consumer ifOp (first time): " << ifOp << "\n");
          }
          // Else: already a consumer, do nothing
        }
      }
      // Check: must have both producers AND consumers
      if (!producerIfOp || consumerIfOps.empty()) {
        LDBG("tensor iter_arg " << iterArg << " has only "
                                << (!producerIfOp ? "consumers" : "producers")
                                << ", skipped");
        continue;
      }
      TensorIterArgIfOpRelation relation;
      relation.iterArg = iterArg;
      relation.producer = producerIfOp;
      relation.consumers = consumerIfOps;

      // Record into the map (keyed on Operation*; both scf.for and scf.while
      // participate).
      info->tensorIterArgDepsMap[op].push_back(relation);
      LDBG("Recorded tensor iter_arg dependency: "
           << iterArg << " has 1 producer, " << relation.consumers.size()
           << " consumers");
    }

    return WalkResult::advance();
  });

  return failed ? failure() : success();
}

void UpdateLoopOpsPass::runOnOperation() {
  ModuleOp module = getOperation();

  if (CVPipeline::hasFallbackAttr(module)) {
    return;
  }

  LDBG("before updateLoopOps:\n" << module);

  // Use provided info, or create a local one if not available
  ControlFlowConditionInfo localInfo;
  ControlFlowConditionInfo *infoToUse = info ? info : &localInfo;

  // Analyze the dependencies of the tensor type iter_args in the main_loop with
  // the ssbuffer.if ops
  if (failed(analyzeTensorIterArgDependencies(module, infoToUse))) {
    CVPipeline::setFallbackAttr(module, CVPipeline::ERRCODE_FAILED);
    return;
  }

  // Derive block counters from ssbuffer.if if blockCounterNums is empty
  if (infoToUse->blockCounterNums.empty()) {
    if (failed(deriveBlockCountersFromIfOps(module, infoToUse))) {
      CVPipeline::setFallbackAttr(module, CVPipeline::ERRCODE_FAILED);
      return;
    }
  }

  // Update for/while ops iter_args for block counters and inner dep conds.
  // forOp from info-driven sets (no-ops when empty); whileOp unconditionally.
  if (infoToUse &&
      (failed(addBlockCountersAndInnerDepConds(module, infoToUse)))) {
    CVPipeline::setFallbackAttr(module, CVPipeline::ERRCODE_FAILED);
    return;
  }

  // Insert PIPE_S inter-core synchronization
  if (failed(insertInterCorePipeS(module))) {
    CVPipeline::setFallbackAttr(module, CVPipeline::ERRCODE_FAILED);
    return;
  }

  // Dump whileBlockArgMap (whileOp -> block_id -> (new_arg_idx -> old_arg_idx))
  // after all whileOp replacements; verify it survived
  // replaceMainLoopOpUsesAndErase.
  dumpWhileBlockArgMap(infoToUse->whileBlockArgMap,
                       "whileBlockArgMap contents after updateLoopOps "
                       "(whileOp -> block_id -> (new_arg_idx -> old_arg_idx))");

  LDBG("after updateLoopOps:\n" << module);
}

namespace mlir {
namespace triton {

std::unique_ptr<OperationPass<ModuleOp>> createUpdateLoopOpsPass() {
  return std::make_unique<UpdateLoopOpsPass>();
}

} // namespace triton
} // namespace mlir
