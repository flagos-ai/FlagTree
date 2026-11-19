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

#include "CodePartitionUtility.h"
#include "Utils/TritonVersionCompat.h"

#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/ImplicitLocOpBuilder.h"
#include "mlir/Transforms/RegionUtils.h"
#include "llvm/Support/DebugLog.h"

#define DEBUG_TYPE "gcu-ws-specialize"

using namespace mlir;

namespace ttg = mlir::triton::gpu;

namespace mlir::triton::gcu {

unsigned scanRegUsage(Block *block, AsyncTaskId asyncTaskId,
                      unsigned requestedRegisters) {
  assert(asyncTaskId != 0 && "producer group should not request registers");
  return requestedRegisters == 0 ? 192 : requestedRegisters;
}

Operation *SpecializeOp(Operation *op, IRMapping &mapping,
                        OpBuilderWithAsyncTaskIds &builder,
                        AsyncTaskId asyncTaskId);

Operation *SpecializeWhileOp(scf::WhileOp whileOp, IRMapping &mapping,
                             OpBuilderWithAsyncTaskIds &builder,
                             AsyncTaskId asyncTaskId);

// Collect argument indices that are used by the specific taskId.
static SmallVector<unsigned> collectBlockArgsForTask(scf::ForOp forOp,
                                                     int asyncTaskId) {
  // Collect argument indices that can be reached along the definition chain.
  SetVector<unsigned> argIndices;
  std::function<void(scf::ForOp, Value, unsigned)> dfs =
      [&](scf::ForOp nestedForOp, Value arg, unsigned argIdx) {
        for (auto user : arg.getUsers()) {
          // Skip ops that are not in the same async task
          if (!hasAsyncTaskId(user, asyncTaskId))
            continue;

          if (isa<scf::YieldOp>(user)) {
            if (auto ifOp = dyn_cast<scf::IfOp>(user->getParentOp())) {
              // For block arguments, we need to check the initial value as well
              if (auto blockArg = dyn_cast<BlockArgument>(arg)) {
                auto initArg =
                    nestedForOp.getInitArgs()[blockArg.getArgNumber() - 1];
                if (Operation *def = initArg.getDefiningOp()) {
                  if (hasAsyncTaskId(def, asyncTaskId)) {
                    argIndices.insert(argIdx);
                    return;
                  }
                } else {
                  llvm_unreachable("Initial value should have a defining op");
                }
              }
            }

            // Skip control flow ops that are shared by all async tasks
            continue;
          }

          // If use is the initial value of ForOp argument.
          if (auto userFor = dyn_cast<scf::ForOp>(user)) {
            // For block arguments, we need to check the initial value as well.
            if (auto blockArg = dyn_cast<BlockArgument>(arg)) {
              auto initArg =
                  nestedForOp.getInitArgs()[blockArg.getArgNumber() - 1];
              if (Operation *def = initArg.getDefiningOp()) {
                if (hasAsyncTaskId(def, asyncTaskId)) {
                  argIndices.insert(argIdx);
                  return;
                }
              } else {
                // Recursive search the nested loop for the real users.
                // find corresponding arg of userFor
                Value userArg;
                for (auto item : llvm::enumerate(userFor.getInitArgs())) {
                  if (item.value() == arg) {
                    userArg = userFor.getRegionIterArg(item.index());
                    break;
                  }
                }
                if (userArg) {
                  dfs(userFor, userArg, argIdx);
                }
              }
            }
            // Skip control flow ops that are shared by all async tasks
            continue;
          }

          // Found a real user, the arg is needed
          if (user->getNumRegions() == 0) {
            argIndices.insert(argIdx);
            return;
          }

          // Iterate through all regions of the user operation
          for (auto &region : user->getRegions()) {
            for (auto regionArg : region.getArguments()) {
              if (arg == regionArg)
                dfs(nestedForOp, regionArg, argIdx);
            }
          }
        }
      };

  // Check dependency with DFS traversal for loop args and results.
  mlir::Block &block = forOp.getRegion().front();
  for (unsigned i = forOp.getNumInductionVars(); i < block.getNumArguments();
       ++i) {
    auto arg = block.getArgument(i);
    dfs(forOp, arg, i - forOp.getNumInductionVars());
  }
  for (unsigned i = 0; i < forOp.getNumResults(); ++i) {
    auto result = forOp->getResult(i);
    dfs(forOp, result, i);
  }

  SmallVector<unsigned> args(argIndices.begin(), argIndices.end());
  llvm::sort(args);
  return args;
}

Operation *SpecializeIfOp(scf::IfOp ifOp, IRMapping &mapping,
                          OpBuilderWithAsyncTaskIds &builder,
                          AsyncTaskId asyncTaskId) {
  LLVM_DEBUG({
    LDBG() << "specialize ifOp ";
    ifOp.dump();
  });

  // It is possible that we need to reduce the results. One example is that
  // the defining op for the yield operation is not for this taskId and
  // the defining op is not specialized, thus we should remove the result.
  // We need to update the result types correctly here.
  unsigned resultIdx = 0;
  SmallVector<unsigned> keptResultVec;
  if (!ifOp->getResultTypes().empty()) {
    for (Value yieldV : ifOp.thenYield().getOperands()) {
      // Check the defining op for the corresponding result.
      if (Operation *def = yieldV.getDefiningOp()) {
        if (hasAsyncTaskId(def, asyncTaskId)) {
          keptResultVec.push_back(resultIdx);
        }
      } else {
        assert(isa<BlockArgument>(yieldV) && "Unexpected yield value");
        auto bbArg = cast<BlockArgument>(yieldV);
        // Find transitive defining op for the block arg
        Operation *bbAargOwner = bbArg.getOwner()->getParentOp();
        if (auto forOp = dyn_cast<scf::ForOp>(bbAargOwner)) {
          // track initial value
          auto initArg = forOp.getInitArgs()[bbArg.getArgNumber() - 1];
          if (Operation *def = initArg.getDefiningOp()) {
            if (hasAsyncTaskId(def, asyncTaskId))
              keptResultVec.push_back(resultIdx);
          } else {
            llvm_unreachable("Initial value should have a defining op");
          }
        } else {
          llvm_unreachable("Unexpected block argument owner");
        }
      }
      ++resultIdx;
    }
  }

  SmallVector<Type> newResultTypes;
  for (auto idx : keptResultVec) {
    newResultTypes.push_back(ifOp->getResultTypes()[idx]);
  }
  auto newIfOp = builder.createWithAsyncTaskIds<scf::IfOp>(
      ifOp.getLoc(), newResultTypes, mapping.lookup(ifOp.getCondition()), true,
      ifOp.elseBlock());

  OpBuilderWithAsyncTaskIds ifBuilder(ifOp.getContext());
  ifBuilder.setAsynTaskIdsFromArray({asyncTaskId});

  // Handle thenRegion of this IfOp.
  ifBuilder.setInsertionPointToEnd(newIfOp.thenBlock());
  for (Operation &thenOp : ifOp.thenBlock()->getOperations()) {
    SpecializeOp(&thenOp, mapping, ifBuilder, asyncTaskId);
  }

  // Update yields
  auto updateYield = [&](scf::YieldOp yield, SmallVector<Value> &operands) {
    ifBuilder.setInsertionPoint(yield);
    ifBuilder.createWithAsyncTaskIds<scf::YieldOp>(yield.getLoc(), operands);
    yield.erase();
  };
  if (keptResultVec.size() < ifOp->getResultTypes().size()) {
    SmallVector<Value> ifYieldOperands;
    for (auto idx : keptResultVec) {
      ifYieldOperands.push_back(newIfOp.thenYield().getOperand(idx));
    }
    updateYield(newIfOp.thenYield(), ifYieldOperands);
  }

  // Handle elseRegion of the IfOp.
  if (ifOp.elseBlock()) {
    ifBuilder.setInsertionPointToEnd(newIfOp.elseBlock());
    for (Operation &elseOp : ifOp.elseBlock()->getOperations()) {
      SpecializeOp(&elseOp, mapping, ifBuilder, asyncTaskId);
    }
    if (keptResultVec.size() < ifOp->getResultTypes().size()) {
      SmallVector<Value> elseYieldOperands;
      for (auto idx : keptResultVec) {
        elseYieldOperands.push_back(newIfOp.elseYield().getOperand(idx));
      }
      updateYield(newIfOp.elseYield(), elseYieldOperands);
    }
  }

  unsigned newResIdx = 0;
  for (auto idx : keptResultVec) {
    mapping.map(ifOp.getResult(idx), newIfOp.getResult(newResIdx));
    ++newResIdx;
  }
  return newIfOp;
}

// Compare two async task id lists as sets.
static bool sameTaskIds(ArrayRef<AsyncTaskId> lhs, ArrayRef<AsyncTaskId> rhs) {
  if (lhs.size() != rhs.size())
    return false;
  SmallVector<AsyncTaskId> a(lhs.begin(), lhs.end());
  SmallVector<AsyncTaskId> b(rhs.begin(), rhs.end());
  std::sort(a.begin(), a.end());
  std::sort(b.begin(), b.end());
  return a == b;
}

// The code partition clones every op that carries async task ids into the
// region of each task that owns it. For scf.for and scf.if that is self
// contained: their results/operands carry the task ids of whoever consumes
// them, so a clone never refers to a value that was cloned into another task.
// An scf.while cannot work that way, because the loop hands its state from one
// iteration to the next *inside* its regions and the partition has no way to
// pass a scalar value from one task to another: the loop body has to be cloned
// as a whole into every task that owns the loop (see SpecializeWhileOp). That
// is only sound when the whole loop -- the op, both region terminators and
// every op nested in either region -- carries the same set of task ids.
// Reject the others here, before the function is rewritten, so that the caller
// can keep the kernel on the non-WS path instead of partitioning it wrongly.
bool isSupportedForCodePartition(triton::FuncOp &funcOp) {
  bool supported = true;
  funcOp.walk([&](scf::WhileOp whileOp) {
    auto whileTaskIds = getAsyncTaskIds(whileOp);
    // Without task ids the loop is not part of a partition; it stays where it
    // is. (An op inside it that got partitioned would have labelled the loop.)
    if (whileTaskIds.empty())
      return;
    whileOp->walk([&](Operation *op) {
      if (sameTaskIds(getAsyncTaskIds(op), whileTaskIds))
        return;
      LDBG() << "scf.while is not partitioned as a whole, body op " << *op;
      supported = false;
    });
  });
  return supported;
}

// Clone one scf.while per task. The loop interface (inits, results, region
// arguments) is shared by every task that owns the loop, so each clone keeps
// it unchanged and only the ops of the two regions are cloned per task -- the
// terminator of each region is rebuilt from the remapped operands. See
// isSupportedForCodePartition for why a mixed-task loop is rejected earlier
// instead of being handled here.
Operation *SpecializeWhileOp(scf::WhileOp whileOp, IRMapping &mapping,
                             OpBuilderWithAsyncTaskIds &builder,
                             AsyncTaskId asyncTaskId) {
  LLVM_DEBUG({
    LDBG() << "specialize whileOp ";
    whileOp.dump();
  });

  SmallVector<Value> newInits;
  for (Value init : whileOp.getInits())
    newInits.push_back(mapping.lookupOrDefault(init));
  SmallVector<Type> newResultTypes(whileOp->getResultTypes());

  MLIRContext *context = whileOp.getContext();
  SmallVector<AsyncTaskId> taskIds{asyncTaskId};

  // Clone the ops of one region that belong to this task and rebuild that
  // region's terminator: scf.condition for the "before" region, scf.yield
  // (which feeds the loop back) for the "after" region.
  auto cloneRegion = [&](Block *oldBlock, Block *newBlock, bool isBefore) {
    OpBuilderWithAsyncTaskIds regionBuilder(context);
    regionBuilder.setInsertionPointToEnd(newBlock);
    regionBuilder.setAsynTaskIdsFromArray(taskIds);
    for (Operation &op : oldBlock->without_terminator())
      SpecializeOp(&op, mapping, regionBuilder, asyncTaskId);

    Operation *terminator = oldBlock->getTerminator();
    SmallVector<Value> operands;
    for (Value operand : terminator->getOperands())
      operands.push_back(mapping.lookupOrDefault(operand));
    if (isBefore)
      regionBuilder.createWithAsyncTaskIds<scf::ConditionOp>(
          terminator->getLoc(), operands.front(),
          ValueRange(operands).drop_front());
    else
      regionBuilder.createWithAsyncTaskIds<scf::YieldOp>(terminator->getLoc(),
                                                         operands);
  };

  // The region arguments only exist while the bodies are built, so map them
  // from inside the body builders.
  auto newWhileOp = builder.createWithAsyncTaskIds<scf::WhileOp>(
      whileOp.getLoc(), newResultTypes, newInits,
      [&](OpBuilder &b, Location, ValueRange beforeArgs) {
        auto oldArgs = whileOp.getBeforeArguments();
        for (unsigned i = 0; i < beforeArgs.size(); ++i)
          mapping.map(oldArgs[i], beforeArgs[i]);
        cloneRegion(whileOp.getBeforeBody(), b.getBlock(), /*isBefore=*/true);
      },
      [&](OpBuilder &b, Location, ValueRange afterArgs) {
        auto oldArgs = whileOp.getAfterArguments();
        for (unsigned i = 0; i < afterArgs.size(); ++i)
          mapping.map(oldArgs[i], afterArgs[i]);
        cloneRegion(whileOp.getAfterBody(), b.getBlock(), /*isBefore=*/false);
      });

  for (unsigned i = 0; i < whileOp.getNumResults(); ++i)
    mapping.map(whileOp.getResult(i), newWhileOp.getResult(i));
  return newWhileOp;
}

Operation *SpecializeForOp(scf::ForOp forOp, IRMapping &mapping,
                           OpBuilderWithAsyncTaskIds &builder,
                           AsyncTaskId asyncTaskId) {
  // Create newForOp for each task Id.
  auto usedArgs = collectBlockArgsForTask(forOp, asyncTaskId);

  // Prepare newLoopArgs.
  SmallVector<Value> newLoopArgs;
  for (unsigned argNumber : usedArgs) {
    auto arg = forOp.getInitArgs()[argNumber];
    auto newArg = mapping.lookupOrDefault(arg);
    assert(newArg && "Unexpected missing mapping");
    newLoopArgs.push_back(newArg);
  }

  // Prepare loop bounds.
  auto newLowerBound = mapping.lookupOrDefault(forOp.getLowerBound());
  auto newUpperBound = mapping.lookupOrDefault(forOp.getUpperBound());
  auto newStep = mapping.lookupOrDefault(forOp.getStep());

  // Create newForOp.
  auto newForOp = builder.createWithAsyncTaskIds<scf::ForOp>(
      forOp.getLoc(), newLowerBound, newUpperBound, newStep, newLoopArgs);
  if (forOp->getAttr("tt.loop_schedule"))
    newForOp->setAttr("tt.loop_schedule", forOp->getAttr("tt.loop_schedule"));

  // Initialize Value mapping from forOp to newForOp
  mapping.map(forOp.getInductionVar(), newForOp.getInductionVar());
  for (unsigned i = 0; i < usedArgs.size(); ++i) {
    auto oldArg = forOp.getRegionIterArgs()[usedArgs[i]];
    auto newArg = newForOp.getRegionIterArgs()[i];
    mapping.map(oldArg, newArg);
  }

  // Recursively clone all operations with this asyncTaskId to newForOp.
  OpBuilderWithAsyncTaskIds forBuilder(forOp.getContext());
  forBuilder.setAsynTaskIdsFromArray({asyncTaskId});
  forBuilder.setInsertionPointToStart(newForOp.getBody());
  for (Operation &op : forOp.getBody()->without_terminator()) {
    SpecializeOp(&op, mapping, forBuilder, asyncTaskId);
  }

  // Create YieldOp for newForOp.
  auto yieldOp = llvm::cast<scf::YieldOp>(forOp.getBody()->getTerminator());
  SmallVector<Value> newYieldOperands;
  for (unsigned i : usedArgs)
    newYieldOperands.push_back(mapping.lookup(yieldOp.getOperand(i)));

  bool createNewYield = true;
  if (newForOp.getBody()->mightHaveTerminator()) {
    auto initialYield =
        llvm::cast<scf::YieldOp>(newForOp.getBody()->getTerminator());
    if (newYieldOperands.size() == 0) {
      setAsyncTaskIds(initialYield, {asyncTaskId});
      createNewYield = false;
    }
  }
  if (createNewYield) {
    auto newYieldOp =
        forBuilder.create<scf::YieldOp>(yieldOp.getLoc(), newYieldOperands);
    setAsyncTaskIds(newYieldOp, {asyncTaskId});
  }

  // Replace results of forOp with results of newForOp.
  for (unsigned i = 0; i < usedArgs.size(); ++i) {
    auto oldResult = forOp.getResult(usedArgs[i]);
    auto newResult = newForOp.getResult(i);
    mapping.map(oldResult, newResult);
  }

  return newForOp;
}

Operation *SpecializeOp(Operation *op, IRMapping &mapping,
                        OpBuilderWithAsyncTaskIds &builder,
                        AsyncTaskId asyncTaskId) {
  auto taskIds = getAsyncTaskIds(op);
  // yieldOp are sometimes implicit, meaning they do not necessarily have a task
  // id, but they should be shared by all async tasks.
  if (!hasAsyncTaskId(op, asyncTaskId) && !isa<scf::YieldOp>(op))
    return nullptr;

  if (op->getNumRegions() == 0) {
    Operation *newOp = builder.clone(*op, mapping);
    setAsyncTaskIds(newOp, asyncTaskId);
    for (unsigned i = 0; i < op->getNumResults(); ++i)
      mapping.map(op->getResult(i), newOp->getResult(i));
    return newOp;
  } else {
    if (auto ifOp = dyn_cast<scf::IfOp>(op)) {
      return SpecializeIfOp(ifOp, mapping, builder, asyncTaskId);
    } else if (auto forOp = dyn_cast<scf::ForOp>(op)) {
      return SpecializeForOp(forOp, mapping, builder, asyncTaskId);
    } else if (auto whileOp = dyn_cast<scf::WhileOp>(op)) {
      return SpecializeWhileOp(whileOp, mapping, builder, asyncTaskId);
    } else if (auto reduceOp = dyn_cast<triton::ReduceOp>(op)) {
      Operation *newOp = builder.clone(*op, mapping);
      // recursively set async task ids for child ops
      newOp->walk(
          [&](Operation *childOp) { setAsyncTaskIds(childOp, asyncTaskId); });
      for (unsigned i = 0; i < op->getNumResults(); ++i)
        mapping.map(op->getResult(i), newOp->getResult(i));
      return newOp;
    } else {
      llvm_unreachable("Unexpected Op with regions");
    }
  }

  return nullptr;
}

void specializeRegion(triton::FuncOp funcOp, int32_t numWarps,
                      unsigned requestedRegisters) {
  LLVM_DEBUG({ LDBG() << "Start specializing region"; });

  MLIRContext *context = funcOp.getContext();
  OpBuilder builder(context);
  auto loc = funcOp.getLoc();

  // Collect original operations
  SmallVector<Operation *> opList;
  for (auto &block : funcOp.getBody().getBlocks()) {
    for (Operation &op : block.getOperations()) {
      auto taskIds = getAsyncTaskIds(&op);
      if (!taskIds.empty())
        opList.push_back(&op);
    }
  }

  LLVM_DEBUG({
    LDBG() << "ops to be specialized: ";
    for (Operation *op : opList) {
      op->dump();
    }
  });

  // Create GetAsyncTaskIdOp.
  Block *lastBlock = &funcOp.getBody().back();
  auto returnOp = llvm::cast<triton::ReturnOp>(lastBlock->getTerminator());
  builder.setInsertionPoint(returnOp);

  // Instead of a new IfOp for each task, we create one partitionRegion.
  auto nTaskIds = getNestedAsyncTaskIds(funcOp);
  SmallVector<int32_t> partitionNumWarps;
  for (AsyncTaskId asyncTaskId : nTaskIds) {
    if (asyncTaskId == 0)
      continue;
    partitionNumWarps.push_back(numWarps);
  }

  // The partitioner assigns a task id only to the ops that produce or consume
  // partition state, so an op with no id at all can survive inside a region
  // that is about to be cloned -- a side-effecting op whose results no
  // partition depends on, such as tt.print. SpecializeOp clones only ops that
  // carry the target task id and the original op is erased below, so such an op
  // would be dropped silently. Run it in the same tasks as the other ops of its
  // region instead (every task when the region has none), mirroring upstream's
  // assignPartitionsForOpsWithNoUse: an op no partition depends on belongs to
  // every partition the region is split into. A result consumed by a
  // partitioned op already carries its id -- the backward propagation meets a
  // result's lattice into all of its operands -- so this only adds ids to code
  // that no partition uses.
  DenseMap<Block *, SmallVector<AsyncTaskId>> blockTaskIds;
  for (Operation *op : opList) {
    op->walk([&](Block *block) {
      if (blockTaskIds.count(block))
        return;
      SmallVector<AsyncTaskId> &taskIds = blockTaskIds[block];
      for (Operation &nestedOp : *block)
        for (AsyncTaskId taskId : getAsyncTaskIds(&nestedOp))
          if (!llvm::is_contained(taskIds, taskId))
            taskIds.push_back(taskId);
    });
  }
  for (auto &[block, taskIds] : blockTaskIds) {
    ArrayRef<AsyncTaskId> ids = taskIds.empty()
                                    ? ArrayRef<AsyncTaskId>(nTaskIds)
                                    : ArrayRef<AsyncTaskId>(taskIds);
    for (Operation &nestedOp : *block)
      if (getAsyncTaskIds(&nestedOp).empty())
        setAsyncTaskIds(&nestedOp, ids);
  }

  // Search insertion point
  Operation *insertionPoint = returnOp;
  while (insertionPoint->getPrevNode() &&
         getNestedAsyncTaskIds(insertionPoint->getPrevNode()).empty()) {
    insertionPoint = insertionPoint->getPrevNode();
  }

  ArrayRef<Type> dummyTypes;
  ImplicitLocOpBuilder impB(opList[0]->getLoc(), opList[0]);
  impB.setInsertionPoint(insertionPoint);
  auto wsOp = impB.create<ttg::WarpSpecializeOp>(dummyTypes, partitionNumWarps,
                                                 nTaskIds.size() - 1);

  // Clone all operations into the corresponding if blocks. If the operation
  // has multiple taskIds, it will be cloned for multiple if blocks.
  // If the original code has an IfOp, we should only clone its body with
  // the right asyncTaskId, instead of cloning the IfOp. Handle producer WG.
  {
    AsyncTaskId asyncTaskId = nTaskIds[0];
    OpBuilderWithAsyncTaskIds taskBuilder(context);
    taskBuilder.setAsynTaskIdsFromArray({asyncTaskId});
    Block *defaultBlock = impB.createBlock(&wsOp.getDefaultRegion());
    taskBuilder.setInsertionPointToStart(defaultBlock);

    IRMapping mapping;
    for (Operation *op : opList) {
      SpecializeOp(op, mapping, taskBuilder, asyncTaskId);
    }
    taskBuilder.create<ttg::WarpYieldOp>(loc, SmallVector<Value>{});
  }

  // compute warps
  unsigned idx = 1;
  SmallVector<int32_t> estRegUsage;
  for (Region *region : wsOp.getPartitionRegions()) {
    AsyncTaskId asyncTaskId = nTaskIds[idx];
    OpBuilderWithAsyncTaskIds taskBuilder(context);
    taskBuilder.setAsynTaskIdsFromArray({asyncTaskId});
    LDBG() << "region idx " << idx << " size " << nTaskIds.size();
    ++idx;
    Block *partitionBlock = impB.createBlock(region);
    taskBuilder.setInsertionPointToStart(partitionBlock);

    IRMapping mapping;
    for (Operation *op : opList) {
      SpecializeOp(op, mapping, taskBuilder, asyncTaskId);
    }
    taskBuilder.create<ttg::WarpReturnOp>(loc);
    auto regAlloc =
        scanRegUsage(partitionBlock, asyncTaskId, requestedRegisters);
    estRegUsage.push_back(regAlloc);
  }

  // The default region doesn't request registers.
  wsOp.setRequestedRegisters(estRegUsage);

  // The capture set is the same for every partition region, so now find the
  // captures and thread them in to the regions.
  SetVector<Value> captures;
  getUsedValuesDefinedAbove(wsOp.getPartitionOpHolder(), captures);
  for (Value capture : captures) {
    // Rematerialize constants.
    if (capture.getDefiningOp() &&
        capture.getDefiningOp()->hasTrait<OpTrait::ConstantLike>()) {
      for (Region *region : wsOp.getPartitionRegions()) {
        impB.setInsertionPointToStart(&region->front());
        Value copy = impB.clone(*capture.getDefiningOp())->getResult(0);
        replaceAllUsesInRegionWith(capture, copy, *region);
      }
      continue;
    }

    if (isa<RankedTensorType>(capture.getType())) {
      mlir::emitWarning(capture.getLoc(),
                        "FIXME: capturing tensor values into warp "
                        "partitions is not supported");
    }
    triton_gcu::compat::insertWsCapture(wsOp, capture);
    for (Region *region : wsOp.getPartitionRegions()) {
      // Does this include default region?
      BlockArgument arg =
          region->addArgument(capture.getType(), capture.getLoc());
      replaceAllUsesInRegionWith(capture, arg, *region);
    }
  }

  LLVM_DEBUG({
    LDBG() << "\nWith task Id checks";
    funcOp.dump();
  });

  // Remove original operations that have been cloned in reverse order.
  for (auto it = opList.rbegin(); it != opList.rend(); ++it) {
    Operation *op = *it;
    LLVM_DEBUG({
      LDBG() << "erasing op ";
      op->dump();
    });
    // For debugging purposes, check to see if the original op is still in use.
    for (unsigned i = 0; i < op->getNumResults(); ++i) {
      for (Operation *user : op->getResult(i).getUsers()) {
        (void)user;
        LLVM_DEBUG({
          LDBG() << "op has use, should be removed";
          user->dump();
        });
      }
    }
    op->erase();
  }
}

} // namespace mlir::triton::gcu
