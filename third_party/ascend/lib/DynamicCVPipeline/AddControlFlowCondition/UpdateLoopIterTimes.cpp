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

#include "ascend/include/DynamicCVPipeline/AddControlFlowCondition/UpdateLoopIterTimes.h"
#include "ascend/include/DynamicCVPipeline/AddControlFlowCondition.h"
#include "bishengir/Dialect/HIVM/IR/HIVM.h"
#include "bishengir/Dialect/Scope/IR/Scope.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/Support/Debug.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"

static constexpr const char *DEBUG_TYPE = "UpdateLoopIterTimes";
#define DBGS() (llvm::dbgs() << '[' << DEBUG_TYPE << "] ")
#define LDBG(...) LLVM_DEBUG(DBGS() << __VA_ARGS__ << "\n")

using namespace llvm;
using namespace mlir;
using namespace triton;
using namespace hivm;

// Find the IfOp index that an operation belongs to
// Uses isAncestor to check if the operation is within the IfOp's region
static int findIfOpIndexInList(Operation *op, SmallVector<scf::IfOp> &ifOps,
                               llvm::DenseMap<Operation *, int> &ifOpIndex)
{
  for (scf::IfOp ifOp : ifOps) {
    if (ifOp->isAncestor(op)) {
      return ifOpIndex[ifOp.getOperation()];
    }
  }
  return -1;
}

// calculateFactor computes the buffer factor needed for loop iteration extension
// Core idea: if producer ifOp (n) and consumer ifOp (m) have dependency,
// we need (m - n + 1) buffers to support the loop extension
// Returns {maxRequiredBuffers, x} for computing: ceil(original_iter_count * requiredBuffers / x) + ifCount
std::pair<int, int> UpdateLoopIterTimesPass::calculateFactor(scf::ForOp forOp)
{
  int maxRequiredBuffers = 1;
  int maxX = 1;

  // Step1: Collect all IfOps with ssbuffer.if attribute in this for loop and build index
  // Index represents execution order (smaller index = earlier execution)
  SmallVector<scf::IfOp> ifOps;
  DenseMap<Operation *, int> ifOpIndex;
  int index = 1;

  int ret = 0;
  forOp.walk([&](Operation* op) {
    if (op->hasAttr("ssbuffer.if")) {
      auto ifOp = dyn_cast<scf::IfOp>(op);
      if (!ifOp) {
        ret = 1;
        LDBG("ssbuffer.if attribute is not allocated on ifOp!");
        return WalkResult::interrupt();
      }
      ifOps.push_back(ifOp);
      ifOpIndex[ifOp.getOperation()] = index++;
    }
    return WalkResult::advance();
  });
  if (ret == -1) {
    return {-1, -1};
  }

  if (ifOps.empty()) {
    LDBG("mainloop do not contains ifblocks!");
    return {-1, -1};
  }

  // No dependency, multiple factor = 1
  if (!info->intraCoreDependentMap.count(forOp)) {
    return {1, 1};
  }
  auto &deps = info->intraCoreDependentMap[forOp];
  if (deps.empty()) {
    return {1, 1};
  }

  // Step2: Iterate all dependencies and calculate required buffer count
  for (auto &entry : deps) {
    Value consumerResult = entry.first;           // Consumer result value
    SmallVector<Value> producerBuffers = entry.second;  // Producer buffer list
    int x = producerBuffers.size();               // Producer buffer count

    // Get the defining operation of consumer result
    Operation *consumerDefOp = consumerResult.getDefiningOp();
    if (!consumerDefOp) {
      LDBG("consumerResult do not have the defOp!");
      return {-1, -1};
    }

    // Find the IfOp index that consumer belongs to (m)
    int m = findIfOpIndexInList(consumerDefOp, ifOps, ifOpIndex);
    if (m == -1) {
      LDBG("Can not find the consumerDefOp in any ifOps!\nconsumerDefOp: " << *consumerDefOp);
      return {-1, -1};
    }

    // Step3: Find the IfOp index that producer belongs to (n)
    // Producer is the operation that uses producer buffer, i.e., bufferization.materialize_in_destination
    // This op is the production behavior op, with producer buffer as its input
    if (producerBuffers.empty()) {
      LDBG("consumer do not have the producerBuffers!");
      return {-1, -1};
    }

    int producerIfOpIndex = -1;
    for (Value buffer : producerBuffers) {
      for (Operation *user : buffer.getUsers()) {
        if (isa<mlir::bufferization::MaterializeInDestinationOp>(user) || isa<hivm::CopyOp>(user)) {
          producerIfOpIndex = findIfOpIndexInList(user, ifOps, ifOpIndex);
          if (producerIfOpIndex == -1) {
            LDBG("Can not find the producerBuffers in any ifOps\nproducerBuffers: " << *user);
            return {-1, -1};
          }
          break;
        }
      }
      // any producerBuffer found in a ifblock means all producerBuffers in that ifblock
      if (producerIfOpIndex != -1) {
        break;
      }
    }

    // If cannot find the IfOp producer belongs to, skip this dependency
    if (producerIfOpIndex == -1) {
      LDBG("All producerBuffers are not found in any ifOps");
      return {-1, -1};
    }

    int n = producerIfOpIndex;

    // Step4: If consumer is after producer (m > n), calculate required buffer count
    // m - n + 1 represents the buffer count needed to cover this distance
    if (m <= n) {
      LDBG("producer is after the comsumer!");
      return {-1, -1};
    }
    int requiredBuffers = m - n + 1;

    // Step5: Update max value using fraction comparison to avoid precision issues
    // Comparing requiredBuffers/maxX vs maxRequiredBuffers/x is equivalent to
    // comparing requiredBuffers * maxX vs maxRequiredBuffers * x
    if (requiredBuffers * maxX > maxRequiredBuffers * x) {
      maxRequiredBuffers = requiredBuffers;
      maxX = x;
    }
  }

  return {maxRequiredBuffers, maxX};
}

// Part 1: Compute new loop upper bound and create arithmetic ops
Value UpdateLoopIterTimesPass::computeNewLoopUpperBound(
    OpBuilder &builder, Location loc, scf::ForOp forOp,
    int ifCount, int requiredBuffers, int x)
{
  Value originalLowerBound = forOp.getLowerBound();
  Value originalUpperBound = forOp.getUpperBound();
  Value originalStep = forOp.getStep();
  Type ubType = originalStep.getType();

  // Helper lambda to create constant value based on type
  auto createConstant = [&](int val) -> Value {
    if (ubType.isIndex()) {
      return builder.create<arith::ConstantIndexOp>(loc, val);
    } else if (auto intType = dyn_cast<IntegerType>(ubType)) {
      return builder.create<arith::ConstantIntOp>(loc, intType, val);
    } else {
      auto indexVal = builder.create<arith::ConstantIndexOp>(loc, val);
      return builder.create<arith::IndexCastOp>(loc, ubType, indexVal);
    }
  };

  // Helper lambda to create ceil division op based on type
  auto createCeilDiv = [&](Value lhs, Value rhs) -> Value {
    if (ubType.isIndex()) {
      return builder.create<arith::CeilDivUIOp>(loc, lhs, rhs);
    } else if (auto intType = dyn_cast<IntegerType>(ubType)) {
      if (intType.isSigned()) {
        return builder.create<arith::CeilDivSIOp>(loc, lhs, rhs);
      } else {
        return builder.create<arith::CeilDivUIOp>(loc, lhs, rhs);
      }
    } else {
      return builder.create<arith::CeilDivUIOp>(loc, lhs, rhs);
    }
  };

  // Create constant values
  Value ifCountValue = createConstant(ifCount);
  Value requiredBuffersValue = createConstant(requiredBuffers);
  Value xValue = createConstant(x);

  // Calculate: rangeDiff = upperBound - lowerBound
  Value rangeDiff = builder.create<arith::SubIOp>(loc, originalUpperBound, originalLowerBound);

  // Calculate: iterCount = ceil(rangeDiff / step)
  Value iterCount = createCeilDiv(rangeDiff, originalStep);

  // Calculate: scaledIterCount = iterCount * requiredBuffers
  Value scaledIterCount = builder.create<arith::MulIOp>(loc, iterCount, requiredBuffersValue);

  // Calculate: ceiledScaledIterCount = ceil(scaledIterCount / x)
  Value ceiledScaledIterCount = createCeilDiv(scaledIterCount, xValue);

  // Calculate: newIterCount = ceiledScaledIterCount + ifCount
  Value newIterCount = builder.create<arith::AddIOp>(loc, ceiledScaledIterCount, ifCountValue);

  // Calculate: totalSteps = step * newIterCount
  Value totalSteps = builder.create<arith::MulIOp>(loc, originalStep, newIterCount);

  // Calculate: newUpperBound = lowerBound + totalSteps
  Value newUpperBound = builder.create<arith::AddIOp>(loc, originalLowerBound, totalSteps);

  return newUpperBound;
}

// Part 2: Clone for loop with new upper bound
scf::ForOp UpdateLoopIterTimesPass::cloneForOpWithNewUpperBound(
    OpBuilder &builder, Location loc, scf::ForOp oldForOp, Value newUpperBound,
    IRMapping &mapper)
{
  Value originalLowerBound = oldForOp.getLowerBound();
  Value originalStep = oldForOp.getStep();

  SmallVector<Value> newInitArgs(oldForOp.getInitArgs().begin(), oldForOp.getInitArgs().end());

  auto newForOp = builder.create<scf::ForOp>(loc, originalLowerBound, newUpperBound, originalStep, newInitArgs);

  // Copy attributes from old forOp
  for (auto &attr : oldForOp->getAttrs()) {
    newForOp->setAttr(attr.getName(), attr.getValue());
  }

  // Map induction variable and iter args
  mapper.map(oldForOp.getInductionVar(), newForOp.getInductionVar());

  for (auto [oldArg, newArg] : llvm::zip(oldForOp.getRegionIterArgs(), newForOp.getRegionIterArgs())) {
    mapper.map(oldArg, newArg);
  }

  Block *oldBlock = oldForOp.getBody();
  Block *newBlock = newForOp.getBody();

  // Replace block arguments
  unsigned totalArgs = oldBlock->getNumArguments();
  for (unsigned i = 0; i < totalArgs; ++i) {
    BlockArgument oldArg = oldBlock->getArgument(i);
    BlockArgument newArg = newBlock->getArgument(i);
    oldArg.replaceAllUsesWith(newArg);
  }

  Operation *oldTerminator = oldBlock->getTerminator();

  // Clone operations in the loop body
  builder.setInsertionPointToStart(newBlock);
  for (Operation &op : llvm::make_early_inc_range(oldBlock->without_terminator())) {
    builder.clone(op, mapper);
  }

  // Create new yield operation
  auto oldYield = cast<scf::YieldOp>(oldTerminator);
  SmallVector<Value> newYieldOperands;
  for (unsigned i = 0; i < oldYield.getNumOperands(); ++i) {
    newYieldOperands.push_back(mapper.lookupOrDefault(oldYield.getOperand(i)));
  }

  builder.setInsertionPointToEnd(newBlock);
  builder.create<scf::YieldOp>(loc, newYieldOperands);

  // Handle results replacement
  unsigned numOriginalResults = oldForOp.getNumResults();
  if (numOriginalResults > 0) {
    SmallVector<Value> originalResults;
    for (unsigned i = 0; i < numOriginalResults; ++i) {
      originalResults.push_back(newForOp.getResult(i));
    }
    oldForOp.replaceAllUsesWith(originalResults);
  }

  return newForOp;
}

// Part 3: Update cntArgs mapping after cloning
int UpdateLoopIterTimesPass::updateCntArgsAfterClone(
    scf::ForOp oldForOp, IRMapping &mapper,
    SmallVector<scf::IfOp> &ifOpsInThisFor)
{
  // Update cntArgs for each old ifOp
  for (scf::IfOp oldIfOp : ifOpsInThisFor) {
    scf::IfOp newIfOp = dyn_cast<scf::IfOp>(mapper.lookupOrDefault(oldIfOp));
    if (!newIfOp)
      continue;

    Value oldCntVal = info->cntArgs[oldIfOp];
    Value newCntVal = mapper.lookupOrDefault(oldCntVal);
    LDBG("updating CntArgs After Clone...");
    LDBG("oldCntVal: " << oldCntVal);
    LDBG("newCntVal: " << newCntVal);

    // Remove old entry and add new entry
    info->cntArgs.erase(oldIfOp);
    info->cntArgs[newIfOp] = newCntVal;
  }
  return 0;
}

// Extend for loop iteration count
scf::ForOp UpdateLoopIterTimesPass::extendForOpIterationCount(
    scf::ForOp oldForOp, int ifCount, int requiredBuffers, int x,
    IRMapping &mapper, SmallVector<scf::IfOp> &ifOpsInThisFor)
{
  OpBuilder builder(oldForOp);
  Location loc = oldForOp.getLoc();

  // Part 1: Compute new loop upper bound
  Value newUpperBound = computeNewLoopUpperBound(builder, loc, oldForOp, ifCount, requiredBuffers, x);
  if (!newUpperBound) {
      LDBG("computeNewLoopUpperBound failed!");
      return nullptr;
  }

  // Part 2: Clone for loop with new upper bound
  scf::ForOp newForOp = cloneForOpWithNewUpperBound(builder, loc, oldForOp, newUpperBound, mapper);
  if (!newForOp) {
      LDBG("cloneForOpWithNewUpperBound failed!");
      return nullptr;
  } else {
    LDBG("cloneForOpWithNewUpperBound Success!");
    LDBG("new ForOp: " << newForOp);
    LDBG("old ForOp: " << oldForOp);
  }

  // Part 3: Update cntArgs mapping
  if (updateCntArgsAfterClone(oldForOp, mapper, ifOpsInThisFor) != 0) {
      LDBG("updateCntArgsAfterClone failed!");
      return nullptr;
  }

  return newForOp;
}

// step4: Replace loop counter by if blocks' counter
// Traverse each mainloop, find ifOp with ssbuffer.if attribute inside,
// and replace the mainloop's induction variable with the counter in cntArgs
int UpdateLoopIterTimesPass::replaceForOpCounterInIfOps()
{
  int ret = 0;
  // Traverse all mainloops in the module
  getOperation().walk([&](Operation* op) {
    if (op->hasAttr("ssbuffer.main_loop")) {
      auto forOp = dyn_cast<scf::ForOp>(op);
      if (!forOp) {
        ret = -1;
        LDBG("Do not support other mainloop except forOp!");
        return WalkResult::interrupt();
      }

      Value indVar = forOp.getInductionVar();

      // Find all ifOps with ssbuffer.if attribute inside this mainloop
      forOp.walk([&](scf::IfOp ifOp) {
        if (ifOp->hasAttr("ssbuffer.if")) {
          if (!info->cntArgs.count(ifOp)) {
            LDBG("ifblock has no counter in cntArgs");
            ret = -1;
            return WalkResult::interrupt();
          }

          Value cntVal = info->cntArgs[ifOp];

          // Replace all uses of induction variable in the ifOp with cntVal
          ifOp.walk([&](Operation *op) {
            for (OpOperand &operand : op->getOpOperands()) {
              if (operand.get() == indVar) {
                operand.set(cntVal);
              }
            }
          });
        }
        return mlir::WalkResult::advance();
      });
    }
    return mlir::WalkResult::advance();
  });
  return ret;
}

// step1: Get mainloop id to loop operation mapping, separated into cube and vector
int UpdateLoopIterTimesPass::GetMainLoopIdToLoopOpMap(
    ModuleOp module, DenseMap<int, SmallVector<Operation *>> &cmap,
    DenseMap<int, SmallVector<Operation *>> &vmap)
{
  cmap.clear();
  vmap.clear();
  int ret = 0;

  module.walk([&](scope::ScopeOp scopeOp) {
    // Determine if it's CUBE or VECTOR
    bool isCube = false;
    bool isVector = false;
    if (scopeOp->hasAttr("hivm.tcore_type")) {
      auto attr = scopeOp->getAttr("hivm.tcore_type");
      auto aiCAttr = hivm::TCoreTypeAttr::get(scopeOp->getContext(), hivm::TCoreType::CUBE);
      auto aiVAttr = hivm::TCoreTypeAttr::get(scopeOp->getContext(), hivm::TCoreType::VECTOR);
      if (attr == aiCAttr) {
        isCube = true;
      } else if (attr == aiVAttr) {
        isVector = true;
      }
    }

    if (!(isCube || isVector)) {
      ret = -1;
      LDBG("mlir do not processed by split mix kernel!");
      return mlir::WalkResult::interrupt();
    }

    // Walk for loops inside the scope
    scopeOp.walk([&](Operation* op) {
      if (op->hasAttr("ssbuffer.main_loop")) {
        auto forOp = dyn_cast<scf::ForOp>(op);
        if (!forOp) {
          ret = -1;
          LDBG("do not surpport other loop op temprarily!");
          return mlir::WalkResult::interrupt();
        }

        auto mainLoopId = forOp->getAttrOfType<IntegerAttr>("ssbuffer.main_loop");
        if (mainLoopId) {
          int id = mainLoopId.getInt();
          if (isCube) {
            cmap[id].push_back(forOp.getOperation());
          } else if (isVector) {
            vmap[id].push_back(forOp.getOperation());
          }
        }
      }
      return mlir::WalkResult::advance();
    });
    return mlir::WalkResult::advance();
  });

  return ret;
}

// Input: loopMap： {mainloop_id: [loopOp1, loopOp2], ...}
// Output: infoMap: {loopOp: extraIterationTimesInfo, ...}
int UpdateLoopIterTimesPass::ComputeMainLoopTimes(
    DenseMap<int, SmallVector<Operation *>> &loopMap,
    DenseMap<Operation *, IterationTimesInfo> &infoMap)
{
  for (auto &entry : loopMap) {
    for (Operation *loopOp : entry.second) {
      scf::ForOp forOp = dyn_cast<scf::ForOp>(loopOp);
      if (!forOp) {
        LDBG("currently only support forOp!");
        return -1;
      }

      // Collect if operations in this loop
      IterationTimesInfo iterInfo;
      if (info->cntArgs.empty()) {
        LDBG("cntArgs is empty, no ifblock is contained!");
        return -1;
      }
      for (auto &[ifOp, cntVal] : info->cntArgs) {
        if (ifOp->hasAttr("ssbuffer.if")) {
          auto parentOp = ifOp->getParentOp();
          if (parentOp->hasAttr("ssbuffer.main_loop") && isa<scf::ForOp>(parentOp)) {
            if (parentOp == forOp.getOperation()) {
              iterInfo.ifCount++;
              iterInfo.ifOpsInThisFor.push_back(ifOp);
            }
          } else {
            LDBG("Get wrong ifblock structure: ifblock's parentOp is not mainloop!");
            LDBG("ifblock Op: " << ifOp);
            LDBG("Parent Op: " << *parentOp);
            return -1;
          }
        } else {
          LDBG("ifOp in cntArgs does not contain ssbuffer.if Attribute\n ifOp: " << ifOp);
          return -1;
        }
      }

      // Calculate factor
      auto [requiredBuffers, x] = calculateFactor(forOp);
      if (requiredBuffers == -1 || x == -1) {
        LDBG("calculateFactor failed!");
        return -1;
      }
      iterInfo.requiredBuffers = requiredBuffers;
      iterInfo.x = x;
      infoMap[loopOp] = iterInfo;
    }
  }
  return 0;
}

int UpdateLoopIterTimesPass::collectForOpsAndUpdateMax(
    DenseMap<int, SmallVector<Operation *>> &map,
    int id,
    SmallVector<Operation *> &allForOps,
    int &maxIfCount,
    int &maxRequiredBuffers,
    int &maxX,
    DenseMap<Operation *, IterationTimesInfo> &infoMap)
{
  if (map.count(id)) {
    for (Operation *loopOp : map[id]) {
      allForOps.push_back(loopOp);
      if (infoMap.count(loopOp)) {
        IterationTimesInfo &iterInfo = infoMap[loopOp];
        if (iterInfo.ifCount > maxIfCount) {
          maxIfCount = iterInfo.ifCount;
        }
        // requiredBuffers/x > maxRequiredBuffers/maxX =
        // requiredBuffers * maxX > maxRequiredBuffers * x
        if (iterInfo.requiredBuffers * maxX > maxRequiredBuffers * iterInfo.x) {
          maxRequiredBuffers = iterInfo.requiredBuffers;
          maxX = iterInfo.x;
        }
      } else {
        LDBG("mainloop Op: " << *loopOp << " do not include in infoMap!");
        return -1;
      }
    }
  }
  return 0;
}

int UpdateLoopIterTimesPass::UpdateForLoopIteration(
    DenseMap<int, SmallVector<Operation *>> &cmap,
    DenseMap<int, SmallVector<Operation *>> &vmap,
    DenseMap<Operation *, IterationTimesInfo> &infoMap)
{
  int ret = 0;
  // Collect all mainloop ids
  DenseSet<int> allIds;
  for (auto &entry : cmap) {
    allIds.insert(entry.first);
  }
  for (auto &entry : vmap) {
    allIds.insert(entry.first);
  }

  // For each mainloop id, calculate max values and update all related loops
  SmallVector<Operation *> allForOps;
  for (int id : allIds) {
    int maxIfCount = 0;
    int maxRequiredBuffers = 1;
    int maxX = 1;

    // Collect all loops with same id from cmap and vmap
    SmallVector<Operation *> sameIdForOps;
    ret = collectForOpsAndUpdateMax(cmap, id, sameIdForOps, maxIfCount, maxRequiredBuffers, maxX, infoMap);
    if (ret != 0)
      return -1;
    ret = collectForOpsAndUpdateMax(vmap, id, sameIdForOps, maxIfCount, maxRequiredBuffers, maxX, infoMap);
    if (ret != 0)
      return -1;

    if (maxIfCount == 0) {
      LDBG("no ifblock in mainloop!");
      return -1;
    }

    // Update all loops with this id using the same max values
    for (Operation *loopOp : sameIdForOps) {
      scf::ForOp oldForOp = dyn_cast<scf::ForOp>(loopOp);
      if (!oldForOp) {
        LDBG("do not surpport other loop op except forOp!");
        return -1;
      }

      IterationTimesInfo &iterInfo = infoMap[loopOp];
      IRMapping mapper;
      scf::ForOp newForOp = extendForOpIterationCount(oldForOp, maxIfCount, maxRequiredBuffers, maxX, mapper, iterInfo.ifOpsInThisFor);
      if (!newForOp) {
        LDBG("extendForOpIterationCount failed!");
        return -1;
      }
      allForOps.push_back(loopOp);
    }
  }

  // Delete old forOp after cntArgs has been updated in extendForOpIterationCount
  for (Operation *loopOp : allForOps) {
    if (!loopOp) {
      LDBG("erasing error: loopOp is nullptr, there are nested mainloop!");
      return -1;
    }
    loopOp->erase();
  }
  return 0;
}

void UpdateLoopIterTimesPass::runOnOperation()
{
  ModuleOp module = getOperation();

  LDBG("before updateloopitertimes:\n" << module);
  LDBG("\nEnter UpdateLoopIterTimesPass!");

  int ret = 0;
  // step1: Get mainloop id to loop operation mapping, separated into cube and vector
  DenseMap<int, SmallVector<Operation *>> cmap;
  DenseMap<int, SmallVector<Operation *>> vmap;
  ret = GetMainLoopIdToLoopOpMap(module, cmap, vmap);
  if (ret != 0) {
    LDBG("\nGetMainLoopIdToLoopOpMap Failed!");
    signalPassFailure();
    return;
  }

  // step2: Calculate info for each loop operation, store into the same iterationTimesinfoMap
  DenseMap<Operation *, IterationTimesInfo> infoMap;
  ret = ComputeMainLoopTimes(cmap, infoMap);
  if (ret != 0) {
    LDBG("\nComputeMainLoopTimes from cube Failed!");
    signalPassFailure();
    return;
  }
  ret = ComputeMainLoopTimes(vmap, infoMap);
  if (ret != 0) {
    LDBG("\nComputeMainLoopTimes from vector Failed!");
    signalPassFailure();
    return;
  }

  // step3: Update loop iteration count (process loop operations with same id from both cmap and vmap)
  ret = UpdateForLoopIteration(cmap, vmap, infoMap);
  if (ret != 0) {
    LDBG("Update ForLoop Iteration Failed!\n");
    signalPassFailure();
  }
  LDBG("after UpdateForLoopIteration:\n" << module);

  // step4: Replace loop counter by if blocks' counter
  if (replaceForOpCounterInIfOps() != 0) {
    LDBG("replaceForOpCounterInIfOps Failed!\n");
    signalPassFailure();
  }

  LDBG("after updateloopitertimes:\n" << module);
  LDBG("\nExit UpdateLoopIterTimes pass.");
}

namespace mlir {
namespace triton {
std::unique_ptr<OperationPass<ModuleOp>> createUpdateLoopIterTimesPass()
{
  return std::make_unique<UpdateLoopIterTimesPass>();
}
} // namespace triton
} // namespace mlir