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
#include "ascend/include/DynamicCVPipeline/AddControlFlowCondition/Utils.h"
#include "ascend/include/DynamicCVPipeline/Common/Utils.h"
#include "bishengir/Dialect/Annotation/IR/Annotation.h"
#include "bishengir/Dialect/HIVM/IR/HIVM.h"
#include "bishengir/Dialect/Scope/IR/Scope.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/Support/Debug.h"

static constexpr const char *DEBUG_TYPE = "UpdateLoopIterTimes";
#define DBGS() (llvm::dbgs() << '[' << DEBUG_TYPE << "] ")
#define LDBG(...) LLVM_DEBUG(DBGS() << __VA_ARGS__ << "\n")

using llvm::DenseMap;
using llvm::DenseSet;
using llvm::SmallVector;
using namespace mlir;
using namespace triton;
using namespace hivm;

// Find the IfOp index that an operation belongs to
// Uses isAncestor to check if the operation is within the IfOp's region
static int findIfOpIndexInList(Operation *op, SmallVector<scf::IfOp> &ifOps,
                               llvm::DenseMap<Operation *, int> &ifOpIndex) {
  for (scf::IfOp ifOp : ifOps) {
    if (ifOp->isAncestor(op)) {
      return ifOpIndex[ifOp.getOperation()];
    }
  }
  return -1;
}

// Filter out entries where consumer op is not inside the specified forOp
static ConsumerProducerMap
filterCrossCoreMapByForOp(scf::ForOp forOp, ConsumerProducerMap &crossCoreMap) {
  ConsumerProducerMap filteredMap;
  for (auto &entry : crossCoreMap) {
    Operation *consumerOp = entry.first;
    if (!forOp->isAncestor(consumerOp)) {
      continue;
    }
    filteredMap[consumerOp] = entry.second;
  }
  return filteredMap;
}

// Find the other side's mainloop with the same mainloop id in the module
// currentIsCube/currentIsVector indicates which side is the current scope
// Returns the forOp of the other side's mainloop, or nullptr if not found
static scf::ForOp getOtherScopeMainloop(ModuleOp module, bool currentIsCube,
                                        bool currentIsVector, int mainLoopId) {
  scf::ForOp otherSideForOp = nullptr;
  int ret = 1;

  module.walk([&](scope::ScopeOp scopeOp) {
    // Determine this scope's type
    bool scopeIsCube = false;
    bool scopeIsVector = false;
    if (failed(getScopeType(scopeOp, scopeIsCube, scopeIsVector))) {
      ret = -1;
      LDBG("failed to get ScopeOp core type!");
      return mlir::WalkResult::interrupt();
    }

    // Only proceed when current is cube and scope is vector, or current is
    // vector and scope is cube
    bool isOtherSide =
        (currentIsCube && scopeIsVector) || (currentIsVector && scopeIsCube);
    if (!isOtherSide) {
      return WalkResult::advance();
    }

    // Walk for loops inside this scope to find the one with matching mainloop
    // id
    scopeOp.walk([&](Operation *op) {
      if (op->hasAttr(CVPipeline::kMainLoop)) {
        // Support both ForOp and WhileOp for mainloop
        bool isValidLoop = isa<scf::ForOp>(op) || isa<scf::WhileOp>(op);
        if (!isValidLoop) {
          LDBG("mainloop must be ForOp or WhileOp!");
          return WalkResult::advance();
        }

        // Only cast to ForOp for compatibility with current interface
        // WhileOp will be skipped in later steps
        auto targetForOp = dyn_cast<scf::ForOp>(op);
        if (!targetForOp) {
          // WhileOp case: skip for now, will be handled separately
          return WalkResult::advance();
        }

        auto targetMainLoopId =
            targetForOp->getAttrOfType<IntegerAttr>(CVPipeline::kMainLoop);
        if (targetMainLoopId && targetMainLoopId.getInt() == mainLoopId) {
          otherSideForOp = targetForOp;
          return WalkResult::interrupt();
        }
      }
      return WalkResult::advance();
    });

    if (otherSideForOp) {
      return WalkResult::interrupt();
    }
    return WalkResult::advance();
  });

  if (ret == -1) {
    return nullptr;
  } else {
    return otherSideForOp;
  }
}

// Check if the current compute block runs first by checking if the first ifOp
// has any consumer defOp Returns true if the current compute block runs first
// (no consumer in first ifOp) Returns false if the current compute block runs
// later (has consumer in first ifOp)
static bool isRunFirst(SmallVector<scf::IfOp> &ifOps,
                       ConsumerProducerMap &crossDeps) {
  if (ifOps.empty()) {
    return true;
  }
  scf::IfOp firstIfOp = ifOps[0];
  for (auto &entry : crossDeps) {
    Operation *consumerOp = entry.first;
    if (consumerOp && firstIfOp->isAncestor(consumerOp)) {
      return false;
    }
  }
  return true;
}

// Type definition for IfOp filter function
// Returns true if the IfOp should be collected, false otherwise
using IfOpFilter = std::function<bool(scf::IfOp)>;

// Default filter that accepts all IfOps
static bool defaultIfOpFilter(scf::IfOp ifOp) { return true; }

// Filter that checks if IfOp contains sync_block_wait or sync_block_set op
static bool syncBlockFilter(scf::IfOp ifOp) {
  bool containsSyncBlock = false;
  ifOp.walk([&](Operation *innerOp) {
    if (isa<hivm::SyncBlockWaitOp>(innerOp) ||
        isa<hivm::SyncBlockSetOp>(innerOp)) {
      containsSyncBlock = true;
      return WalkResult::interrupt();
    }
    return WalkResult::advance();
  });
  return containsSyncBlock;
}

// Collect ifOps from forOp based on the provided filter
// Fills ifOps and ifOpIndexMap with the collected ifOps
// Returns true on success, false on error or if no ifOps found
bool collectIfOps(scf::ForOp forOp, SmallVector<scf::IfOp> &ifOps,
                  llvm::DenseMap<Operation *, int> &ifOpIndexMap,
                  IfOpFilter filter = defaultIfOpFilter) {
  ifOps.clear();
  ifOpIndexMap.clear();
  int index = 1;
  int ret = 0;

  forOp.walk([&](Operation *op) {
    if (op->hasAttr(CVPipeline::kIf)) {
      auto ifOp = dyn_cast<scf::IfOp>(op);
      if (!ifOp) {
        ret = -1;
        LDBG("ssbuffer.if attribute is not allocated on ifOp!");
        return WalkResult::interrupt();
      }
      // Apply the filter to determine if this IfOp should be collected
      if (filter(ifOp)) {
        ifOps.push_back(ifOp);
        ifOpIndexMap[ifOp.getOperation()] = index++;
      }
    }
    return WalkResult::advance();
  });

  if (ret == -1) {
    LDBG("Failed to collect ifOps!");
    return false;
  }

  if (ifOps.empty()) {
    LDBG("mainloop does not contain any ifblocks!");
    return false;
  }

  return true;
}

// Helper function to find the other side's mainloop and collect its ifOps
// Returns the other side's forOp and fills otherSideIfOps with ifOps from that
// mainloop Returns nullptr if not found
static scf::ForOp findOtherSideMainloopAndIfOps(
    scf::ForOp currentForOp, bool currentIsCube, bool currentIsVector,
    SmallVector<scf::IfOp> &otherSideIfOps,
    llvm::DenseMap<Operation *, int> &otherSideIfOpIndexMap) {
  otherSideIfOps.clear();
  otherSideIfOpIndexMap.clear();

  // Get current forOp's mainloop id
  auto currentMainLoopId =
      currentForOp->getAttrOfType<IntegerAttr>(CVPipeline::kMainLoop);
  if (!currentMainLoopId) {
    LDBG("Current forOp does not have main_loop id!");
    return nullptr;
  }
  int mainLoopId = currentMainLoopId.getInt();

  // Find the other side's mainloop with the same mainloop id
  // Traverse up the parent chain to find ModuleOp
  Operation *currentOp = currentForOp->getParentOp();
  ModuleOp module = nullptr;
  while (currentOp) {
    if (auto mod = dyn_cast<ModuleOp>(currentOp)) {
      module = mod;
      break;
    }
    currentOp = currentOp->getParentOp();
  }
  if (!module) {
    LDBG("Can not find ModuleOp in parent chain!");
    return nullptr;
  }

  // Find other side mainloop with the same mainloop id
  scf::ForOp otherSideForOp =
      getOtherScopeMainloop(module, currentIsCube, currentIsVector, mainLoopId);

  if (!otherSideForOp) {
    LDBG("Can not find the other side's mainloop with id: " << mainLoopId);
    return nullptr;
  }

  // Collect ifOps from the other side's mainloop
  // Only collect ifOps that contain hivm.hir.sync_block_wait op
  if (!collectIfOps(otherSideForOp, otherSideIfOps, otherSideIfOpIndexMap,
                    syncBlockFilter)) {
    LDBG("Failed to collect ifOps from other side mainloop!");
    return nullptr;
  }

  return otherSideForOp;
}

// calculateFactor computes the buffer factor needed for loop iteration
// extension Core idea: if producer ifOp (n) and consumer ifOp (m) have
// dependency, we need (m - n + 1) buffers to support the loop extension This
// function handles both intra-core and cross-core dependencies Returns
// {maxRequiredBuffers, x} for computing: ceil(original_iter_count *
// requiredBuffers / x) + ifCount
std::pair<int, int> UpdateLoopIterTimesPass::calculateFactor(scf::ForOp forOp) {
  int maxRequiredBuffers = 1;
  int maxX = 1;

  // Step1: Collect all IfOps with ssbuffer.if attribute in this for loop
  SmallVector<scf::IfOp> ifOps;
  DenseMap<Operation *, int> ifOpIndex;
  if (!collectIfOps(forOp, ifOps, ifOpIndex)) {
    LDBG("Failed to collect ifOps!");
    return {-1, -1};
  }

  // Step2: Calculate factor based on intra-core dependencies
  bool hasIntraDeps = info->intraCoreDependentMap.count(forOp) &&
                      !info->intraCoreDependentMap[forOp].empty();
  bool hasCrossDeps = !info->crossCoreDependentMap.empty();

  // If no dependencies at all, return factor = 1
  if (!hasIntraDeps && !hasCrossDeps) {
    return {1, 1};
  }

  // Calculate intra-core factor
  if (hasIntraDeps) {
    auto &intraDeps = info->intraCoreDependentMap[forOp];
    auto [intraRequiredBuffers, intraX] =
        calculateIntraDepsFactor(ifOps, ifOpIndex, intraDeps);
    if (intraRequiredBuffers == -1 || intraX == -1) {
      LDBG("calculateIntraDepsFactor failed!");
      return {-1, -1};
    }
    maxRequiredBuffers = intraRequiredBuffers;
    maxX = intraX;
  }

  // Calculate cross-core factor and merge with intra-core factor
  if (hasCrossDeps) {
    // Filter out entries where consumer op is not inside current forOp
    ConsumerProducerMap filteredCrossCoreMap =
        filterCrossCoreMapByForOp(forOp, info->crossCoreDependentMap);

    // for caculating the crossdeps, need to filter ifblocks without
    // sync_wait/sync_set op
    SmallVector<scf::IfOp> filterIfOps;
    DenseMap<Operation *, int> filterIfOpIndex;
    if (!collectIfOps(forOp, filterIfOps, filterIfOpIndex, syncBlockFilter)) {
      LDBG("Failed to collect filtered ifOps!");
      return {-1, -1};
    }

    auto [crossRequiredBuffers, crossX] = calculateCrossDepsFactor(
        forOp, filterIfOps, filterIfOpIndex, filteredCrossCoreMap);
    if (crossRequiredBuffers == -1 || crossX == -1) {
      LDBG("calculateCrossDepsFactor failed!");
      return {-1, -1};
    }

    // Update maxRequiredBuffers/maxX
    if (crossRequiredBuffers * maxX > maxRequiredBuffers * crossX) {
      maxRequiredBuffers = crossRequiredBuffers;
      maxX = crossX;
    }
  }

  // Step3: Calculate factor based on iteration dependencies (tensor iter_args
  // dependencies)
  bool hasIterDeps = info->tensorIterArgDepsMap.count(forOp) &&
                     !info->tensorIterArgDepsMap[forOp].empty();

  if (hasIterDeps) {
    LDBG("find cross-iteration dependency, size: "
         << info->tensorIterArgDepsMap[forOp].size());
    auto [iterRequiredBuffers, iterX] =
        calculateIterDepsFactor(forOp, ifOps, ifOpIndex);
    if (iterRequiredBuffers == -1 || iterX == -1) {
      LDBG("calculateIterDepsFactor failed!");
      return {-1, -1};
    }
    // Compare iterRequiredBuffers/iterX vs maxRequiredBuffers/maxX
    // Take the larger fraction
    if (iterRequiredBuffers * maxX > maxRequiredBuffers * iterX) {
      maxRequiredBuffers = iterRequiredBuffers;
      maxX = iterX;
    }
  }

  return {maxRequiredBuffers, maxX};
}

static int getConsumerIfOpIndex(Operation *consumerOp,
                                SmallVector<scf::IfOp> &ifOps,
                                llvm::DenseMap<Operation *, int> &ifOpIndex) {
  if (!consumerOp) {
    LDBG("consumerOp is null!");
    return -1;
  }
  int idx = findIfOpIndexInList(consumerOp, ifOps, ifOpIndex);
  if (idx == -1) {
    LDBG("Can not find the consumerOp in any ifOps!");
    return -1;
  }
  return idx;
}

static int getProducerIfOpIndex(SmallVector<Operation *> &producerOps,
                                SmallVector<scf::IfOp> &ifOps,
                                llvm::DenseMap<Operation *, int> &ifOpIndex) {
  if (producerOps.empty()) {
    LDBG("consumer do not have the producerOps!");
    return -1;
  }

  int producerIfOpIndex = -1;
  for (Operation *producerOp : producerOps) {
    int currentIndex = findIfOpIndexInList(producerOp, ifOps, ifOpIndex);

    if (currentIndex == -1) {
      LDBG("ProducerOp not found in any ifOps: " << *producerOp);
      return -1;
    }

    // First producer found
    if (producerIfOpIndex == -1) {
      producerIfOpIndex = currentIndex;
    } else {
      // Check if all producers are in the same ifOp
      if (currentIndex != producerIfOpIndex) {
        LDBG("ProducerOps are not in the same ifOp! "
             << "First producer ifOp index: " << producerIfOpIndex
             << ", Current producer ifOp index: " << currentIndex);
        return -1;
      }
    }
  }

  if (producerIfOpIndex == -1) {
    LDBG("All producerOps are not found in any ifOps!");
    return -1;
  }

  return producerIfOpIndex;
}

std::pair<int, int> UpdateLoopIterTimesPass::calculateIntraDepsFactor(
    SmallVector<scf::IfOp> &ifOps, DenseMap<Operation *, int> &ifOpIndex,
    llvm::DenseMap<Operation *, SmallVector<Operation *>> &deps) {
  int maxRequiredBuffers = 1;
  int maxX = 1;

  // Iterate all dependencies and calculate required buffer count
  for (auto &entry : deps) {
    Operation *consumerOp = entry.first;                 // Consumer operation
    SmallVector<Operation *> producerOps = entry.second; // Producer op list
    int x = producerOps.size();                          // Producer op count

    // Find the IfOp index that consumer belongs to (ConsumerIdx)
    int ConsumerIdx = getConsumerIfOpIndex(consumerOp, ifOps, ifOpIndex);
    if (ConsumerIdx == -1) {
      return {-1, -1};
    }

    // Find the IfOp index that producer belongs to (producerIdx)
    int producerIdx = getProducerIfOpIndex(producerOps, ifOps, ifOpIndex);
    if (producerIdx == -1) {
      return {-1, -1};
    }

    // If consumer is after producer (ConsumerIdx > producerIdx), calculate
    // required buffer count ConsumerIdx - producerIdx + 1 represents the buffer
    // count needed to cover this distance
    if (ConsumerIdx <= producerIdx) {
      LDBG("producer is after the consumer!");
      return {-1, -1};
    }
    int requiredBuffers = ConsumerIdx - producerIdx + 1;

    // Update maxRequiredBuffers/maxX
    if (requiredBuffers * maxX > maxRequiredBuffers * x) {
      maxRequiredBuffers = requiredBuffers;
      maxX = x;
    }
  }

  return {maxRequiredBuffers, maxX};
}

// calculateCrossDepsFactor computes the buffer factor based on cross-core
// dependencies For cross-core deps consumer is in current forOp (one side: cube
// or vector) producer is in another mainloop with the same id but in the other
// scope
std::pair<int, int> UpdateLoopIterTimesPass::calculateCrossDepsFactor(
    scf::ForOp forOp, SmallVector<scf::IfOp> &ifOps,
    DenseMap<Operation *, int> &ifOpIndex, ConsumerProducerMap &crossDeps) {
  int maxRequiredBuffers = 1;
  int maxX = 1;

  // Determine current forOp's scope type (cube or vector)
  bool currentIsCube = false;
  bool currentIsVector = false;
  Operation *currentScope = forOp->getParentOp();
  while (currentScope) {
    if (isa<scope::ScopeOp>(currentScope)) {
      break;
    }
    currentScope = currentScope->getParentOp();
  }
  if (failed(getScopeType(currentScope, currentIsCube, currentIsVector))) {
    LDBG("Current forOp is not in a valid cube or vector scope!");
    return {-1, -1};
  }

  // Find the other side scope's mainloop and collect its ifOps
  SmallVector<scf::IfOp> otherSideIfOps;
  DenseMap<Operation *, int> otherSideIfOpIndexMap;
  scf::ForOp otherSideForOp =
      findOtherSideMainloopAndIfOps(forOp, currentIsCube, currentIsVector,
                                    otherSideIfOps, otherSideIfOpIndexMap);

  if (!otherSideForOp) {
    LDBG("Failed to find other side mainloop or its ifOps!");
    return {-1, -1};
  }

  // Check if the first ifOp has any consumer op in crossDeps
  // If not, it means the current compute block executes first, need to subtract
  // 1 from requiredBuffers
  bool runFirst = isRunFirst(ifOps, crossDeps);

  // Iterate all cross-core dependencies
  for (auto &entry : crossDeps) {
    Operation *consumerOp = entry.first; // Consumer operation
    for (SmallVector<Operation *> &producerOps : entry.second) {
      int x = producerOps.size(); // Producer op count (one buffer has
                                  // two ops in different scope)
      // some special buffer is not Symmetrical
      if (producerOps.size() == 1) {
        x = 1;
      }

      // Find the IfOp index that consumer belongs to (comsumerIdx)
      int comsumerIdx = getConsumerIfOpIndex(consumerOp, ifOps, ifOpIndex);
      if (comsumerIdx == -1) {
        return {-1, -1};
      }

      // Find producer's position in the other side's ifOps (producerIdx)
      int producerIdx = getProducerIfOpIndex(producerOps, otherSideIfOps,
                                             otherSideIfOpIndexMap);
      if (producerIdx == -1) {
        return {-1, -1};
      }

      // If consumer is after producer (comsumerIdx >= producerIdx), calculate
      // required buffer count comsumerIdx - producerIdx + 1 represents the
      // buffer count needed to cover this distance If the current compute
      // block executes first (firstIfOp has no consumer), subtract 1
      if (comsumerIdx < producerIdx) {
        // case : C1 -> V1V2V3 -> C2
        // in this case comsumerIdx < producerIdx, crossDeps hard to process, do
        // not change the loop iteration times
        LDBG("there is complex case!");
        return {1, 1};
      }
      int requiredBuffers = comsumerIdx - producerIdx + 1;
      if (runFirst) {
        requiredBuffers = requiredBuffers - 1;
      }
      LDBG("consumer : " << *consumerOp);
      LDBG("consumer comsumerIdx: " << comsumerIdx
                                    << ", producerIdx: " << producerIdx);
      LDBG("requiredBuffers: " << requiredBuffers);
      LDBG("buffer: " << x);
      LDBG("runFirst: " << runFirst);

      // Update max value using fraction comparison to avoid precision issues
      // Comparing requiredBuffers/maxX vs maxRequiredBuffers/x is equivalent to
      // comparing requiredBuffers * maxX vs maxRequiredBuffers * x
      if (requiredBuffers * maxX > maxRequiredBuffers * x) {
        maxRequiredBuffers = requiredBuffers;
        maxX = x;
      }
    }
  }

  return {maxRequiredBuffers, maxX};
}

// calculateIterDepsFactor computes the buffer factor based on iteration
// dependencies For iter deps: consumer and producer are IfOps within the same
// forOp Core idea: iteration dependencies are special - consume first, then
// produce So if consumer ifOp (comsumerIdx) and producer ifOp (producerIdx)
// have dependency, we need (producerIdx - comsumerIdx) buffers to support the
// loop extension This function iterates all dependencies from
// tensorIterArgDepsMap and finds the maximum required buffer count
std::pair<int, int> UpdateLoopIterTimesPass::calculateIterDepsFactor(
    scf::ForOp forOp, SmallVector<scf::IfOp> &ifOps,
    DenseMap<Operation *, int> &ifOpIndex) {
  int maxRequiredBuffers = 1;
  int maxX = 1;

  // Check if this forOp has iteration dependencies
  if (!info->tensorIterArgDepsMap.count(forOp)) {
    return {maxRequiredBuffers, maxX};
  }

  auto &iterArgDepsVec = info->tensorIterArgDepsMap[forOp];

  // Iterate all tensor iter_args dependencies in this forOp
  // tensorIterArgDepsMap now stores a SmallVector of TensorIterArgIfOpRelation
  for (TensorIterArgIfOpRelation &relation : iterArgDepsVec) {
    scf::IfOp producerIfOp = relation.producer;
    llvm::SmallVector<scf::IfOp> &consumerIfOps = relation.consumers;

    // Skip if no producer or consumers
    if (!producerIfOp || consumerIfOps.empty()) {
      continue;
    }

    // x is the number of producer buffers (1 for iteration dependencies)
    int x = 1;

    // Get producer IfOp index (producerIdx)
    auto producerIt = ifOpIndex.find(producerIfOp.getOperation());
    if (producerIt == ifOpIndex.end()) {
      LDBG("Producer IfOp not found in ifOps list!");
      return {-1, -1};
    }
    int producerIdx = producerIt->second;

    // For each consumer IfOp, find its index (comsumerIdx)
    // Calculate requiredBuffers = producerIdx - comsumerIdx (consume first,
    // then produce)
    for (scf::IfOp consumerIfOp : consumerIfOps) {
      // Find consumer IfOp index in ifOps list
      auto consumerIt = ifOpIndex.find(consumerIfOp.getOperation());
      if (consumerIt == ifOpIndex.end()) {
        LDBG("Consumer IfOp not found in ifOps list!");
        return {-1, -1};
      }
      int comsumerIdx = consumerIt->second;

      if (producerIdx <= comsumerIdx) {
        LDBG("Producer IfOp index (producerIdx) is not greater than consumer "
             "IfOp index "
             "(comsumerIdx)!");
        LDBG("arg value: " << relation.iterArg);
        LDBG("Producer IfOp index producerIdx: " << producerIdx);
        LDBG("consumer IfOp index comsumerIdx: " << comsumerIdx);
        return {-1, -1};
      }
      int requiredBuffers = producerIdx - comsumerIdx + 1;
      LDBG("consumerIfOp : " << consumerIfOp);
      LDBG("consumer comsumerIdx: " << comsumerIdx
                                    << ", producerIdx: " << producerIdx);
      LDBG("requiredBuffers: " << requiredBuffers);

      if (requiredBuffers * maxX > maxRequiredBuffers * x) {
        maxRequiredBuffers = requiredBuffers;
        maxX = x;
      }
    }
  }

  return {maxRequiredBuffers, maxX};
}

// Part 1: Compute new loop upper bound and create arithmetic ops
Value UpdateLoopIterTimesPass::computeNewLoopUpperBound(
    OpBuilder &builder, Location loc, scf::ForOp forOp, int ifCount,
    int requiredBuffers, int x) {
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
  Value rangeDiff = builder.create<arith::SubIOp>(loc, originalUpperBound,
                                                  originalLowerBound);

  // Calculate: iterCount = ceil(rangeDiff / step)
  Value iterCount = createCeilDiv(rangeDiff, originalStep);

  // Calculate: scaledIterCount = iterCount * requiredBuffers
  Value scaledIterCount =
      builder.create<arith::MulIOp>(loc, iterCount, requiredBuffersValue);

  // Calculate: ceiledScaledIterCount = ceil(scaledIterCount / x)
  Value ceiledScaledIterCount = createCeilDiv(scaledIterCount, xValue);

  // Calculate: newIterCount = ceiledScaledIterCount + ifCount
  Value newIterCount =
      builder.create<arith::AddIOp>(loc, ceiledScaledIterCount, ifCountValue);

  // Calculate: totalSteps = step * newIterCount
  Value totalSteps =
      builder.create<arith::MulIOp>(loc, originalStep, newIterCount);

  // Calculate: newUpperBound = lowerBound + totalSteps
  Value newUpperBound =
      builder.create<arith::AddIOp>(loc, originalLowerBound, totalSteps);

  return newUpperBound;
}

// Part 2: Clone for loop with new upper bound
scf::ForOp UpdateLoopIterTimesPass::cloneForOpWithNewUpperBound(
    OpBuilder &builder, Location loc, scf::ForOp oldForOp, Value newUpperBound,
    IRMapping &mapper) {
  Value originalLowerBound = oldForOp.getLowerBound();
  Value originalStep = oldForOp.getStep();

  SmallVector<Value> newInitArgs(oldForOp.getInitArgs().begin(),
                                 oldForOp.getInitArgs().end());

  auto newForOp = builder.create<scf::ForOp>(
      loc, originalLowerBound, newUpperBound, originalStep, newInitArgs);

  // Copy attributes from old forOp
  for (auto &attr : oldForOp->getAttrs()) {
    newForOp->setAttr(attr.getName(), attr.getValue());
  }

  // Map induction variable and iter args
  mapper.map(oldForOp.getInductionVar(), newForOp.getInductionVar());

  for (auto [oldArg, newArg] :
       llvm::zip(oldForOp.getRegionIterArgs(), newForOp.getRegionIterArgs())) {
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
  for (Operation &op :
       llvm::make_early_inc_range(oldBlock->without_terminator())) {
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
    SmallVector<scf::IfOp> &ifOpsInThisFor) {
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
    IRMapping &mapper, SmallVector<scf::IfOp> &ifOpsInThisFor) {
  OpBuilder builder(oldForOp);
  Location loc = oldForOp.getLoc();

  // Part 1: Compute new loop upper bound
  Value newUpperBound = computeNewLoopUpperBound(builder, loc, oldForOp,
                                                 ifCount, requiredBuffers, x);
  if (!newUpperBound) {
    LDBG("computeNewLoopUpperBound failed!");
    return nullptr;
  }

  // Part 2: Clone for loop with new upper bound
  scf::ForOp newForOp = cloneForOpWithNewUpperBound(builder, loc, oldForOp,
                                                    newUpperBound, mapper);
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

// Update WhileOp condition based on ifblock conditions
// Uses info->whileBlockArgMap: {WhileOp: {block_id: {new_arg_idx:
// old_arg_idx}}} For each IfOp in WhileOp, get its block_id Get new args and
// mapping from info->whileBlockArgMap Copy entire beforeRegion and replace old
// args with new args Combine all ifblock conditions with OR operation
int UpdateLoopIterTimesPass::UpdateWhileLoopCondition(
    DenseMap<int, SmallVector<Operation *>> &mainLoopIdMap) {
  int ret = 0;

  for (auto &entry : mainLoopIdMap) {
    for (Operation *loopOp : entry.second) {
      if (!isa<scf::WhileOp>(loopOp)) {
        continue;
      }

      auto whileOp = dyn_cast<scf::WhileOp>(loopOp);
      if (!info->whileBlockArgMap.count(whileOp)) {
        LDBG("WhileOp not found in whileBlockArgMap!");
        return -1;
      }
      auto &blockArgMap = info->whileBlockArgMap[whileOp];
      if (blockArgMap.empty()) {
        LDBG("blockArgMap is empty for WhileOp!");
        return -1;
      }

      // Get the 'before' region which contains the condition
      Region &beforeRegion = whileOp.getBefore();
      Block &beforeBlock = beforeRegion.front();
      Operation *terminator = beforeBlock.getTerminator();
      if (!terminator || !isa<scf::ConditionOp>(terminator)) {
        LDBG("Before block has no valid ConditionOp terminator!");
        return -1;
      }
      auto conditionOp = dyn_cast<scf::ConditionOp>(terminator);
      Value originalCondition = conditionOp.getCondition();
      OpBuilder builder(conditionOp);
      Location loc = whileOp.getLoc();

      // Collect all operations to clone
      SmallVector<Operation *> opsToClone;
      for (Operation &op : beforeBlock.without_terminator()) {
        opsToClone.push_back(&op);
      }

      // Traverse all IfOps with ssbuffer.if and
      // build condition for each IfOp directly
      Value combinedCondition;
      whileOp.walk([&](scf::IfOp ifOp) {
        if (!ifOp->hasAttr(CVPipeline::kIf)) {
          return WalkResult::advance();
        }

        auto blockIdAttr = ifOp->getAttrOfType<IntegerAttr>(CVPipeline::kIf);
        if (!blockIdAttr) {
          ret = -1;
          return WalkResult::interrupt();
        }
        int blockId = blockIdAttr.getInt();
        if (!blockArgMap.count(blockId)) {
          ret = -1;
          LDBG("blockId " << blockId << " not found in blockArgMap!");
          return WalkResult::interrupt();
        }

        auto &argMap = blockArgMap[blockId]; // {new_arg_idx: old_arg_idx}

        // Step 1: Build arg replacement mapping
        IRMapping mapper;
        for (auto &argEntry : argMap) {
          int newArgIdx = argEntry.first;
          int oldArgIdx = argEntry.second;

          BlockArgument oldArg = beforeBlock.getArgument(oldArgIdx);
          BlockArgument newArg = beforeBlock.getArgument(newArgIdx);

          mapper.map(oldArg, newArg);
        }

        // Step 2: Clone entire beforeRegion
        // Set insertion point before conditionOp
        builder.setInsertionPoint(conditionOp);

        // Clone all operations
        Value newCondition;
        for (Operation *op : opsToClone) {
          Operation *clonedOp = builder.clone(*op, mapper);

          // If this op defines the original condition, get the corresponding
          // result
          if (op->getResultTypes().size() > 0) {
            for (unsigned i = 0; i < op->getNumResults(); ++i) {
              if (op->getResult(i) == originalCondition) {
                newCondition = clonedOp->getResult(i);
                break;
              }
            }
          }
        }

        if (!newCondition) {
          ret = -1;
          LDBG("Failed to create new condition for blockId: " << blockId);
          return WalkResult::interrupt();
        }

        // Step 3: Combine with existing conditions using OR
        if (!combinedCondition) {
          combinedCondition = newCondition;
        } else {
          combinedCondition = builder.create<arith::OrIOp>(
              loc, combinedCondition, newCondition);
        }

        return WalkResult::advance();
      });

      // Step 4: Update the condition in the before block
      if (combinedCondition) {
        conditionOp.getConditionMutable().assign(combinedCondition);
        LDBG("Updated WhileOp condition for: " << whileOp);
      } else {
        LDBG("Failed to generate valid condition!");
        return -1;
      }
    }
  }

  return ret;
}

// Traverse each mainloop, find ifOp with ssbuffer.if attribute inside,
// and replace the mainloop's induction variable with the counter in cntArgs
int UpdateLoopIterTimesPass::replaceForOpCounterInIfOps() {
  int ret = 0;
  // Traverse all mainloops in the module
  getOperation().walk([&](Operation *op) {
    if (op->hasAttr(CVPipeline::kMainLoop)) {
      // Skip WhileOp, only process ForOp in this step
      if (isa<scf::WhileOp>(op)) {
        LDBG("Skip WhileOp in replaceForOpCounterInIfOps.");
        return WalkResult::advance();
      }

      auto forOp = dyn_cast<scf::ForOp>(op);
      if (!forOp) {
        ret = -1;
        LDBG("Do not support other mainloop except forOp!");
        return WalkResult::interrupt();
      }

      Value indVar = forOp.getInductionVar();

      // Find all ifOps with ssbuffer.if attribute inside this mainloop
      forOp.walk([&](scf::IfOp ifOp) {
        if (ifOp->hasAttr(CVPipeline::kIf)) {
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

// step1: Get mainloop id to loop operation mapping, separated into cube and
// vector
int UpdateLoopIterTimesPass::GetMainLoopIdToLoopOpMap(
    ModuleOp module, DenseMap<int, SmallVector<Operation *>> &cmap,
    DenseMap<int, SmallVector<Operation *>> &vmap) {
  cmap.clear();
  vmap.clear();
  int ret = 0;

  module.walk([&](scope::ScopeOp scopeOp) {
    // Determine if it's CUBE or VECTOR
    bool isCube = false;
    bool isVector = false;
    if (failed(getScopeType(scopeOp, isCube, isVector))) {
      ret = -1;
      LDBG("mlir do not processed by split mix kernel!");
      return mlir::WalkResult::interrupt();
    }

    // Walk for loops inside the scope
    scopeOp.walk([&](Operation *op) {
      if (op->hasAttr(CVPipeline::kMainLoop)) {
        // Support both ForOp and WhileOp for mainloop
        bool isValidLoop = isa<scf::ForOp>(op) || isa<scf::WhileOp>(op);
        if (!isValidLoop) {
          ret = -1;
          LDBG("mainloop must be ForOp or WhileOp!");
          return mlir::WalkResult::interrupt();
        }

        auto mainLoopId = op->getAttrOfType<IntegerAttr>(CVPipeline::kMainLoop);
        if (mainLoopId) {
          int id = mainLoopId.getInt();
          if (isCube) {
            cmap[id].push_back(op);
          } else if (isVector) {
            vmap[id].push_back(op);
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
    DenseMap<Operation *, IterationTimesInfo> &infoMap) {
  for (auto &entry : loopMap) {
    for (Operation *loopOp : entry.second) {
      if (isa<scf::WhileOp>(loopOp)) {
        continue;
      }

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
        if (ifOp->hasAttr(CVPipeline::kIf)) {
          auto parentOp = ifOp->getParentOp();
          if (parentOp->hasAttr(CVPipeline::kMainLoop) &&
              isa<scf::ForOp>(parentOp)) {
            if (parentOp == forOp.getOperation()) {
              iterInfo.ifCount++;
              iterInfo.ifOpsInThisFor.push_back(ifOp);
            }
          } else {
            LDBG("Get wrong ifblock structure: ifblock's parentOp is not "
                 "mainloop!");
            LDBG("ifblock Op: " << ifOp);
            LDBG("Parent Op: " << *parentOp);
            return -1;
          }
        } else {
          LDBG("ifOp in cntArgs does not contain ssbuffer.if Attribute\n ifOp: "
               << ifOp);
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
    DenseMap<int, SmallVector<Operation *>> &map, int id,
    SmallVector<Operation *> &allForOps, int &maxIfCount,
    int &maxRequiredBuffers, int &maxX,
    DenseMap<Operation *, IterationTimesInfo> &infoMap) {
  if (map.count(id)) {
    for (Operation *loopOp : map[id]) {
      if (!isa<scf::ForOp>(loopOp)) {
        continue;
      }

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

void UpdateLoopIterTimesPass::updateMainLoopMaps(
    Operation *oldForOp, Operation *newForOp,
    DenseMap<int, SmallVector<Operation *>> &cmap,
    DenseMap<int, SmallVector<Operation *>> &vmap,
    DenseMap<Operation *, IterationTimesInfo> &infoMap) {
  // Update cmap
  for (auto &entry : cmap) {
    for (Operation *&op : entry.second) {
      if (op == oldForOp) {
        op = newForOp;
      }
    }
  }
  // Update vmap
  for (auto &entry : vmap) {
    for (Operation *&op : entry.second) {
      if (op == oldForOp) {
        op = newForOp;
      }
    }
  }
  // Update infoMap
  auto it = infoMap.find(oldForOp);
  if (it != infoMap.end()) {
    IterationTimesInfo iterInfo = std::move(it->second);
    infoMap.erase(it);
    infoMap[newForOp] = std::move(iterInfo);
  }
}

int UpdateLoopIterTimesPass::UpdateForLoopIteration(
    DenseMap<int, SmallVector<Operation *>> &cmap,
    DenseMap<int, SmallVector<Operation *>> &vmap,
    DenseMap<Operation *, IterationTimesInfo> &infoMap) {
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
    ret = collectForOpsAndUpdateMax(cmap, id, sameIdForOps, maxIfCount,
                                    maxRequiredBuffers, maxX, infoMap);
    if (ret != 0)
      return -1;
    ret = collectForOpsAndUpdateMax(vmap, id, sameIdForOps, maxIfCount,
                                    maxRequiredBuffers, maxX, infoMap);
    if (ret != 0)
      return -1;

    // Update all loops with this id using the same max values
    for (Operation *loopOp : sameIdForOps) {
      if (maxIfCount == 0) {
        LDBG("no ifblock in mainloop!");
        return -1;
      }

      scf::ForOp oldForOp = dyn_cast<scf::ForOp>(loopOp);
      if (!oldForOp) {
        LDBG("do not surpport other loop op except forOp!");
        return -1;
      }

      IterationTimesInfo &iterInfo = infoMap[loopOp];
      IRMapping mapper;
      scf::ForOp newForOp =
          extendForOpIterationCount(oldForOp, maxIfCount, maxRequiredBuffers,
                                    maxX, mapper, iterInfo.ifOpsInThisFor);
      if (!newForOp) {
        LDBG("extendForOpIterationCount failed!");
        return -1;
      }
      updateMainLoopMaps(loopOp, newForOp.getOperation(), cmap, vmap, infoMap);
      allForOps.push_back(loopOp);
    }
  }

  // Delete old forOp after cntArgs and maps have been updated
  for (Operation *loopOp : allForOps) {
    if (!loopOp) {
      LDBG("erasing error: loopOp is nullptr, there are nested mainloop!");
      return -1;
    }
    loopOp->erase();
  }
  return 0;
}

void UpdateLoopIterTimesPass::runOnOperation() {
  ModuleOp module = getOperation();

  if (CVPipeline::hasFallbackAttr(module)) {
    return;
  }

  LDBG("before updateloopitertimes:\n" << module);
  LDBG("\nEnter UpdateLoopIterTimesPass!");

  int ret = 0;
  // step1: Get mainloop id to loop operation mapping, separated into cube and
  // vector
  DenseMap<int, SmallVector<Operation *>> cmap;
  DenseMap<int, SmallVector<Operation *>> vmap;
  ret = GetMainLoopIdToLoopOpMap(module, cmap, vmap);
  if (ret != 0) {
    LDBG("GetMainLoopIdToLoopOpMap Failed!");
    CVPipeline::setFallbackAttr(module, CVPipeline::ERRCODE_FAILED);
  }

  // step2: Calculate info for each loop operation, store into the same
  // iterationTimesinfoMap
  DenseMap<Operation *, IterationTimesInfo> infoMap;
  ret = ComputeMainLoopTimes(cmap, infoMap);
  if (ret != 0) {
    LDBG("ComputeMainLoopTimes from cube Failed!");
    CVPipeline::setFallbackAttr(module, CVPipeline::ERRCODE_FAILED);
  }
  ret = ComputeMainLoopTimes(vmap, infoMap);
  if (ret != 0) {
    LDBG("ComputeMainLoopTimes from vector Failed!");
    CVPipeline::setFallbackAttr(module, CVPipeline::ERRCODE_FAILED);
  }

  // step3: Update loop iteration count (process loop operations with same id
  // from both cmap and vmap)
  ret = UpdateForLoopIteration(cmap, vmap, infoMap);
  if (ret != 0) {
    LDBG("Update ForLoop Iteration Failed!");
    CVPipeline::setFallbackAttr(module, CVPipeline::ERRCODE_FAILED);
  }
  LDBG("after UpdateForLoopIteration:\n" << module);

  // step4: Replace loop counter by if blocks' counter
  ret = replaceForOpCounterInIfOps();
  if (ret != 0) {
    LDBG("replaceForOpCounterInIfOps Failed!");
    CVPipeline::setFallbackAttr(module, CVPipeline::ERRCODE_FAILED);
  }

  // step5: Update WhileOp condition based on ifblock conditions
  ret = UpdateWhileLoopCondition(cmap);
  if (ret != 0) {
    LDBG("UpdateWhileLoopCondition from cube Failed!");
    CVPipeline::setFallbackAttr(module, CVPipeline::ERRCODE_FAILED);
  }
  ret = UpdateWhileLoopCondition(vmap);
  if (ret != 0) {
    LDBG("UpdateWhileLoopCondition from vector Failed!");
    CVPipeline::setFallbackAttr(module, CVPipeline::ERRCODE_FAILED);
  }

  LDBG("after updateloopitertimes:\n" << module);
  LDBG("\nExit UpdateLoopIterTimes pass.");
}

namespace mlir {
namespace triton {
std::unique_ptr<OperationPass<ModuleOp>> createUpdateLoopIterTimesPass() {
  return std::make_unique<UpdateLoopIterTimesPass>();
}
} // namespace triton
} // namespace mlir
