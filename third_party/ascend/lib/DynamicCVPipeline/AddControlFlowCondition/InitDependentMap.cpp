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

#include "third_party/ascend/include/DynamicCVPipeline/AddControlFlowCondition/InitDependentMap.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Debug.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/BuiltinAttributes.h"

// Role in dependency attribute: ssbuffer.crossDeps/intraDeps = [groupId, roleId]
// role: 1=producer, 0=consumer
static const int producerId = 1;
static const int consumerId = 0;
static constexpr const char *DEBUG_TYPE = "InitDependentMap";
#define DBGS() (llvm::dbgs() << '[' << DEBUG_TYPE << "] ")
#define LDBG(...) LLVM_DEBUG(DBGS() << __VA_ARGS__ << "\n")

using namespace mlir;
using namespace triton;

// Function: Check if a consumer op is inside a given mainLoop
//            but not inside any nested mainloop, and push to consumers if true
// Input: Consumer operation, target mainLoop forOp, reference to consumers vector
// Output: consumers - push consumer to this vector if it's inside mainLoop (not in nested mainloop)
// Return: 0 for success (consumer pushed), -1 for failure (not in mainLoop or in nested mainLoop)
static int isConsumerInMainLoop(Operation *consumer, scf::ForOp mainLoop,
                                SmallVector<Operation *> &consumers)
{
  Operation *current = consumer->getParentOp();

  // Traverse up the parent chain until we reach the top (nullptr)
  while (current != nullptr) {
    if (auto forOp = dyn_cast<scf::ForOp>(current)) {
      if (forOp->hasAttr("ssbuffer.main_loop") && forOp != mainLoop) {
        // comsumer Op not in the current mainloop
        return 0;
      }
    }
    // If we reach the target mainLoop, consumer is inside it
    if (current == mainLoop) {
      consumers.push_back(consumer);
      return 0;
    }
    current = current->getParentOp();
  }

  LDBG("Can not find the consumer's mainloop!");
  return -1;
}

// Function: Collect ops with dependency attributes, grouped by group ID
// Input: Root operation to traverse (module or forOp), attribute name
// Output: depsByGroup - Ops grouped by group ID, format: group -> [(op, role), ...]
//         Attribute format: [group, role], role: 1=producer, 0=consumer
// Return: 0 for success, -1 for failure
static int collectDepsByGroup(Operation *rootOp, const char *attrName,
                              llvm::DenseMap<int, SmallVector<std::pair<Operation *, int>>> &depsByGroup)
{
  // Attribute format: {ssbuffer.crossDeps/intraDeps = [group, role]}
  int ret = 0;
  int depSize = 2;

  rootOp->walk([&](Operation *op) {
    auto depsAttr = op->getAttrOfType<ArrayAttr>(attrName);
    if (!depsAttr)
      return;

    if (depsAttr.size() < depSize) {
      LDBG("format of dependency attribute error!");
      ret = -1;
      return;
    }

    int group = cast<IntegerAttr>(depsAttr[0]).getInt();
    int role = cast<IntegerAttr>(depsAttr[1]).getInt();
    depsByGroup[group].push_back({op, role});
  });

  return ret;
}

// Function: Build mapping from consumer to producer
// Input: Ops grouped by group ID, format: group -> [(op, role), ...]
//        role: 1=producer, 0=consumer
//        mainLoop: if not nullptr, only include consumers inside this mainLoop
// Output: result - Mapping from consumer Value to list of producer Values
// Return: 0 for success, -1 for failure
static int buildProducerConsumerMapping(
    llvm::DenseMap<int, SmallVector<std::pair<Operation *, int>>> &depsByGroup,
    llvm::DenseMap<Value, SmallVector<Value>> &result,
    scf::ForOp mainLoop = nullptr)
{
  for (auto &groupEntry : depsByGroup) {
    auto &ops = groupEntry.second;

    // Collect all producers and consumers in this group
    SmallVector<Operation *> producers;
    SmallVector<Operation *> consumers;

    for (auto &opRole : ops) {
      Operation *op = opRole.first;
      int role = opRole.second;
      if (role == producerId) {
        producers.push_back(op);
      } else if (role == consumerId) {
        // For intra-core mapping, only include consumers inside mainLoop
        if (mainLoop != nullptr) {
          if (isConsumerInMainLoop(op, mainLoop, consumers) != 0) {
            LDBG("isConsumerInMainLoop failed");
            return -1;
          }
        } else {
          consumers.push_back(op);
        }
      } else {
        LDBG("Get error role id in dependency attribute: OP: " << *op << ", role: " << role);
        return -1;
      }
    }

    // Skip if no consumers (for intra-core mapping with mainLoop filter)
    if (mainLoop != nullptr && consumers.empty())
      continue;

    // For each consumer, build mapping to all producers
    for (Operation *consumer : consumers) {
      for (Value consumerResult : consumer->getResults()) {
        SmallVector<Value> producerValues;
        for (Operation *producer : producers) {
          for (Value producerResult : producer->getResults()) {
            producerValues.push_back(producerResult);
          }
        }
        result[consumerResult] = producerValues;
      }
    }
  }

  return 0;
}

// Initialize crossCoreDependentMap (cross-core data dependency)
// Find ops with ssbuffer.crossDeps attribute
// Attribute value is a list: [group, role], role: 1=producer, 0=consumer
// Map key is consumer, value is list of all producers in the same group
// Return: 0 for success, -1 for failure
int initCrossCoreDependentMap(ModuleOp module, ControlFlowConditionInfo *info)
{
  llvm::DenseMap<int, SmallVector<std::pair<Operation *, int>>> crossDepsByGroup;
  if (collectDepsByGroup(module, "ssbuffer.crossDeps", crossDepsByGroup) != 0) {
    LDBG("collectDepsByGroup on crossDeps Failed!");
    return -1;
  }

  llvm::DenseMap<Value, SmallVector<Value>> crossDepsMap;
  if (buildProducerConsumerMapping(crossDepsByGroup, crossDepsMap) != 0) {
    LDBG("buildProducerConsumerMapping on crossDeps Failed!");
    return -1;
  }
  info->crossCoreDependentMap = crossDepsMap;
  return 0;
}

// Initialize intraCoreDependentMap (intra-core data dependency)
// Find forOp with ssbuffer.main_loop attribute
// Collect all intra-core deps from module (producers may be outside the loop)
// For each mainLoop, filter consumers that are inside it (not in nested mainloops)
// Return: 0 for success, -1 for failure
int initIntraCoreDependentMap(ModuleOp module, ControlFlowConditionInfo *info)
{
  // Collect all intra-core deps from the entire module
  llvm::DenseMap<int, SmallVector<std::pair<Operation *, int>>> allIntraDepsByGroup;
  if (collectDepsByGroup(module, "ssbuffer.intraDeps", allIntraDepsByGroup) != 0) {
    LDBG("collectDepsByGroup on intraDeps Failed!");
    return -1;
  }

  // For each mainLoop, build mapping with consumers inside it
  int ret = 0;
  module.walk([&](Operation* op) {
    if (!op->hasAttr("ssbuffer.main_loop"))
      return;
    auto forOp = dyn_cast<scf::ForOp>(op);
    if (!forOp) {
      LDBG("Do not support other mainloop except forOp!");
      ret = -1;
      return;
    }

    llvm::DenseMap<Value, SmallVector<Value>> depMap;
    if (buildProducerConsumerMapping(allIntraDepsByGroup, depMap, forOp) != 0) {
      LDBG("buildProducerConsumerMapping on intraDeps Failed!");
      ret = -1;
      return;
    }

    // Only insert if there are dependencies for this mainLoop
    if (!depMap.empty()) {
      info->intraCoreDependentMap[forOp] = depMap;
    }
  });
  return ret;
}

// Print all dependent maps for verification
static void printDependentMaps(ControlFlowConditionInfo *info)
{
  // Print crossCoreDependentMap
  LDBG("crossCoreDependentMap size: " << info->crossCoreDependentMap.size());
  LDBG("crossCoreDependentMap contents:");
  for (auto &entry : info->crossCoreDependentMap) {
      Value consumer = entry.first;
      SmallVector<Value> &producers = entry.second;
      LDBG("    Consumer: " << consumer << " (producers count: " << producers.size() << ")");
      for (Value producer : producers) {
          LDBG("      Producer: " << producer);
      }
  }

  // Print intraCoreDependentMap
  LDBG("intraCoreDependentMap size: " << info->intraCoreDependentMap.size());
  LDBG("intraCoreDependentMap contents:");
  for (auto &forEntry : info->intraCoreDependentMap) {
      scf::ForOp forOp = forEntry.first;
      auto &depMap = forEntry.second;
      LDBG("  ForOp (depMap size: " << depMap.size() << "):");
      LDBG("    ");
      forOp->print(llvm::dbgs(), OpPrintingFlags().skipRegions());

      for (auto &entry : depMap) {
          Value consumer = entry.first;
          SmallVector<Value> &producers = entry.second;
          LDBG("    Consumer: " << consumer << " (producers count: " << producers.size() << ")");
          for (Value producer : producers) {
              LDBG("      Producer: " << producer);
          }
      }
  }
}

void InitDependentMapPass::runOnOperation()
{
    ModuleOp module = getOperation();
    LDBG("Enter InitDependentMap pass.");

    // Step 1: Initialize crossCoreDependentMap
    if (initCrossCoreDependentMap(module, info) != 0) {
        LDBG("initCrossCoreDependentMap failed!");
        signalPassFailure();
        return;
    }

    // Step 2: Initialize intraCoreDependentMap
    if (initIntraCoreDependentMap(module, info) != 0) {
        LDBG("initIntraCoreDependentMap failed!");
        signalPassFailure();
        return;
    }

    // Print all dependent maps for verification
    printDependentMaps(info);

    LDBG("Exit InitDependentMap pass.");
}

namespace mlir {
namespace triton {
std::unique_ptr<OperationPass<ModuleOp>> createInitDependentMapPass()
{
  return std::make_unique<InitDependentMapPass>();
}
} // namespace triton
} // namespace mlir