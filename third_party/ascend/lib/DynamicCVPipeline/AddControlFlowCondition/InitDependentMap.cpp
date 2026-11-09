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
#include "ascend/include/DynamicCVPipeline/Common/BufferCountManager.h"
#include "bishengir/Dialect/Annotation/IR/Annotation.h"
#include "bishengir/Dialect/HIVM/IR/HIVM.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "third_party/ascend/include/DynamicCVPipeline/AddControlFlowCondition.h"
#include "third_party/ascend/include/DynamicCVPipeline/Common/Utils.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Debug.h"

#include "bishengir/Dialect/Annotation/IR/Annotation.h"
#include "bishengir/Dialect/HIVM/IR/HIVM.h"
#include "bishengir/Dialect/Scope/IR/Scope.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/BuiltinAttributes.h"

static constexpr const char *DEBUG_TYPE = "InitDependentMap";
#define DBGS() (llvm::dbgs() << '[' << DEBUG_TYPE << "] ")
#define LDBG(...) LLVM_DEBUG(DBGS() << __VA_ARGS__ << "\n")

using namespace mlir;
using namespace triton;
using namespace CVPipeline;
using namespace hivm;

// Returns 0 if `consumer` is inside `mainLoop` (and pushes it to `consumers`)
// or inside a nested mainloop (skip), or -1 on error.
static int isConsumerInMainLoop(Operation *consumer, Operation *mainLoop,
                                SmallVector<Operation *> &consumers) {
  Operation *current = consumer->getParentOp();

  // Traverse up the parent chain until we reach the top (nullptr)
  while (current != nullptr) {
    if (isMainLoopOp(current) && current != mainLoop) {
      // consumer Op not in the current mainloop
      return 0;
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

// Collect ops with dependency attr `attrName` into depsByGroup (group ->
// [(op, role)]). Attr is a flat list of pairs: [group, role, group, role, ...].
// role: 1=producer, 0=consumer. One op may contribute multiple pairs (dual role
// or several groups). 0 ok, -1 fail.
static int
collectDepsByGroup(Operation *rootOp, const char *attrName,
                   llvm::DenseMap<int, SmallVector<std::pair<Operation *, int>>>
                       &depsByGroup) {
  int ret = 0;
  // One record is [group, role]; the attr concatenates N such records.
  int depSize = 2;
  int groupOffset = 0;
  int roleOffset = 1;

  rootOp->walk([&](Operation *op) {
    auto depsAttr = op->getAttrOfType<ArrayAttr>(attrName);
    if (!depsAttr)
      return;

    if (depsAttr.empty() || depsAttr.size() % depSize != 0) {
      LDBG("format of dependency attribute error, expect even-length pairs!");
      ret = -1;
      return;
    }

    for (size_t i = 0; i < depsAttr.size(); i += depSize) {
      Attribute groupAttr = depsAttr[i + groupOffset];
      Attribute roleAttr = depsAttr[i + roleOffset];
      if (!isa<IntegerAttr>(groupAttr) || !isa<IntegerAttr>(roleAttr)) {
        LDBG("type of dependency attritbute is not Int! error op:" << *op);
        ret = -1;
        return;
      }

      int group = cast<IntegerAttr>(groupAttr).getInt();
      int role = cast<IntegerAttr>(roleAttr).getInt();
      depsByGroup[group].push_back({op, role});
    }
  });

  return ret;
}

// Cross-core: consumer -> one producer list per group it consumes. An op may
// be both roles and/or in several groups; each group keeps its own list.
static int buildCrossCoreProducerConsumerMapping(
    llvm::DenseMap<int, SmallVector<std::pair<Operation *, int>>> &depsByGroup,
    ConsumerProducerMap &result) {
  for (auto &groupEntry : depsByGroup) {
    auto &ops = groupEntry.second;

    SmallVector<Operation *> producers;
    SmallVector<Operation *> consumers;

    for (auto &opRole : ops) {
      Operation *op = opRole.first;
      int role = opRole.second;
      if (role == CVPipeline::crossCoreProducerId) {
        if (!llvm::is_contained(producers, op))
          producers.push_back(op);
      } else if (role == CVPipeline::crossCoreConsumerId) {
        if (!llvm::is_contained(consumers, op))
          consumers.push_back(op);
      } else {
        LDBG("Get error role id in dependency attribute: OP: "
             << *op << ", role: " << role);
        return -1;
      }
    }

    for (Operation *consumer : consumers) {
      result[consumer].push_back(producers);
    }
  }

  return 0;
}

// Build consumer -> producers mapping from depsByGroup (role 1=producer,
// 0=consumer); if mainLoop != nullptr only consumers inside it. 0 ok, -1 fail.
static int buildIntraCoreProducerConsumerMapping(
    llvm::DenseMap<int, SmallVector<std::pair<Operation *, int>>> &depsByGroup,
    llvm::DenseMap<Operation *, SmallVector<Operation *>> &result,
    Operation *mainLoop = nullptr) {
  for (auto &groupEntry : depsByGroup) {
    auto &ops = groupEntry.second;

    // Collect all producers and consumers in this group
    SmallVector<Operation *> producers;
    SmallVector<Operation *> consumers;

    for (auto &opRole : ops) {
      Operation *op = opRole.first;
      int role = opRole.second;
      if (role == CVPipeline::crossCoreProducerId) {
        producers.push_back(op);
      } else if (role == CVPipeline::crossCoreConsumerId) {
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
        LDBG("Get error role id in dependency attribute: OP: "
             << *op << ", role: " << role);
        return -1;
      }
    }

    // Skip if no consumers (for intra-core mapping with mainLoop filter)
    if (mainLoop != nullptr && consumers.empty())
      continue;

    // For each consumer, build mapping to all producers
    for (Operation *consumer : consumers) {
      result[consumer] = producers;
    }
  }

  return 0;
}

// Collects every scf.for/scf.while op tagged CVPipeline::kMainLoop, stored
// uniformly as Operation* so downstream lookups ignore the op kind.
static int collectMainLoopById(ModuleOp module,
                               llvm::DenseMap<Operation *, int> &mainLoopById) {
  int ret = 0;
  module.walk([&](Operation *op) {
    if (!op->hasAttr(CVPipeline::kMainLoop))
      return;
    if (!isMainLoopOp(op)) {
      LDBG("Do not support mainloop op other than scf.for or scf.while: "
           << op->getName());
      ret = -1;
      return;
    }
    auto mainLoopIdAttr = op->getAttrOfType<IntegerAttr>(CVPipeline::kMainLoop);
    if (mainLoopIdAttr) {
      mainLoopById[op] = mainLoopIdAttr.getInt();
    }
  });
  return ret;
}

static int
findMainLoopIdContainingOp(Operation *op,
                           llvm::DenseMap<Operation *, int> &mainLoopById) {
  for (auto &entry : mainLoopById) {
    if (entry.first->isAncestor(op)) {
      return entry.second;
    }
  }
  return -1;
}

static int
filterMemCrossCoreDepsByMainLoop(ModuleOp module,
                                 ConsumerProducerMap &initialDepsMap,
                                 ConsumerProducerMap &filteredDepsMap) {
  LDBG("memCrossCore dependencies before filter: "
       << countProducerGroups(initialDepsMap));

  // Step 1: Collect all main_loop ops (scf.for or scf.while) and their ids
  llvm::DenseMap<Operation *, int> mainLoopById;
  if (collectMainLoopById(module, mainLoopById) != 0) {
    LDBG("collectMainLoopById Failed!");
    return -1;
  }

  // Step 2: Filter mapping - only keep producer/consumer pairs in the same
  // main_loop. Each inner producer list is one group and is filtered on its
  // own so a multi-group consumer can keep some groups and drop others.
  for (auto &entry : initialDepsMap) {
    Operation *consumer = entry.first;

    // Find the main_loop id containing the consumer
    int consumerMainLoopId = findMainLoopIdContainingOp(consumer, mainLoopById);
    if (consumerMainLoopId == -1) {
      LDBG("Consumer op is not in any main_loop, skip: " << *consumer);
      continue;
    }

    for (SmallVector<Operation *> &producers : entry.second) {
      if (producers.empty()) {
        LDBG("Producers list is empty!");
        return -1;
      }

      // Find the main_loop id containing the producer
      int producerMainLoopId =
          findMainLoopIdContainingOp(producers[0], mainLoopById);
      if (producerMainLoopId == -1) {
        LDBG("producer op is not in any main_loop: " << *producers[0]);
        continue;
      }

      // Check all producers in the same mainloop
      for (size_t i = 1; i < producers.size(); i++) {
        int otherProducerMainLoopId =
            findMainLoopIdContainingOp(producers[i], mainLoopById);
        if (otherProducerMainLoopId != producerMainLoopId) {
          LDBG("Producers are not in the same main_loop. "
               << "First producer main_loop id: " << producerMainLoopId
               << ", Producer[" << i
               << "] main_loop id: " << otherProducerMainLoopId);
          return -1;
        }
      }

      // Check if consumer and producers are in the same main_loop
      if (consumerMainLoopId != producerMainLoopId) {
        LDBG("Consumer and producers are in different main_loop, skip. "
             << "Consumer main_loop id: " << consumerMainLoopId
             << ", Producer main_loop id: " << producerMainLoopId);
        continue;
      }

      filteredDepsMap[consumer].push_back(producers);
    }
  }

  LDBG("memCrossCore dependencies after filter: "
       << countProducerGroups(filteredDepsMap));

  return 0;
}

// Init crossCoreDependentMap from ssbuffer.crossCoreDeps pairs
// [group, role, ...]; 1=producer 0=consumer. Consumer -> one producer list per
// group, same main_loop. 0 ok, -1 fail.
int initCrossCoreDependentMap(ModuleOp module, ControlFlowConditionInfo *info) {
  // Step 1: Collect all crossDeps by group (including memCrossDeps)
  llvm::DenseMap<int, SmallVector<std::pair<Operation *, int>>>
      crossDepsByGroup;
  if (collectDepsByGroup(module, CVPipeline::kCrossCoreDeps.data(),
                         crossDepsByGroup) != 0) {
    LDBG("collectDepsByGroup on crossDeps Failed!");
    return -1;
  }

  // Step 2: Build initial mapping (all producers for each consumer)
  ConsumerProducerMap initialCrossDepsMap;
  if (buildCrossCoreProducerConsumerMapping(crossDepsByGroup,
                                            initialCrossDepsMap) != 0) {
    LDBG("buildCrossCoreProducerConsumerMapping on crossDeps Failed!");
    return -1;
  }

  // Step 3: Filter by main_loop constraint
  ConsumerProducerMap filteredCrossDepsMap;
  if (filterMemCrossCoreDepsByMainLoop(module, initialCrossDepsMap,
                                       filteredCrossDepsMap) != 0) {
    LDBG("filterCrossCoreDepsByMainLoop Failed!");
    return -1;
  }

  info->crossCoreDependentMap = filteredCrossDepsMap;
  return 0;
}

// Initializes intraCoreDependentMap: per main_loop op (scf.for/scf.while) keeps
// consumers inside it but not in nested mainloops. 0 on success, -1 on error.
int initIntraCoreDependentMap(ModuleOp module, ControlFlowConditionInfo *info) {
  // Collect all intra-core deps from the entire module
  llvm::DenseMap<int, SmallVector<std::pair<Operation *, int>>>
      allIntraDepsByGroup;
  if (collectDepsByGroup(module, CVPipeline::kIntraDeps.data(),
                         allIntraDepsByGroup) != 0) {
    LDBG("collectDepsByGroup on intraDeps Failed!");
    return -1;
  }

  // For each mainLoop, build the mapping with consumers inside it; both scf.for
  // and scf.while can carry ssbuffer.main_loop (map keyed on Operation*).
  int ret = 0;
  module.walk([&](Operation *op) {
    if (!op->hasAttr(CVPipeline::kMainLoop))
      return;
    if (!isMainLoopOp(op)) {
      LDBG("Do not support mainloop op other than scf.for or scf.while: "
           << op->getName());
      ret = -1;
      return;
    }

    llvm::DenseMap<Operation *, SmallVector<Operation *>> depMap;
    if (buildIntraCoreProducerConsumerMapping(allIntraDepsByGroup, depMap,
                                              op) != 0) {
      LDBG("buildIntraCoreProducerConsumerMapping on intraDeps Failed!");
      ret = -1;
      return;
    }

    // Only insert if there are dependencies for this mainLoop
    if (!depMap.empty()) {
      info->intraCoreDependentMap[op] = depMap;
    }
  });
  return ret;
}

// Print all dependent maps for verification
static void printDependentMaps(ControlFlowConditionInfo *info) {
  // Print crossCoreDependentMap
  LDBG("crossCoreDependentMap size: "
       << countProducerGroups(info->crossCoreDependentMap));
  LDBG("crossCoreDependentMap contents:");
  for (auto &entry : info->crossCoreDependentMap) {
    Operation *consumer = entry.first;
    LDBG("    Consumer: " << *consumer << " (groups: " << entry.second.size()
                          << ")");
    for (SmallVector<Operation *> &producers : entry.second) {
      LDBG("      group producers count: " << producers.size());
      for (Operation *producer : producers) {
        LDBG("        Producer: " << *producer);
      }
    }
  }

  // Print intraCoreDependentMap
  LDBG("intraCoreDependentMap size: " << info->intraCoreDependentMap.size());
  LDBG("intraCoreDependentMap contents:");
  for (auto &forEntry : info->intraCoreDependentMap) {
    Operation *loopOp = forEntry.first;
    auto &depMap = forEntry.second;
    LDBG("  MainLoopOp (depMap size: " << depMap.size() << "):");
    LDBG("    " << OpWithFlags(loopOp, OpPrintingFlags().skipRegions()));

    for (auto &entry : depMap) {
      Operation *consumer = entry.first;
      SmallVector<Operation *> &producers = entry.second;
      LDBG("    Consumer: " << *consumer << " (producers count: "
                            << producers.size() << ")");
      for (Operation *producer : producers) {
        LDBG("      Producer: " << *producer);
      }
    }
  }
}

// Find the IfOp that contains a given operation
static scf::IfOp findIfOpContainingOp(Operation *op) {
  if (!op) {
    return nullptr;
  }

  constexpr int maxDepth = 100;
  int depth = 0;

  Operation *current = op;
  while (current && depth < maxDepth) {
    if (auto ifOp = dyn_cast<scf::IfOp>(current)) {
      if (ifOp->hasAttr(CVPipeline::kIf)) {
        LDBG("Found ssbuffer.if at depth " << depth);
        return ifOp;
      }
    }
    current = current->getParentOp();
    depth++;
  }

  if (depth >= maxDepth) {
    LDBG("Warning: Max depth " << maxDepth
                               << " exceeded in findIfOpContainingOp");
  }

  return nullptr;
}

// Compute producer buffer counts (max map size) from cross/intra-core maps;
// falls back to BufferCountManager IntraCore when the intra-core map is empty.
static void computeProducerBufferCount(ControlFlowConditionInfo *info,
                                       ModuleOp module) {
  // Get cross-core buffer count (max size in the map)
  info->crossCoreBufferCount = 0;
  for (auto &entry : info->crossCoreDependentMap) {
    for (SmallVector<Operation *> &producers : entry.second) {
      info->crossCoreBufferCount =
          std::max(info->crossCoreBufferCount, (int)producers.size());
    }
  }
  LDBG("Cross-core buffer count (max): " << info->crossCoreBufferCount);

  // Get intra-core buffer count (max size across all main loops)
  info->intraCoreBufferCount = 0;
  for (auto &loopEntry : info->intraCoreDependentMap) {
    auto &intraDepMap = loopEntry.second;
    for (auto &entry : intraDepMap) {
      info->intraCoreBufferCount =
          std::max(info->intraCoreBufferCount, (int)entry.second.size());
    }
  }
  LDBG("Intra-core buffer count (max across all main loops): "
       << info->intraCoreBufferCount);

  // If intra-core map is empty, use BufferCountManager's IntraCore value
  if (info->intraCoreBufferCount == 0) {
    BufferCountManager bufferCountMgr(module);
    info->intraCoreBufferCount = bufferCountMgr.getBufferCountByType(
        BufferCountManager::DepType::IntraCore);
    LDBG("Intra-core map is empty, using BufferCountManager IntraCore value: "
         << info->intraCoreBufferCount);
  }
}

// Add an edge to ifBlockDAG if it is not already present
static void addIfBlockEdge(ControlFlowConditionInfo *info, scf::IfOp producerIf,
                           scf::IfOp consumerIf, IfBlockDepKind kind) {
  auto &edges = info->ifBlockDAG[producerIf];
  bool seen =
      llvm::any_of(edges, [&](const std::pair<scf::IfOp, IfBlockDepKind> &e) {
        return e.first == consumerIf;
      });
  if (!seen) {
    edges.push_back({consumerIf, kind});
  }
}

// Build if block DAG from crossCoreDependentMap and intraCoreDependentMap
static int buildIfBlockDAG(ModuleOp module, ControlFlowConditionInfo *info) {
  // Step 1: Traverse crossCoreDependentMap to add cross-core edges.
  for (auto &entry : info->crossCoreDependentMap) {
    Operation *consumerOp = entry.first;

    // Consumer op is inside an if block
    scf::IfOp consumerIf = findIfOpContainingOp(consumerOp);
    if (!consumerIf) {
      LDBG("Consumer op not in any ssbuffer.if block: " << *consumerOp);
      return -1;
    }

    // Each inner list is one dependency group
    for (SmallVector<Operation *> &producers : entry.second) {
      for (Operation *producerOp : producers) {
        scf::IfOp producerIf = findIfOpContainingOp(producerOp);
        if (!producerIf) {
          LDBG("Producer op not in any ssbuffer.if block: " << *producerOp);
          return -1;
        }

        if (producerIf == consumerIf) {
          LDBG("Producer and consumer are in the same if block, this is "
               "invalid: "
               << *producerIf);
          return -1;
        }

        addIfBlockEdge(info, producerIf, consumerIf, IfBlockDepKind::CrossCore);
      }
    }
  }

  // Step 2: Traverse intraCoreDependentMap to add intra-core edges.
  for (auto &loopEntry : info->intraCoreDependentMap) {
    for (auto &consumerProducers : loopEntry.second) {
      Operation *consumerOp = consumerProducers.first;
      scf::IfOp consumerIf = findIfOpContainingOp(consumerOp);
      if (!consumerIf) {
        LDBG("Intra consumer op not in any ssbuffer.if block: " << *consumerOp);
        return -1;
      }

      for (Operation *producerOp : consumerProducers.second) {
        scf::IfOp producerIf = findIfOpContainingOp(producerOp);
        if (!producerIf) {
          LDBG("Intra producer op not in any ssbuffer.if block: "
               << *producerOp);
          return -1;
        }

        if (producerIf == consumerIf) {
          LDBG("Intra producer and consumer are in the same if block, this "
               "is invalid: "
               << *producerIf);
          return -1;
        }

        addIfBlockEdge(info, producerIf, consumerIf, IfBlockDepKind::IntraCore);
      }
    }
  }

  return 0;
}

// Detect cycle in the if-block DAG via DFS over all edges.
enum class DfsState : uint8_t { Unvisited, Visiting, Done };

static bool
dfsCycle(scf::IfOp node,
         llvm::DenseMap<scf::IfOp,
                        llvm::SmallVector<std::pair<scf::IfOp, IfBlockDepKind>>>
             &dag,
         llvm::DenseMap<scf::IfOp, DfsState> &state) {
  state[node] = DfsState::Visiting;
  auto it = dag.find(node);
  if (it != dag.end()) {
    for (auto &edge : it->second) {
      scf::IfOp neighbor = edge.first;
      auto s = state.lookup(neighbor);
      if (s == DfsState::Visiting)
        return true;
      if (s == DfsState::Unvisited && dfsCycle(neighbor, dag, state))
        return true;
    }
  }
  state[node] = DfsState::Done;
  return false;
}

static int detectCycle(ControlFlowConditionInfo *info) {
  // Collect all nodes in the DAG (both producers and consumers)
  llvm::DenseMap<scf::IfOp, DfsState> state;
  for (auto &entry : info->ifBlockDAG) {
    state.try_emplace(entry.first, DfsState::Unvisited);
    for (auto &edge : entry.second) {
      state.try_emplace(edge.first, DfsState::Unvisited);
    }
  }

  for (auto &entry : state) {
    if (entry.second == DfsState::Unvisited) {
      if (dfsCycle(entry.first, info->ifBlockDAG, state)) {
        LDBG("Cycle detected in DAG");
        return -1;
      }
    }
  }

  return 0;
}

// DFS helper function to find nodes at target distance from start node
static int collectFlowOptIfOpPairs(ModuleOp module,
                                   ControlFlowConditionInfo *info) {
  // Step 1: Collect every node appearing in the DAG (key or value).
  llvm::DenseSet<scf::IfOp> allNodes;
  for (auto &entry : info->ifBlockDAG) {
    allNodes.insert(entry.first);
    for (auto &edge : entry.second) {
      allNodes.insert(edge.first);
    }
  }

  // Step 2: Compute in-degree (every edge, cross or intra, contributes 1).
  llvm::DenseMap<scf::IfOp, int> inDegree;
  for (auto &entry : info->ifBlockDAG) {
    for (auto &edge : entry.second) {
      inDegree[edge.first]++;
    }
  }

  // Step 3: Identify start nodes (in-degree 0).
  llvm::SmallVector<scf::IfOp> startNodes;
  for (auto node : allNodes) {
    if (inDegree.lookup(node) == 0) {
      startNodes.push_back(node);
    }
  }

  LDBG("Number of start nodes (depth=1): " << startNodes.size());

  // Step 4: For each start node, run a per-source DFS to compute each
  // reachable node's depth (max over all paths from this start). Every node
  // with depth = 3 becomes a flowOpt pair keyed on this start.
  constexpr int targetDepth = 3;
  for (scf::IfOp start : startNodes) {
    llvm::DenseMap<scf::IfOp, int> depth;
    llvm::SmallVector<std::pair<scf::IfOp, int>> worklist;
    worklist.push_back({start, 1});
    depth[start] = 1;

    while (!worklist.empty()) {
      auto [node, nodeDepth] = worklist.pop_back_val();
      auto it = info->ifBlockDAG.find(node);
      if (it == info->ifBlockDAG.end())
        continue;
      for (auto &edge : it->second) {
        scf::IfOp neighbor = edge.first;
        int step = (edge.second == IfBlockDepKind::CrossCore) ? 1 : 0;
        int candidate = nodeDepth + step;
        auto inserted = depth.try_emplace(neighbor, candidate);
        if (inserted.second) {
          worklist.push_back({neighbor, candidate});
        } else if (candidate > inserted.first->second) {
          inserted.first->second = candidate;
          worklist.push_back({neighbor, candidate});
        }
      }
    }

    for (auto &entry : depth) {
      if (entry.second == targetDepth) {
        info->flowOptIfOpPairs[entry.first] = start;
      }
    }
  }

  LDBG("flowOptIfOpPairs size: " << info->flowOptIfOpPairs.size());

  return 0;
}

// Print DAG and flowOpt pairs for verification
static void printDAGInfo(ControlFlowConditionInfo *info) {
  LDBG("ifBlockDAG contents:");
  for (auto &entry : info->ifBlockDAG) {
    scf::IfOp producer = entry.first;
    LDBG("  Producer IfOp has " << entry.second.size() << " consumers");
    for (auto &edge : entry.second) {
      LDBG("    -> Consumer IfOp (kind: "
           << (edge.second == IfBlockDepKind::CrossCore ? "CrossCore"
                                                        : "IntraCore")
           << ")");
    }
  }

  LDBG("flowOptIfOpPairs contents:");
  for (auto &entry : info->flowOptIfOpPairs) {
    scf::IfOp target = entry.first;
    scf::IfOp source = entry.second;
    LDBG("  Target IfOp (third node) -> Source IfOp (start node)");
  }
}

void InitDependentMapPass::runOnOperation() {
  ModuleOp module = getOperation();

  if (CVPipeline::hasFallbackAttr(module)) {
    return;
  }

  LDBG("Enter InitDependentMap pass.");

  // Step 1: Initialize crossCoreDependentMap
  if (initCrossCoreDependentMap(module, info) != 0) {
    LDBG("initCrossCoreDependentMap failed!");
    CVPipeline::setFallbackAttr(module, CVPipeline::ERRCODE_FAILED);
    return;
  }

  // Step 2: Initialize intraCoreDependentMap
  if (initIntraCoreDependentMap(module, info) != 0) {
    LDBG("initIntraCoreDependentMap failed!");
    CVPipeline::setFallbackAttr(module, CVPipeline::ERRCODE_FAILED);
    return;
  }

  // Print all dependent maps for verification
  LLVM_DEBUG(printDependentMaps(info));

  // Step 4: Compute producer buffer count for flowOpt condition
  computeProducerBufferCount(info, module);

  // Step 4: Build if block DAG from crossCoreDependentMap (always)
  if (buildIfBlockDAG(module, info) != 0) {
    LDBG("buildIfBlockDAG failed!");
    CVPipeline::setFallbackAttr(module, CVPipeline::ERRCODE_FAILED);
    return;
  }

  // Step 5: Detect cycle in DAG
  if (detectCycle(info) != 0) {
    LDBG("Cycle detected!");
    CVPipeline::setFallbackAttr(module, CVPipeline::ERRCODE_IGNORED);
    return;
  }

  // Step 6: Collect flowOpt if block pairs from DAG (only when buffer counts
  // exceed threshold)
  if (info->crossCoreBufferCount > CROSS_CORE_BUFFER_COUNT_THRESHOLD &&
      info->intraCoreBufferCount > INTRA_CORE_BUFFER_COUNT_THRESHOLD) {
    LDBG("Buffer counts meet requirements, collecting flowOpt pairs.");

    if (collectFlowOptIfOpPairs(module, info) != 0) {
      LDBG("collectFlowOptIfOpPairs failed!");
      CVPipeline::setFallbackAttr(module, CVPipeline::ERRCODE_FAILED);
      return;
    }

    LLVM_DEBUG(printDAGInfo(info));
  }

  LDBG("Exit InitDependentMap pass.");
}

namespace mlir {
namespace triton {
std::unique_ptr<OperationPass<ModuleOp>> createInitDependentMapPass() {
  return std::make_unique<InitDependentMapPass>();
}
} // namespace triton
} // namespace mlir
