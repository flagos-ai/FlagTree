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

#include "ascend/include/DynamicCVPipeline/SplitDataflow/FlagIdReuse.h"

#include <algorithm>
#include <cstddef>
#include <limits>
#include <optional>

#include "DynamicCVPipeline/Common/MemoryEffectsTracker.h"
#include "bishengir/Dialect/HIVM/IR/HIVM.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/Operation.h"
#include "mlir/Support/LLVM.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallSet.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Casting.h"

static constexpr const char *DEBUG_TYPE = "flag-id-reuse";
#define LOG_DEBUG(...)                                                         \
  LLVM_DEBUG(llvm::dbgs() << " [" << DEBUG_TYPE << "] " << __VA_ARGS__)

using namespace mlir;
using namespace mlir::triton;
using namespace hivm;

void FlagIdReuseManager::insertRelationBetweenSetAndWait(Operation *before,
                                                         Operation *after) {
  if (!before || !after) {
    LOG_DEBUG("Warning: inserting a relation with null operation, ignored.");
    return;
  }

  LOG_DEBUG("Add an edge from \n" << *before << " to \n" << *after << "\n");
  relations[before].push_back(after);
  return;
}

DenseMap<int, int> FlagIdReuseManager::reuseInterCoreTransferFlagIds(
    const llvm::SmallVector<Operation *> &syncOps) {
  DenseMap<int, int> remapResult;
  if (syncOps.empty()) {
    return remapResult;
  }

  preworkForAnalyze(syncOps);
  return colorInterferenceGraph();
}

void FlagIdReuseManager::preworkForAnalyze(
    const llvm::SmallVector<Operation *> &syncOps) {
  flagIdToOps.clear();
  opOrder.clear();
  // syncOps arrive in module-walk (program) order, so the index is a stable
  // program-order rank used to pick the earliest set / latest wait of a flag.
  int order = 0;
  for (auto op : syncOps) {
    opOrder[op] = order++;
    auto flagId = getFlagId(op);
    if (flagId == -1) {
      continue;
    }
    flagIdToOps[flagId].push_back(op);
  }
}

Operation *FlagIdReuseManager::getEarliestSet(int flagId) {
  Operation *earliest = nullptr;
  for (auto op : flagIdToOps[flagId]) {
    if (!llvm::isa<SyncBlockSetOp>(op)) {
      continue;
    }
    if (!earliest || opOrder[op] < opOrder[earliest]) {
      earliest = op;
    }
  }
  return earliest;
}

Operation *FlagIdReuseManager::getLatestWait(int flagId) {
  Operation *latest = nullptr;
  for (auto op : flagIdToOps[flagId]) {
    if (!llvm::isa<SyncBlockWaitOp>(op)) {
      continue;
    }
    if (!latest || opOrder[op] > opOrder[latest]) {
      latest = op;
    }
  }
  return latest;
}

bool FlagIdReuseManager::opPrecedes(Operation *p, Operation *q) {
  if (!p || !q) {
    return false;
  }
  if (p == q) {
    return true;
  }
  // Same block: MLIR static program order is an exact, sound ordering and
  // covers straight-line chains and sibling loops without needing a graph edge.
  if (p->getBlock() == q->getBlock()) {
    return p->isBeforeInBlock(q);
  }
  // Cross block: only the real happens-before edges (E1 per-pipe FIFO, E2
  // set->wait, E3 data, E4 read-wait->consumed data) may prove ordering.
  llvm::SmallSet<Operation *, CVPipeline::INIT_SIZE> visited;
  return hasPath(visited, p, q);
}

// `before` is fully released before `after` is acquired iff the latest wait of
// `before` is ordered before the earliest set of `after`.
bool FlagIdReuseManager::flagReleasedBefore(int before, int after) {
  Operation *release = getLatestWait(before);
  Operation *acquire = getEarliestSet(after);
  if (!release || !acquire) {
    return false;
  }
  return opPrecedes(release, acquire);
}

// A transfer's data-movement direction, read off its sync pipes. Cube->Vector
// transfers (fixpipe) ride the FIX/V pipes; Vector->Cube transfers (copy) ride
// the MTE*/M pipes. The two never share a pipe, so a single pipe decides.
FlagIdReuseManager::SyncDir FlagIdReuseManager::getFlagDirection(int flagId) {
  for (auto *op : flagIdToOps[flagId]) {
    PipeAttr srcPipe, dstPipe;
    if (auto setOp = llvm::dyn_cast<SyncBlockSetOp>(op)) {
      srcPipe = setOp.getTpipeAttr();
      dstPipe = setOp.getPipeAttr();
    } else if (auto waitOp = llvm::dyn_cast<SyncBlockWaitOp>(op)) {
      srcPipe = waitOp.getTpipeAttr();
      dstPipe = waitOp.getPipeAttr();
    }
    for (PipeAttr pipe : {srcPipe, dstPipe}) {
      if (pipe && (pipe.getPipe() == hivm::PIPE::PIPE_FIX ||
                   pipe.getPipe() == hivm::PIPE::PIPE_V)) {
        return SyncDir::CubeToVector;
      }
    }
  }
  return SyncDir::VectorToCube;
}

// Conservative: two flags interfere unless one is provably released before the
// other is acquired. Any uncertainty falls back to "interfere" (no reuse), so a
// reused id is never shared by two concurrently-live transfers.
bool FlagIdReuseManager::flagsInterfere(int lhs, int rhs) {
  // Opposite-direction transfers (Vector->Cube vs Cube->Vector) execute
  // concurrently on the two cores, so their flag lifetimes always overlap
  if (getFlagDirection(lhs) != getFlagDirection(rhs)) {
    return true;
  }
  return !flagReleasedBefore(lhs, rhs) && !flagReleasedBefore(rhs, lhs);
}

DenseMap<int, int> FlagIdReuseManager::colorInterferenceGraph() {
  // Flags ordered by first program appearance, for deterministic coloring and
  // compact renumbering.
  llvm::SmallVector<int> flagIds;
  for (auto &[flagId, ops] : flagIdToOps) {
    if (!ops.empty()) {
      flagIds.push_back(flagId);
    }
  }
  auto firstOrder = [&](int flagId) {
    int best = std::numeric_limits<int>::max();
    for (auto op : flagIdToOps[flagId]) {
      best = std::min(best, opOrder[op]);
    }
    return best;
  };
  llvm::sort(flagIds,
             [&](int a, int b) { return firstOrder(a) < firstOrder(b); });

  // Build the (undirected) interference graph.
  DenseMap<int, llvm::SmallVector<int>> adj;
  for (size_t i = 0; i < flagIds.size(); ++i) {
    for (size_t j = i + 1; j < flagIds.size(); ++j) {
      if (flagsInterfere(flagIds[i], flagIds[j])) {
        adj[flagIds[i]].push_back(flagIds[j]);
        adj[flagIds[j]].push_back(flagIds[i]);
      }
    }
  }

  // Welsh-Powell greedy: color highest-degree nodes first (tie: earlier flag),
  // each takes the smallest color unused by its already-colored neighbours.
  llvm::SmallVector<int> order(flagIds.begin(), flagIds.end());
  llvm::sort(order, [&](int a, int b) {
    if (adj[a].size() != adj[b].size()) {
      return adj[a].size() > adj[b].size();
    }
    return firstOrder(a) < firstOrder(b);
  });

  DenseMap<int, int> rawColor;
  for (int flagId : order) {
    llvm::SmallSet<int, CVPipeline::INIT_SIZE> usedByNeighbours;
    for (int nb : adj[flagId]) {
      auto it = rawColor.find(nb);
      if (it != rawColor.end()) {
        usedByNeighbours.insert(it->second);
      }
    }
    int color = 1;
    while (usedByNeighbours.contains(color)) {
      ++color;
    }
    rawColor[flagId] = color;
  }

  // Compact renumber: walk flags in program order, assign 1,2,3,... to colors
  // on first use, so output flag ids are minimal and stably ordered.
  DenseMap<int, int> colorToCompact;
  int nextCompact = 1;
  DenseMap<int, int> remapResult;
  for (int flagId : flagIds) {
    int color = rawColor[flagId];
    if (!colorToCompact.contains(color)) {
      colorToCompact[color] = nextCompact++;
    }
    remapResult[flagId] = colorToCompact[color];
  }
  return remapResult;
}

bool FlagIdReuseManager::hasPath(
    llvm::SmallSet<Operation *, CVPipeline::INIT_SIZE> &visited,
    Operation *from, Operation *to) {
  if (from == to) {
    return true;
  }
  if (visited.contains(from)) {
    return false;
  }
  visited.insert(from);
  for (auto nxt : relations[from]) {
    if (hasPath(visited, nxt, to)) {
      return true;
    }
  }
  return false;
}

int FlagIdReuseManager::getFlagId(Operation *op) {
  if (auto setOp = llvm::dyn_cast<SyncBlockSetOp>(op)) {
    if (auto staticFlagId = setOp.getStaticFlagId()) {
      return staticFlagId->getInt();
    }
    LOG_DEBUG(
        "Warning: SyncBlockSetOp has no static flagId (dynamic?), skipped: \n"
        << *op << "\n");
    return -1;
  }
  if (auto waitOp = llvm::dyn_cast<SyncBlockWaitOp>(op)) {
    if (auto staticFlagId = waitOp.getStaticFlagId()) {
      return staticFlagId->getInt();
    }
    LOG_DEBUG(
        "Warning: SyncBlockWaitOp has no static flagId (dynamic?), skipped: \n"
        << *op << "\n");
    return -1;
  }
  LOG_DEBUG("Warning: failed to get flagId from op: \n" << *op << "\n");
  return -1;
}
