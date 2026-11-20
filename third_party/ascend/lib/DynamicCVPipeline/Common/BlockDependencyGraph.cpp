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

#include "DynamicCVPipeline/Common/BlockDependencyGraph.h"
#include "DynamicCVPipeline/Common/DependencyHelper.h"
#include "DynamicCVPipeline/Common/MemoryEffectsTracker.h"
#include "DynamicCVPipeline/PlanComputeBlock/ComputeBlockIdManager.h"
#include "ascend/include/DynamicCVPipeline/Common/Utils.h"
#include "llvm/Support/Debug.h"
#include <queue>

using namespace mlir;
using namespace CVPipeline;

static constexpr const char *DEBUG_TYPE = "block-dependency-graph";
#define DBGS() (llvm::dbgs() << '[' << DEBUG_TYPE << "] ")
#define LDBG(...) LLVM_DEBUG(DBGS() << __VA_ARGS__ << "\n")

static bool isCubeSimpleOpOrCf(Operation *op) {
  return !isSyncOp(op) && getCoreTypeOfSimpleOpOrCf(op) == CoreType::CUBE_ONLY;
}

BlockDependencyGraph::BlockDependencyGraph(
    Block *block, const MemoryDependenceGraph &memGraph,
    ComputeBlockIdManager &bm)
    : block(block), memGraph(memGraph), bm(bm) {}

llvm::LogicalResult BlockDependencyGraph::buildGraph() {
  // Step 1: Collect all compute blocks (including cube and vector) in current
  // MLIR Block
  for (Operation &op : *block) {
    int blockId = bm.getBlockIdByOp(&op);
    if (blockId == -1)
      continue;

    if (!blockNodes.contains(blockId)) {
      auto node = std::make_unique<BlockNode>();
      node->blockId = blockId;
      node->isCube = isCubeSimpleOpOrCf(&op);
      node->depth = 0;
      blockNodes[blockId] = std::move(node);
    }

    blockNodes[blockId]->ops.push_back(&op);
  }

  LDBG("Collected " << blockNodes.size() << " blocks");

  // Step 2: Build dependency relationships between blocks (only in current MLIR
  // Block)
  DependencyHelper depHelper(memGraph);
  for (auto &entry : blockNodes) {
    BlockNode *consumerNode = entry.second.get();

    for (Operation *op : consumerNode->ops) {
      // Use DependencyHelper to iterate all source ops (SSA defs + memory
      // exec-before) in one pass.
      depHelper.forEachSource<DependencyHelper::SourceMode::Default>(
          op, [&](Operation *source) {
            if (!isInCurrentBlock(source))
              return;

            int sourceBlockId = bm.getBlockIdByOp(source);
            if (sourceBlockId == -1 || sourceBlockId == consumerNode->blockId)
              return;

            BlockNode *producerNode = getBlockNode(sourceBlockId);
            if (producerNode)
              addEdge(producerNode, consumerNode);
          });
    }
  }

  // Step 3: Compute depths (for all blocks)
  computeDepths();

  return llvm::success();
}

bool BlockDependencyGraph::isInCurrentBlock(Operation *op) {
  if (!op)
    return false;

  // Get the MLIR Block where op is located
  Block *opBlock = op->getBlock();

  // Check if it's in current MLIR Block
  return opBlock == block;
}

void BlockDependencyGraph::addEdge(BlockNode *from, BlockNode *to) {
  to->predecessors.insert(from);
  from->successors.insert(to);
}

void BlockDependencyGraph::computeDepths() {
  // Use topological sort to compute depth for each node
  // Depth definition: maximum steps from nodes with zero in-degree
  // Special rule: depth only increases when cube and vector type conversion
  // occurs

  llvm::DenseMap<BlockNode *, int> indegree;
  std::queue<BlockNode *> queue;

  // Initialize in-degree directly from each node's predecessor set.
  for (auto &entry : blockNodes) {
    BlockNode *node = entry.second.get();
    indegree[node] = node->predecessors.size();
    if (indegree[node] == 0) {
      queue.push(node);
      node->depth = 0;
    }
  }

  // BFS to compute depths
  while (!queue.empty()) {
    BlockNode *current = queue.front();
    queue.pop();

    for (BlockNode *succ : current->successors) {
      // Compute depth from current to succ
      // If types are different (cube->vector or vector->cube), depth+1
      // If types are the same, depth remains unchanged
      int newDepth = current->depth;
      if (current->isCube != succ->isCube) {
        newDepth++;
      }

      // Update successor's depth (take maximum)
      succ->depth = std::max(succ->depth, newDepth);

      indegree[succ]--;
      if (indegree[succ] == 0) {
        queue.push(succ);
      }
    }
  }

  LDBG("Computed depths for all blocks");
}

int BlockDependencyGraph::getDepth(int blockId) {
  auto it = blockNodes.find(blockId);
  if (it != blockNodes.end()) {
    return it->second->depth;
  }
  return -1;
}

BlockNode *BlockDependencyGraph::getBlockNode(int blockId) {
  auto it = blockNodes.find(blockId);
  if (it != blockNodes.end()) {
    return it->second.get();
  }
  return nullptr;
}

llvm::SmallVector<BlockNode *>
BlockDependencyGraph::getPredecessors(BlockNode *node) {
  if (!node)
    return {};
  return llvm::SmallVector<BlockNode *>(node->predecessors.begin(),
                                        node->predecessors.end());
}

llvm::SmallVector<BlockNode *>
BlockDependencyGraph::getSuccessors(BlockNode *node) {
  if (!node)
    return {};
  return llvm::SmallVector<BlockNode *>(node->successors.begin(),
                                        node->successors.end());
}

void BlockDependencyGraph::recomputeDepthsFrom(BlockNode *start) {
  // Walk forward through the successor chain starting at `start`. For each
  // visited node, the new depth is propagated to its successors; a successor
  // is enqueued again only when its depth actually grows. Since depth is
  // monotonically non-decreasing, this terminates.
  std::queue<BlockNode *> queue;
  queue.push(start);
  while (!queue.empty()) {
    BlockNode *current = queue.front();
    queue.pop();

    for (BlockNode *succ : current->successors) {
      int newDepth = current->depth;
      if (current->isCube != succ->isCube)
        newDepth++;
      if (succ->depth < newDepth) {
        succ->depth = newDepth;
        queue.push(succ);
      }
    }
  }

  LDBG("Recomputed depths from block " << start->blockId);
}

llvm::LogicalResult BlockDependencyGraph::rebuildAfterMerge(BlockNode *target,
                                                            BlockNode *source) {
  // Merge node information: append source ops into target.
  target->ops.append(source->ops.begin(), source->ops.end());

  // Sanity check: source must be connected to the graph (either has at
  // least one predecessor or one successor). Merging an isolated node is
  // meaningless and indicates an unsupported scenario.
  if (source->predecessors.empty() && source->successors.empty()) {
    LDBG("Cannot merge block " << source->blockId
                               << ": no predecessors or successors\n");
    return llvm::failure();
  }

  // Re-direct every edge touching source so that it points to target.
  // DenseSet::insert de-duplicates, so any edge target already owns from
  // before the merge is preserved naturally.
  //
  // Note: the iteration loops read `source->predecessors` / `source->
  // successors` while we mutate them in place. To avoid invalidating the
  // current iterator we drain them into temporary vectors first. Using
  // range insert on the destination keeps this O(N) overall.
  llvm::SmallVector<BlockNode *> sourcePreds(source->predecessors.begin(),
                                             source->predecessors.end());
  llvm::SmallVector<BlockNode *> sourceSuccs(source->successors.begin(),
                                             source->successors.end());

  for (BlockNode *pred : sourcePreds) {
    pred->successors.erase(source);
    pred->successors.insert(target);
  }
  target->predecessors.insert(sourcePreds.begin(), sourcePreds.end());

  for (BlockNode *succ : sourceSuccs) {
    succ->predecessors.erase(source);
    succ->predecessors.insert(target);
  }
  target->successors.insert(sourceSuccs.begin(), sourceSuccs.end());

  // Drop the source node from blockNodes. The BlockNode object itself is
  // destroyed here (unique_ptr is reset); only `source`'s local pointer
  // value is invalidated, which we are done using.
  int sourceId = source->blockId;
  blockNodes.erase(source->blockId);

  // Only the successor chain reachable from target is affected by the
  // merge; recompute depth locally rather than walking the entire graph.
  recomputeDepthsFrom(target);

  LDBG("Rebuilt graph after merging block " << sourceId << " into "
                                            << target->blockId);
  return llvm::success();
}
