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

#include "ComputeBlockOpt/SplitIfByBlockId/Common.h"
#include "DynamicCVPipeline/Common/BlockDependencyGraph.h"
#include "DynamicCVPipeline/Common/MemoryEffectsTracker.h"
#include "DynamicCVPipeline/PlanComputeBlock/ComputeBlockIdManager.h"
#include "ascend/include/DynamicCVPipeline/Common/Utils.h"
#include "ascend/include/DynamicCVPipeline/ComputeBlockOpt/Common.h"
#include "ascend/include/DynamicCVPipeline/ComputeBlockOpt/MergeCubeBlockPass.h"
#include "mlir/Analysis/AliasAnalysis.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/BuiltinOps.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/Support/Debug.h"

using namespace mlir;
using namespace CVPipeline;

static constexpr const char *DEBUG_TYPE = "merge-cube-block";
#define DBGS() (llvm::dbgs() << '[' << DEBUG_TYPE << "] ")
#define LDBG(...) LLVM_DEBUG(DBGS() << __VA_ARGS__ << "\n")

static const llvm::DenseSet<llvm::StringRef> kDisableMergeCubeKernel = {
    "_attn_bwd",
    "kernel_sdpa_bwd_q",
    "_sdpa_infer_kernel",
    "pcb06_tc02_c2v2v2c_chain",
    "flex_attention_backward_dq_kernel",
    "parallel_deltaformer_bwd_kernel_qk",
    "chunk_kda_bwd_kernel_intra",
};

void MergeCubeBlockPass::runOnOperation() {
  auto moduleOp = getOperation();

  if (hasFallbackAttr(moduleOp)) {
    return;
  }

  for (auto funcOp : moduleOp.getOps<func::FuncOp>()) {
    if (kDisableMergeCubeKernel.contains(funcOp.getSymName())) {
      LDBG("Found unsupport kernel: " << funcOp.getSymName());
      moduleOp->setAttr(CVPipeline::kMergeComputeBlockApplied,
                        BoolAttr::get(&getContext(), false));
      return;
    }
  }

  auto &aa = getAnalysis<AliasAnalysis>();
  auto memGraph = MemoryDependenceGraph(moduleOp, aa);
  auto bm = ComputeBlockIdManager(moduleOp);

  LDBG("Starting MergeCubeBlockPass\n" << moduleOp);

  // Collect main loop body blocks via the shared walkMainLoop helper.
  llvm::SmallVector<Block *> mainLoopBlocks;
  if (failed(CVPipeline::SplitIf::walkMainLoop(
          moduleOp, [&](Operation *loop) -> llvm::LogicalResult {
            mainLoopBlocks.push_back(&loop->getRegion(0).front());
            return llvm::success();
          }))) {
    CVPipeline::setFallbackAttr(moduleOp, CVPipeline::ERRCODE_FAILED);
    return;
  }

  LDBG("Found " << mainLoopBlocks.size() << " main loop blocks\n");

  // Process each main loop block
  bool mergedAny = false;
  for (Block *block : mainLoopBlocks) {
    if (failed(processBlock(block, memGraph, bm, mergedAny))) {
      CVPipeline::setFallbackAttr(moduleOp, CVPipeline::ERRCODE_FAILED);
      return;
    }
  }

  if (!mergedAny) {
    moduleOp->setAttr(CVPipeline::kMergeComputeBlockApplied,
                      BoolAttr::get(&getContext(), false));
  }

  LDBG("MergeCubeBlockPass completed\n" << moduleOp);
}

llvm::LogicalResult
MergeCubeBlockPass::processBlock(Block *block,
                                 const MemoryDependenceGraph &memGraph,
                                 ComputeBlockIdManager &bm, bool &mergedAny) {
  LDBG("Processing Block: " << *block);
  if (Operation *parentOp = block->getParentOp()) {
    LDBG("  Parent operation: " << parentOp->getName().getStringRef());
  }

  // Step 1: Build Block dependency graph
  BlockDependencyGraph graph(block, memGraph, bm);
  if (failed(graph.buildGraph())) {
    LDBG("Failed to build dependency graph");
    return llvm::failure();
  }

  LDBG("Built dependency graph with " << graph.blockNodes.size()
                                      << " blocks\n");

  // If there are at least two cube blocks, print the graph for debugging.
  LLVM_DEBUG({
    unsigned cubeCount = 0;
    for (auto &entry : graph.blockNodes)
      if (entry.second->isCube)
        ++cubeCount;
    if (cubeCount >= 2) {
      LDBG("=== Printing dependency graph for current region ===");
      printGraph(graph);
    }
  });

  // Step 2: Perform iterative merging. The nested loop inside performs the
  // candidate selection itself by re-scanning the live cube blocks every
  // round, so there is no separate findMergeCandidates phase.
  if (failed(performMerging(graph, memGraph, bm, mergedAny))) {
    LDBG("Failed to perform merging");
    return llvm::failure();
  }

  return llvm::success();
}

llvm::LogicalResult
MergeCubeBlockPass::performMerging(BlockDependencyGraph &graph,
                                   const MemoryDependenceGraph &memGraph,
                                   ComputeBlockIdManager &bm, bool &mergedAny) {

  // Collect the live cube blocks once. After every merge we simply drop
  // the merged-out source from this list, so we never need to re-scan
  // graph.blockNodes during the merging rounds.
  llvm::SmallVector<BlockNode *> cubeBlocks;
  for (auto &entry : graph.blockNodes) {
    if (entry.second->isCube)
      cubeBlocks.push_back(entry.second.get());
  }

  bool merged = true;
  int mergeCount = 0;

  while (merged) {
    merged = false;

    // Nested scan: outer index plays the role of target, inner index the
    // role of source. As soon as one (target, source) pair is mergeable we
    // commit, drop `source` from the list, and restart from the top of the
    // round.
    for (size_t i = 0; i < cubeBlocks.size() && !merged; ++i) {
      for (size_t j = 0; j < cubeBlocks.size(); ++j) {
        if (i == j)
          continue;
        BlockNode *target = cubeBlocks[i];
        BlockNode *source = cubeBlocks[j];
        if (!canMergeBlocks(target, source, graph, memGraph, bm))
          continue;

        LDBG("Merging block " << source->blockId << " into "
                              << target->blockId);

        // Execute merge
        if (failed(mergeBlocks(target, source, bm))) {
          LDBG("Failed to merge blocks");
          return llvm::failure();
        }
        merged = true;
        mergeCount++;

        // Update graph structure. After this call, the BlockNode pointed to
        // by `source` has been destroyed; the pointer value itself remains
        // usable for equality checks, but any dereference is unsafe. The
        // pointers for `target` and unrelated nodes stay valid because
        // blockNodes.erase() does not rehash or relocate other entries.
        //
        // Note: mergeBlocks above has already re-tagged source's ops to
        // target via the ComputeBlockIdManager, which is not transactional.
        // A rebuildAfterMerge failure therefore leaves the graph and the
        // blockIdManager in an inconsistent state, so we abort the whole
        // merging pass and surface the error to processBlock.
        if (failed(graph.rebuildAfterMerge(target, source))) {
          LDBG("Failed to rebuild graph after merging "
               << source->blockId << " into " << target->blockId);
          return llvm::failure();
        }

        // Drop the merged-out source from the candidate list in place; we
        // already know it lives at index j. The pointer itself is dangling
        // after the rebuild but we only use its value for erase, which
        // shifts elements down to fill the gap.
        cubeBlocks.erase(cubeBlocks.begin() + j);
        break;
      }
    }
  }

  mergedAny = mergeCount > 0;

  LDBG("Merged " << mergeCount << " blocks\n");

  return llvm::success();
}

bool MergeCubeBlockPass::canMergeBlocks(BlockNode *target, BlockNode *source,
                                        BlockDependencyGraph &graph,
                                        const MemoryDependenceGraph &memGraph,
                                        ComputeBlockIdManager &bm) {

  // Precondition: both blocks must be cube
  if (!target || !source || !target->isCube || !source->isCube) {
    return false;
  }

  // Step 1: Check if they have common input or output nodes
  if (!hasCommonInputOrOutput(target, source, graph)) {
    LDBG("Blocks " << target->blockId << " and " << source->blockId
                   << " cannot merge: no common input or output nodes\n");
    return false;
  }

  // Step 2: Check if merging would create a cycle
  if (!checkNoCycle(target, source, graph, memGraph, bm)) {
    LDBG("Blocks " << target->blockId << " and " << source->blockId
                   << " cannot merge: would create cycle\n");
    return false;
  }

  // Step 3: cube blocks at the same depth (with matching successor depths).
  if (hasSameDepth(target, source, graph)) {
    LDBG("Blocks " << target->blockId << " and " << source->blockId
                   << " can merge: cube blocks have same depth\n");
    return true;
  }

  LDBG("Blocks " << target->blockId << " and " << source->blockId
                 << " cannot merge: unsupport scenario\n");
  return false;
}

bool MergeCubeBlockPass::hasSameDepth(BlockNode *node1, BlockNode *node2,
                                      BlockDependencyGraph &graph) {

  if (!node1 || !node2) {
    return false;
  }

  if (node1->depth != node2->depth) {
    return false;
  }

  // Check that the max depth among successors of both blocks is the same
  auto getMaxSuccDepth = [&](BlockNode *node) {
    int maxDepth = -1;
    for (BlockNode *succNode : graph.getSuccessors(node)) {
      if (succNode && succNode->isCube != node->isCube &&
          succNode->depth > maxDepth) {
        maxDepth = succNode->depth;
      }
    }
    return maxDepth;
  };

  int maxSuccDepth1 = getMaxSuccDepth(node1);
  int maxSuccDepth2 = getMaxSuccDepth(node2);

  // If either block has no successors, still mergeable
  if (maxSuccDepth1 == -1 || maxSuccDepth2 == -1) {
    return true;
  }

  return maxSuccDepth1 == maxSuccDepth2;
}

llvm::DenseSet<BlockNode *> MergeCubeBlockPass::getBlocksOfDifferentType(
    BlockNode *currentNode, llvm::ArrayRef<BlockNode *> blocks) {

  llvm::DenseSet<BlockNode *> result;
  if (!currentNode)
    return result;

  // Filter blocks: only keep blocks with a different type than currentNode.
  for (BlockNode *blockNode : blocks) {
    if (blockNode && blockNode->isCube != currentNode->isCube)
      result.insert(blockNode);
  }
  return result;
}

bool MergeCubeBlockPass::checkNoCycle(BlockNode *node1, BlockNode *node2,
                                      BlockDependencyGraph &graph,
                                      const MemoryDependenceGraph &memGraph,
                                      ComputeBlockIdManager &bm) {

  // Get all operations from node2 to be merged into node1
  auto opsToUnify = bm.getOpsByBlockId(node2->blockId);

  // Use willCreateCycle to check if merging would create a cycle
  return !willCreateCycle(opsToUnify, memGraph, node1->blockId, bm);
}

bool MergeCubeBlockPass::hasCommonInputOrOutput(BlockNode *node1,
                                                BlockNode *node2,
                                                BlockDependencyGraph &graph) {

  // Get predecessors and successors, then filter to opposite-type blocks.
  // The filtered sets are already DenseSets, so set intersection is O(N).
  auto filteredPreds1 =
      getBlocksOfDifferentType(node1, graph.getPredecessors(node1));
  auto filteredPreds2 =
      getBlocksOfDifferentType(node2, graph.getPredecessors(node2));
  auto filteredSuccs1 =
      getBlocksOfDifferentType(node1, graph.getSuccessors(node1));
  auto filteredSuccs2 =
      getBlocksOfDifferentType(node2, graph.getSuccessors(node2));

  // Check if there are common input nodes (predecessors).
  bool hasCommonPred = llvm::any_of(
      filteredPreds2, [&](BlockNode *p) { return filteredPreds1.count(p); });

  // Check if there are common output nodes (successors).
  bool hasCommonSucc = llvm::any_of(
      filteredSuccs2, [&](BlockNode *s) { return filteredSuccs1.count(s); });

  // Return true if they have common input OR output nodes.
  return hasCommonPred || hasCommonSucc;
}

llvm::LogicalResult MergeCubeBlockPass::mergeBlocks(BlockNode *target,
                                                    BlockNode *source,
                                                    ComputeBlockIdManager &bm) {

  // Merge all ops of source into target
  auto sourceOps = bm.getOpsByBlockId(source->blockId);
  for (Operation *op : sourceOps) {
    bm.updateBlockId(op, target->blockId);
  }

  return llvm::success();
}

void MergeCubeBlockPass::printGraph(BlockDependencyGraph &graph) {

  LDBG("Total blocks in graph: " << graph.blockNodes.size());

  // Print all block nodes
  for (auto &entry : graph.blockNodes) {
    int blockId = entry.first;
    BlockNode *node = entry.second.get();

    LDBG("Block " << blockId << ":");
    LDBG("  Type: " << (node->isCube ? "CUBE" : "VECTOR"));
    LDBG("  Depth: " << node->depth);
    LDBG("  Num ops: " << node->ops.size());

    // Print predecessors
    auto preds = graph.getPredecessors(node);
    if (!preds.empty()) {
      llvm::SmallVector<std::string> predStrs;
      for (BlockNode *p : preds) {
        predStrs.push_back(std::to_string(p->blockId));
      }
      LDBG("  Predecessors: [" << llvm::join(predStrs, ", ") << "]");
    } else {
      LDBG("  Predecessors: []");
    }

    // Print successors
    auto succs = graph.getSuccessors(node);
    if (!succs.empty()) {
      llvm::SmallVector<std::string> succStrs;
      for (BlockNode *s : succs) {
        succStrs.push_back(std::to_string(s->blockId));
      }
      LDBG("  Successors: [" << llvm::join(succStrs, ", ") << "]");
    } else {
      LDBG("  Successors: []");
    }
  }

  LDBG("=== End of graph ===");
}

std::unique_ptr<OperationPass<ModuleOp>>
mlir::triton::createMergeCubeBlockPass() {
  return std::make_unique<MergeCubeBlockPass>();
}
