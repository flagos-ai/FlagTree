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

#ifndef TRITON_ADAPTER_DYNAMIC_CV_PIPELINE_PLAN_COMPUTE_BLOCK_MERGE_CUBE_BLOCK_PASS_H
#define TRITON_ADAPTER_DYNAMIC_CV_PIPELINE_PLAN_COMPUTE_BLOCK_MERGE_CUBE_BLOCK_PASS_H

#include "DynamicCVPipeline/Common/BlockDependencyGraph.h"
#include "DynamicCVPipeline/Common/MemoryEffectsTracker.h"
#include "DynamicCVPipeline/PlanComputeBlock/ComputeBlockIdManager.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/SmallVector.h"

namespace mlir {
namespace CVPipeline {

class MergeCubeBlockPass
    : public PassWrapper<MergeCubeBlockPass, OperationPass<ModuleOp>> {
public:
  MergeCubeBlockPass() = default;

  StringRef getArgument() const override { return "merge-cube-block"; }
  StringRef getDescription() const override {
    return "Merge cube blocks with same loaded data";
  }

  void runOnOperation() override;

private:
  llvm::LogicalResult processBlock(Block *block,
                                   const MemoryDependenceGraph &memGraph,
                                   ComputeBlockIdManager &bm, bool &mergedAny);

  bool canMergeBlocks(BlockNode *target, BlockNode *source,
                      BlockDependencyGraph &graph,
                      const MemoryDependenceGraph &memGraph,
                      ComputeBlockIdManager &bm);

  // Execute merge
  llvm::LogicalResult mergeBlocks(BlockNode *target, BlockNode *source,
                                  ComputeBlockIdManager &bm);

  // Perform iterative merging by repeatedly scanning the live cube blocks.
  llvm::LogicalResult performMerging(BlockDependencyGraph &graph,
                                     const MemoryDependenceGraph &memGraph,
                                     ComputeBlockIdManager &bm,
                                     bool &mergedAny);

  // Print graph structure
  void printGraph(BlockDependencyGraph &graph);

  bool hasCommonInputOrOutput(BlockNode *node1, BlockNode *node2,
                              BlockDependencyGraph &graph);
  bool hasSameDepth(BlockNode *node1, BlockNode *node2,
                    BlockDependencyGraph &graph);
  bool checkNoCycle(BlockNode *node1, BlockNode *node2,
                    BlockDependencyGraph &graph,
                    const MemoryDependenceGraph &memGraph,
                    ComputeBlockIdManager &bm);

  // Return the subset of `blocks` whose type (cube/vector) differs from
  // `currentNode`'s. Returned as a DenseSet so callers can do O(N) set
  // intersection directly.
  llvm::DenseSet<BlockNode *>
  getBlocksOfDifferentType(BlockNode *currentNode,
                           llvm::ArrayRef<BlockNode *> blocks);
};

} // namespace CVPipeline
} // namespace mlir

namespace mlir {
namespace triton {
std::unique_ptr<OperationPass<ModuleOp>> createMergeCubeBlockPass();
} // namespace triton
} // namespace mlir

#endif // TRITON_ADAPTER_DYNAMIC_CV_PIPELINE_PLAN_COMPUTE_BLOCK_MERGE_CUBE_BLOCK_PASS_H
