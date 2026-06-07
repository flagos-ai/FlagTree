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

#include "ascend/include/DynamicCVPipeline/AddControlFlowCondition/Utils.h"

using namespace mlir;
using namespace llvm;

// Collect all nested ops within an operation's regions
LogicalResult triton::collectAllNestedOps(Operation *op, llvm::DenseSet<Operation *> &regionOps)
{
  if (!op) {
    return failure();
  }

  if (regionOps.contains(op)) {
    return success();
  }

  regionOps.insert(op);
  for (Region &region : op->getRegions()) {
    for (Block &block : region) {
      for (Operation &nestedOp : block) {
        if (failed(triton::collectAllNestedOps(&nestedOp, regionOps))) {
          return failure();
        }
      }
    }
  }

  return success();
}

// Group operations by their block_id attribute
LogicalResult triton::collectOpsByBlockId(scf::ForOp forOp,
                                          llvm::DenseMap<int, SmallVector<Operation *>> &blockOps)
{
  for (Operation &op : forOp.getBody()->without_terminator()) {
    if (auto attr = op.getAttrOfType<IntegerAttr>("ssbuffer.block_id")) {
      blockOps[attr.getInt()].push_back(&op);
    } else {
      return failure();
    }
  }

  return success();
}

// DFS for topological sort - returns failure if cycle detected
static LogicalResult dfsTopologicalSort(
    Operation *op, llvm::DenseSet<Operation *> &visited,
    llvm::DenseSet<Operation *> &inStack,
    const llvm::DenseSet<Operation *> &ops,
    llvm::DenseMap<Operation *, int> *opOrder,
    SmallVectorImpl<Operation *> &sorted)
{
  if (!op) {
    return success();
  }
  if (inStack.contains(op)) {
    return failure();
  }
  if (visited.contains(op)) {
    return success();
  }

  visited.insert(op);
  inStack.insert(op);

  SmallVector<Operation *> deps;
  for (Value operand : op->getOperands()) {
    if (Operation *def = operand.getDefiningOp()) {
      if (ops.contains(def)) {
        deps.push_back(def);
      }
    }
  }

  if (opOrder && !opOrder->empty()) {
    llvm::sort(deps, [&](Operation *a, Operation *b) { return (*opOrder)[a] < (*opOrder)[b]; });
  }

  for (Operation *dep : deps) {
    if (failed(dfsTopologicalSort(dep, visited, inStack, ops, opOrder, sorted))) {
      return failure();
    }
  }

  inStack.erase(op);
  sorted.push_back(op);
  return success();
}

// Topological sort of operations based on operand dependencies
LogicalResult triton::topologicalSort(llvm::DenseSet<Operation *> &ops,
                                      llvm::DenseMap<Operation *, int> *opOrder,
                                      SmallVectorImpl<Operation *> &sorted)
{
  llvm::DenseSet<Operation *> visited;
  llvm::DenseSet<Operation *> inStack;

  SmallVector<Operation *> opList(ops.begin(), ops.end());
  if (opOrder && !opOrder->empty()) {
    llvm::sort(opList, [&](Operation *a, Operation *b) { return (*opOrder)[a] < (*opOrder)[b]; });
  }

  for (Operation *op : opList) {
    if (failed(dfsTopologicalSort(op, visited, inStack, ops, opOrder, sorted))) {
      return failure();
    }
  }
  return success();
}

LogicalResult triton::topologicalSort(SmallVector<Operation *> &ops)
{
  llvm::DenseSet<Operation *> opSet(ops.begin(), ops.end());
  SmallVector<Operation *> sorted;

  if (failed(triton::topologicalSort(opSet, nullptr, sorted))) {
    return failure();
  }
  ops.assign(sorted.begin(), sorted.end());
  return success();
}

// Get block_ids in order of appearance in for loop body
SmallVector<int> triton::getBlockIdsInOrder(scf::ForOp forOp)
{
  SmallVector<int> idsInOrder;
  llvm::DenseSet<int> seenIds;

  for (Operation &op : forOp.getBody()->without_terminator()) {
    if (auto blockIdAttr = op.getAttrOfType<IntegerAttr>("ssbuffer.block_id")) {
      int id = blockIdAttr.getInt();
      if (seenIds.insert(id).second) {
        idsInOrder.push_back(id);
      }
    }
  }
  return idsInOrder;
}
