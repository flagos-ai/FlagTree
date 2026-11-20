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
#include "DynamicCVPipeline/Common/CycleDetector.h"
#include "DynamicCVPipeline/Common/DependencyHelper.h"
#include "DynamicCVPipeline/ComputeBlockOpt/Common.h"
#include "ascend/include/DynamicCVPipeline/Common/MemoryEffectsTracker.h"
#include "ascend/include/DynamicCVPipeline/Common/Utils.h"
#include "ascend/include/DynamicCVPipeline/ComputeBlockOpt/Common.h"
#include "ascend/include/DynamicCVPipeline/ComputeBlockOpt/Passes.h"
#include "ascend/include/DynamicCVPipeline/PlanComputeBlock/Common.h"
#include "ascend/include/DynamicCVPipeline/PlanComputeBlock/ComputeBlockIdManager.h"
#include "mlir/Analysis/AliasAnalysis.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/Block.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Operation.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/LogicalResult.h"
#include "llvm/Support/raw_ostream.h"
#include <algorithm>
#include <cstdint>
#include <optional>
#include <queue>

#define DEBUG_TYPE "ub-usage-opt"
#define LOG_DEBUG(msg)                                                         \
  LLVM_DEBUG(llvm::dbgs() << " [" << DEBUG_TYPE << "] " << msg)

using namespace mlir;
using namespace triton;
using namespace mlir::triton;

namespace mlir {
namespace triton {
class UBUsageOptPass
    : public PassWrapper<UBUsageOptPass, OperationPass<ModuleOp>> {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(UBUsageOptPass)

  UBUsageOptPass() = default;
  void runOnOperation() override;

  llvm::StringRef getArgument() const final { return "ub-usage-opt"; }

private:
  const int64_t MAX_EDGE_SIZE = (1 << 30);
  int64_t getValueSizeInBytes(Value value);
  void buildUBUsageGraph(Block *block, DenseMap<Operation *, int> &op2nodeId,
                         DenseMap<int, Operation *> &nodeId2op,
                         SmallVector<SmallVector<int>> &linkOut,
                         SmallVector<SmallVector<int>> &linkIn,
                         SmallVector<int64_t> &linkSize,
                         SmallVector<int> &linkStart, SmallVector<int> &linkEnd,
                         SmallVector<int> &nodeBlockId,
                         SmallVector<int> &nodeCoreType,
                         SmallVector<int> &nodeArgs,
                         const CVPipeline::MemoryDependenceGraph &memGraph,
                         CVPipeline::ComputeBlockIdManager &bm);
  DenseMap<int, int> collectRecordChange(
      const SmallVector<SmallVector<int>> &needUbOpts,
      const SmallVector<SmallVector<int>> &linkOut,
      const SmallVector<SmallVector<int>> &linkIn,
      const SmallVector<int64_t> &linkSize, const SmallVector<int> &linkStart,
      const SmallVector<int> &linkEnd, const SmallVector<int> &nodeBlockId,
      const SmallVector<int> &nodeCoreType,
      DenseMap<int, Operation *> nodeId2op);
  int64_t sumIncomingLinkSize(int nodeId,
                              const SmallVector<SmallVector<int>> &linkIn,
                              const SmallVector<int64_t> &linkSize,
                              const SmallVector<int> &linkStart,
                              const SmallVector<int> &linkEnd,
                              const SmallVector<int> &nodeBlockId);
  llvm::LogicalResult
  UBUsageOptimization(Block *block,
                      const CVPipeline::MemoryDependenceGraph &memGraph,
                      CVPipeline::ComputeBlockIdManager &bm);
};
} // namespace triton
} // namespace mlir

/**
    The UB occupied by the value is used as the edge weight.
    The compute block will be partitioned at the location with the smallest edge
   weight. Where:
    1. Tensor/Vector: Uses the data size directly.
    2. Memref: Set to the maximum value, indicating that a compute block should
   not be partitioned here.
    3. Isndex/Other: Set to 0, indicating that it does not occupy any UB.
*/
int64_t UBUsageOptPass::getValueSizeInBytes(Value value) {
  Type type = value.getType();
  auto getElemBytes = [](Type elemType) -> int64_t {
    if (elemType.isIntOrFloat()) {
      unsigned bits = elemType.getIntOrFloatBitWidth();
      return std::max<int64_t>(1, bits / 8);
    }
    return 1;
  };
  // Tensor
  if (auto rankedTensorType = dyn_cast<RankedTensorType>(type)) {
    if (!rankedTensorType.hasStaticShape()) {
      return MAX_EDGE_SIZE;
    }
    int64_t numElements = 1;
    for (int64_t dim : rankedTensorType.getShape()) {
      if (dim < 0) {
        return 1;
      }
      numElements *= dim;
    }
    return std::max<int64_t>(
        1, numElements * getElemBytes(rankedTensorType.getElementType()));
  }
  // Memref
  if (auto memRefType = dyn_cast<MemRefType>(type)) {
    return MAX_EDGE_SIZE;
  }
  // Vector
  if (auto vectorType = dyn_cast<VectorType>(type)) {
    return std::max<int64_t>(1, vectorType.getNumElements() *
                                    getElemBytes(vectorType.getElementType()));
  }
  // Index
  if (auto idxTy = dyn_cast<IndexType>(value.getType())) {
    return 0;
  }
  return 0;
}

void UBUsageOptPass::buildUBUsageGraph(
    Block *block, DenseMap<Operation *, int> &op2nodeId,
    DenseMap<int, Operation *> &nodeId2op,
    SmallVector<SmallVector<int>> &linkOut,
    SmallVector<SmallVector<int>> &linkIn, SmallVector<int64_t> &linkSize,
    SmallVector<int> &linkStart, SmallVector<int> &linkEnd,
    SmallVector<int> &nodeBlockId, SmallVector<int> &nodeCoreType,
    SmallVector<int> &nodeArgs,
    const CVPipeline::MemoryDependenceGraph &memGraph,
    CVPipeline::ComputeBlockIdManager &bm) {
  DenseMap<int, int> cubeBlockId2nodeId;
  const int cubeCoreType = static_cast<int>(CVPipeline::CoreType::CUBE_ONLY);

  auto getOrCreateNodeId = [&](Operation *op) -> int {
    if (op2nodeId.find(op) != op2nodeId.end()) {
      return op2nodeId.at(op);
    }
    int coreType = static_cast<int>(CVPipeline::getOpCoreType(op));
    int blockId = bm.getBlockIdByOp(op);
    bool canShrink = (coreType == cubeCoreType && blockId != -1);

    if (canShrink) {
      auto it = cubeBlockId2nodeId.find(blockId);
      if (it != cubeBlockId2nodeId.end()) {
        op2nodeId[op] = it->second;
        return it->second;
      }
    }

    int nodeId = static_cast<int>(nodeBlockId.size());
    op2nodeId[op] = nodeId;
    nodeId2op[nodeId] = op;
    nodeBlockId.push_back(blockId);
    nodeCoreType.push_back(coreType);
    nodeArgs.push_back(-1);
    linkOut.emplace_back();
    linkIn.emplace_back();

    if (canShrink) {
      cubeBlockId2nodeId[blockId] = nodeId;
    }
    return nodeId;
  };

  DenseMap<std::pair<int, int>, bool> visited;
  auto addEdge = [&](int src, int dst, int64_t sizeInBytes) {
    if (visited.contains(std::make_pair(src, dst))) {
      return;
    }
    if (src == dst) {
      // self-cycle can only be args dependency.
      LOG_DEBUG("Find one self-cycle: id edge from "
                << src << " to " << dst << " size = " << sizeInBytes << "\n");
      LOG_DEBUG("op edge from " << *nodeId2op[src] << " to " << *nodeId2op[dst]
                                << "\n");
    }
    int edgeId = linkSize.size();
    linkSize.push_back(sizeInBytes);
    linkStart.push_back(src);
    linkEnd.push_back(dst);
    linkOut[src].push_back(edgeId);
    linkIn[dst].push_back(edgeId);
    visited[std::make_pair(src, dst)] = true;
  };

  Operation *terminator = block->getTerminator();
  if (terminator && isa<scf::YieldOp>(terminator)) {
    unsigned maxArgIdx = std::min<unsigned>(block->getNumArguments(),
                                            terminator->getNumOperands());
    for (unsigned argIdx = 0; argIdx < maxArgIdx; ++argIdx) {
      Value yielded = terminator->getOperand(argIdx);
      Operation *defOp = yielded.getDefiningOp();
      Operation *defInBlock = CVPipeline::getAncestorInBlock(defOp, block);
      if (!defInBlock || defInBlock->getBlock() != block) {
        continue;
      }
      int nodeId = getOrCreateNodeId(defInBlock);
      if (nodeArgs[nodeId] == -1) {
        nodeArgs[nodeId] = static_cast<int>(argIdx);
      }
    }
  }

  for (Operation &blockOp : *block) {
    int dstNode = getOrCreateNodeId(&blockOp);
    blockOp.walk([&](Operation *op) {
      for (Value operand : op->getOperands()) {
        Operation *srcInBlock = nullptr;
        bool fromArgEdge = false;
        if (Operation *defOp = operand.getDefiningOp()) {
          srcInBlock = CVPipeline::getAncestorInBlock(defOp, block);
          if (srcInBlock && srcInBlock == &blockOp) {
            // To avoid unnesessary self-cycle.
            // some ops in block(op) used parent(blockop) args can lead one
            // self-cycle.
            continue;
          }
        } else if (auto blockArg = dyn_cast<BlockArgument>(operand)) {
          if (blockArg.getOwner() == block && terminator &&
              isa<scf::YieldOp>(terminator)) {
            unsigned numArgs = block->getNumArguments();
            unsigned numYieldOperands = terminator->getNumOperands();
            // for op offset=1, while op offset=0
            int offset = (int)numArgs - (int)numYieldOperands;
            int argIdx = (int)blockArg.getArgNumber() - offset;

            if (argIdx >= 0 && argIdx < (int)numYieldOperands) {
              Value yielded = terminator->getOperand(argIdx);
              if (Operation *yieldDefOp = yielded.getDefiningOp()) {
                srcInBlock = CVPipeline::getAncestorInBlock(yieldDefOp, block);
                fromArgEdge = true;
              }
            }
          }
        }
        if (!srcInBlock || srcInBlock->getBlock() != block) {
          continue;
        }
        // To avoid unnesessary self-cycle.
        // Two ops in the same cube block lead to self-cycle, so continue it.
        int coreType = static_cast<int>(CVPipeline::getOpCoreType(op));
        int srcBlockId = bm.getBlockIdByOp(srcInBlock);
        int dstBlockId = bm.getBlockIdByOp(&blockOp);
        if (coreType == cubeCoreType && srcBlockId == dstBlockId) {
          continue;
        }

        int srcNode = getOrCreateNodeId(srcInBlock);
        int64_t edgeSize = getValueSizeInBytes(operand);
        if (fromArgEdge) {
          edgeSize *= 2;
        }
        if (!visited.contains(std::make_pair(srcNode, dstNode))) {
          LOG_DEBUG("Add SSA edage from " << *srcInBlock << " to " << blockOp
                                          << "\nweight = " << edgeSize << "\n");
        }
        addEdge(srcNode, dstNode, edgeSize);
      }

      for (auto memDef : memGraph.getExecBefore(op)) {
        Operation *srcInBlock = nullptr;
        srcInBlock = CVPipeline::getAncestorInBlock(memDef, block);
        if (!srcInBlock || srcInBlock->getBlock() != block) {
          continue;
        }
        if (srcInBlock && srcInBlock == &blockOp) {
          // To avoid unnesessary self-cycle.
          // some ops in block(op) used parent(blockop) args can lead one
          // self-cycle.
          continue;
        }
        // To avoid unnesessary self-cycle.
        // Two ops in the same cube block lead to self-cycle, so continue it.
        int coreType = static_cast<int>(CVPipeline::getOpCoreType(op));
        int srcBlockId = bm.getBlockIdByOp(srcInBlock);
        int dstBlockId = bm.getBlockIdByOp(&blockOp);
        if (coreType == cubeCoreType && srcBlockId == dstBlockId) {
          continue;
        }

        int srcNode = getOrCreateNodeId(srcInBlock);
        if (!visited.contains(std::make_pair(srcNode, dstNode))) {
          LOG_DEBUG("Add memory edage from " << *srcInBlock << " to " << blockOp
                                             << "\nweight = " << 0 << "\n");
        }
        addEdge(srcNode, dstNode, 0);
      }
    });
  }
}

static SmallVector<int>
findDependency(int targetNdoe, int preNode,
               const SmallVector<SmallVector<int>> &linkIn,
               const SmallVector<int> &linkStart) {
  SmallVector<int> dependNodes;
  DenseSet<int> visited;
  std::queue<int> queue;
  queue.push(targetNdoe);
  visited.insert(targetNdoe);
  while (!queue.empty()) {
    int curNode = queue.front();
    queue.pop();
    if (curNode == preNode) {
      continue; // we find other linkin of b in A->B, so pass A.
    }

    for (int inEdgeId : linkIn[curNode]) {
      int inStart = linkStart[inEdgeId];
      if (visited.insert(inStart).second) {
        dependNodes.push_back(inStart);
        queue.push(inStart);
      }
    }
  }
  return dependNodes;
}

static bool isActiveEndNode(int srcNode, int endNode,
                            const SmallVector<SmallVector<int>> &linkIn,
                            const SmallVector<int> &linkStart,
                            const SmallVector<int> &nodeBlockId,
                            const SmallVector<int> &nodeCoreType,
                            DenseMap<int, Operation *> nodeId2op) {
  int nodeNum = static_cast<int>(nodeBlockId.size());
  if (nodeCoreType[endNode] != nodeCoreType[srcNode]) {
    return false;
  }
  if (nodeBlockId[endNode] == -1) {
    // meet scf.yield/return..
    return false;
  }
  if (nodeBlockId[srcNode] == nodeBlockId[endNode]) {
    // same cmpute block not leaf
    return false;
  }
  // To avoid cycles: we only collect 2 kind node:
  // 1. endNode only from src compute block. A->endnode->...
  // 2. endNode's srcs donnot depend any compute block.
  // A->endnode;args/const/->src1->endnode->... In fact the second kind includes
  // first; BFS/DFS search expanding dependNodes transitively
  auto dependNodes = findDependency(endNode, srcNode, linkIn, linkStart);
  for (int node : dependNodes) {
    if (nodeBlockId[node] != nodeBlockId[endNode] &&
        nodeBlockId[node] != nodeBlockId[srcNode]) {
      if (linkIn[node].size() != 0) {
        return false;
      }
    }
  }
  return true;
}

static SmallVector<SmallVector<int>> collectNeedUbOpts(
    const SmallVector<SmallVector<int>> &linkOut,
    const SmallVector<SmallVector<int>> &linkIn,
    const SmallVector<int> &linkStart, const SmallVector<int> &linkEnd,
    const SmallVector<int> &nodeBlockId, const SmallVector<int> &nodeCoreType,
    DenseMap<int, Operation *> nodeId2op) {
  SmallVector<SmallVector<int>> needUbOpts;
  int maxBlockId = -1; // FIXME: We need to count how many compute block in the
                       // graph, but use max is not accurate.
  for (int blockId : nodeBlockId) {
    maxBlockId = std::max(maxBlockId, blockId);
  }
  if (maxBlockId >= 0) {
    needUbOpts.resize(static_cast<size_t>(maxBlockId + 1));
  }
  for (int i = 0; i < nodeBlockId.size(); ++i) {
    int srcBlockId = nodeBlockId[i];
    int srcCoreType = nodeCoreType[i];
    if (srcCoreType != CVPipeline::CoreType::VECTOR_ONLY) {
      // only vector block need ub optimization.
      continue;
    }
    bool canOptimize = false;
    for (int outEdgeId : linkOut[i]) {
      int dstNode = linkEnd[outEdgeId];
      if (isActiveEndNode(i, dstNode, linkIn, linkStart, nodeBlockId,
                          nodeCoreType, nodeId2op)) {
        canOptimize = true;
        break;
      }
    }
    if (canOptimize && srcBlockId >= 0) {
      needUbOpts[srcBlockId].push_back(i);
    }
  }
  return needUbOpts;
}

int64_t UBUsageOptPass::sumIncomingLinkSize(
    int nodeId, const SmallVector<SmallVector<int>> &linkIn,
    const SmallVector<int64_t> &linkSize, const SmallVector<int> &linkStart,
    const SmallVector<int> &linkEnd, const SmallVector<int> &nodeBlockId) {
  int64_t totalSize = 0;
  for (int edgeId : linkIn[nodeId]) {
    if (nodeBlockId[linkStart[edgeId]] != nodeBlockId[linkEnd[edgeId]]) {
      if (linkSize[edgeId] == MAX_EDGE_SIZE) {
        totalSize = MAX_EDGE_SIZE;
      } else {
        totalSize += linkSize[edgeId];
      }
    }
  }
  return totalSize;
}

static bool findUniqueDependentNode(
    int curNode, int optBlockId, const SmallVector<SmallVector<int>> &linkOut,
    const SmallVector<SmallVector<int>> &linkIn,
    const SmallVector<int> &linkStart, const SmallVector<int> &linkEnd,
    const SmallVector<int> &nodeBlockId, int &uniqueNextNode) {
  // only find A link single chain.
  if (linkOut[curNode].size() != 1)
    return false;
  // sigle chian must be in one compute block.
  auto edgeId = linkOut[curNode][0];
  uniqueNextNode = linkEnd[edgeId];
  if (nodeBlockId[uniqueNextNode] != nodeBlockId[curNode]) {
    return false;
  }
  // uniqueNextNode's dependent is all in nodeBlockId[curNode]
  bool onlyDependsOnCur = true;
  auto dependNodes = findDependency(uniqueNextNode, curNode, linkIn, linkStart);
  for (auto node : dependNodes) {
    if (nodeBlockId[node] != nodeBlockId[curNode]) {
      return false;
    }
  }
  return true;
}

DenseMap<int, int> UBUsageOptPass::collectRecordChange(
    const SmallVector<SmallVector<int>> &needUbOpts,
    const SmallVector<SmallVector<int>> &linkOut,
    const SmallVector<SmallVector<int>> &linkIn,
    const SmallVector<int64_t> &linkSize, const SmallVector<int> &linkStart,
    const SmallVector<int> &linkEnd, const SmallVector<int> &nodeBlockId,
    const SmallVector<int> &nodeCoreType,
    DenseMap<int, Operation *> nodeId2op) {
  DenseMap<int, int> recordChange;
  int nodeNum = static_cast<int>(nodeBlockId.size());
  for (int optBlockId = 0, blockNum = static_cast<int>(needUbOpts.size());
       optBlockId < blockNum; ++optBlockId) {
    for (int optNode : needUbOpts[optBlockId]) {
      LOG_DEBUG("optNode: " << *nodeId2op.at(optNode) << "\n");
      SmallVector<int> activateSet;
      for (int outEdgeId : linkOut[optNode]) {
        int dstNode = linkEnd[outEdgeId];
        if (isActiveEndNode(optNode, dstNode, linkIn, linkStart, nodeBlockId,
                            nodeCoreType, nodeId2op)) {
          if (std::find(activateSet.begin(), activateSet.end(), dstNode) ==
              activateSet.end()) {
            activateSet.push_back(dstNode);
          }
        }
      }
      LOG_DEBUG("activateSet Size:" << activateSet.size() << "\n");

      for (int activateNode : activateSet) {
        int64_t originUBSize = sumIncomingLinkSize(
            activateNode, linkIn, linkSize, linkStart, linkEnd, nodeBlockId);
        LOG_DEBUG("activateNode:" << *nodeId2op.at(activateNode) << "\n");
        LOG_DEBUG("originUBSize:" << originUBSize << "\n");
        int64_t minUBSize = originUBSize;
        SmallVector<int> chain;
        chain.push_back(activateNode);
        int bestCutPointIdx = -1;

        // find the activate chain.
        while (true) {
          int curNode = chain.back();
          int uniqueNextNode = -1;
          if (!findUniqueDependentNode(curNode, optBlockId, linkOut, linkIn,
                                       linkStart, linkEnd, nodeBlockId,
                                       uniqueNextNode)) {
            break;
          }

          chain.push_back(uniqueNextNode);
        }
        for (auto i = 0; i < chain.size(); i++) {
          auto nowUBSize = 0LL;
          LOG_DEBUG("now chain op = " << *nodeId2op.at(chain[i]) << "\n");
          for (auto cutEdgeId : linkOut[chain[i]]) {
            if (linkSize[cutEdgeId] == MAX_EDGE_SIZE) {
              nowUBSize = MAX_EDGE_SIZE;
              break;
            } else {
              nowUBSize += linkSize[cutEdgeId];
            }
          }
          if (nowUBSize < minUBSize) {
            bestCutPointIdx = i + 1;
            minUBSize = nowUBSize;
          }
        }

        if (bestCutPointIdx > 0) {
          // need to change not only chain[i], and chain[i]'s dependency.
          for (int i = 0; i < bestCutPointIdx; ++i) {
            recordChange[chain[i]] = optBlockId;
            auto chainPreNode = i - 1 < 0 ? optNode : chain[i - 1];
            auto dependNodes =
                findDependency(chain[i], chainPreNode, linkIn, linkStart);
            for (auto node : dependNodes) {
              if (nodeBlockId[node] != optBlockId) {
                recordChange[node] = optBlockId;
              }
            }
          }
        }
      }
    }
  }
  return recordChange;
}

bool applyRecordChange(DenseMap<int, int> &recordChange,
                       DenseMap<int, Operation *> &nodeId2op,
                       const CVPipeline::MemoryDependenceGraph &memGraph,
                       CVPipeline::ComputeBlockIdManager &bm) {
  // Get ever blockid should be add which nodeId in recordChange.
  DenseMap<int, SmallVector<int>> blockWilladd;
  for (const auto &it : recordChange) {
    int nodeId = it.first;
    int optBlockId = it.second;
    blockWilladd[optBlockId].push_back(nodeId);
  }
  bool hasError = false;
  for (auto it : blockWilladd) {
    int targetBlockId = it.first;
    auto willaddNodes = it.second;
    llvm::SmallVector<Operation *> willaddOps;
    for (int nodeId : willaddNodes) {
      willaddOps.push_back(nodeId2op[nodeId]);
    }
    if (CVPipeline::willCreateCycle(willaddOps, memGraph, targetBlockId, bm)) {
      LOG_DEBUG("Find cycle when apply change for blockId: " << targetBlockId
                                                             << "\n");
      for (auto nodeId : willaddNodes) {
        LOG_DEBUG("  - " << *nodeId2op[nodeId] << "\n");
      }
      hasError = true;
      continue;
    }

    for (auto op : willaddOps) {
      bm.updateBlockIdWithInner(op, targetBlockId);
    }
  }

  return hasError;
}

llvm::LogicalResult UBUsageOptPass::UBUsageOptimization(
    Block *block, const CVPipeline::MemoryDependenceGraph &memGraph,
    CVPipeline::ComputeBlockIdManager &bm) {
  if (!(isa<scf::ForOp>(block->getParentOp()) ||
        isa<scf::WhileOp>(block->getParentOp()))) {
    return llvm::success();
  }
  DenseMap<Operation *, int> op2nodeId;
  DenseMap<int, Operation *> nodeId2op;
  SmallVector<SmallVector<int>> linkOut;
  SmallVector<SmallVector<int>> linkIn;
  SmallVector<int64_t> linkSize;
  SmallVector<int> linkStart;
  SmallVector<int> linkEnd;
  SmallVector<int> nodeBlockId;
  SmallVector<int> nodeCoreType;
  SmallVector<int> nodeArgs;
  buildUBUsageGraph(block, op2nodeId, nodeId2op, linkOut, linkIn, linkSize,
                    linkStart, linkEnd, nodeBlockId, nodeCoreType, nodeArgs,
                    memGraph, bm);
  SmallVector<SmallVector<int>> needUbOpts =
      collectNeedUbOpts(linkOut, linkIn, linkStart, linkEnd, nodeBlockId,
                        nodeCoreType, nodeId2op);
  int candidateCnt = 0;
  for (const auto &nodes : needUbOpts) {
    candidateCnt += static_cast<int>(nodes.size());
    for (auto opid : nodes) {
      LOG_DEBUG("maybe need opt: " << *nodeId2op[opid] << "\t id = " << opid
                                   << "\n");
    }
  }
  LOG_DEBUG("Find " << candidateCnt << " op maybe need UB optimization\n");
  llvm::DenseMap<int, int> recordChange =
      collectRecordChange(needUbOpts, linkOut, linkIn, linkSize, linkStart,
                          linkEnd, nodeBlockId, nodeCoreType, nodeId2op);
  LOG_DEBUG("Need change blockId for " << recordChange.size() << " nodes\n");
  for (auto rec : recordChange) {
    auto node = nodeId2op[rec.first];
    LOG_DEBUG("Change " << *node << " TO " << rec.second << "\n");
  }

  if (applyRecordChange(recordChange, nodeId2op, memGraph, bm)) {
    LOG_DEBUG("Some skiped when apply UB usage optimization changes.\n");
  }
  return llvm::success();
}

void mlir::triton::UBUsageOptPass::runOnOperation() {
  LOG_DEBUG("--- Pass: UBUsageOpt ---\n");

  ModuleOp module = getOperation();

  if (CVPipeline::hasFallbackAttr(module)) {
    return;
  }

  auto &aliasAnalysis = getAnalysis<AliasAnalysis>();
  CVPipeline::MemoryDependenceGraph memDepGraph(module, aliasAnalysis);
  auto bm = CVPipeline::ComputeBlockIdManager(module);

  llvm::SmallVector<Block *> blocks;
  module.walk([&](Block *block) { blocks.push_back(block); });

  for (Block *block : blocks) {
    if (UBUsageOptimization(block, memDepGraph, bm).failed()) {
      LOG_DEBUG("UB usage optimization failed in block.\n");
    }
  }

  LOG_DEBUG("=== Pass UBUsageOpt complete ===\n");
}

namespace mlir {
namespace triton {

std::unique_ptr<OperationPass<ModuleOp>> createUBUsageOptPass() {
  return std::make_unique<UBUsageOptPass>();
}

} // namespace triton
} // namespace mlir
