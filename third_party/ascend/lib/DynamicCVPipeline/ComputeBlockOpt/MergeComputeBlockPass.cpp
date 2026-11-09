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
#include "ascend/include/DynamicCVPipeline/Common/BufferCountManager.h"
#include "ascend/include/DynamicCVPipeline/Common/DependencyHelper.h"
#include "ascend/include/DynamicCVPipeline/ComputeBlockOpt/Common.h"
#include "ascend/include/DynamicCVPipeline/ComputeBlockOpt/Passes.h"
#include "ascend/include/DynamicCVPipeline/PlanComputeBlock/Common.h"
#include "ascend/include/DynamicCVPipeline/PlanComputeBlock/ComputeBlockIdManager.h"

#include "DynamicCVPipeline/Common/Utils.h"

#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/Interfaces/ViewLikeInterface.h"

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Debug.h"

#include <optional>

static constexpr const char *DEBUG_TYPE = "merge-compute-block";
#define LOG_DEBUG(...)                                                         \
  LLVM_DEBUG(llvm::dbgs() << " [" << DEBUG_TYPE << "] " << __VA_ARGS__ << "\n")

static constexpr llvm::StringLiteral kEnableMergeComputeBlockKernels[] = {
    "flex_attention_backward_dkdv_kernel",
    "flex_attention_backward_dkdv_kernel_tasklist", "_swa_bwd_dkdv_kernel",
    "kernel_sdpa_bwd_kv", "kernel_da_bwd_kv_ul"};

using namespace mlir;
using namespace triton;

/// Represents a ComputeBlock: a group of ops sharing the same block_id
struct ComputeBlock {
  int id;                        // block_id value
  CVPipeline::CoreType coreType; // CUBE_ONLY / VECTOR_ONLY
  SmallVector<Operation *> ops;  // all ops in the group (in IR order)
};

static bool hasTensorResult(const ComputeBlock &blk) {
  for (Operation *op : blk.ops) {
    for (Value result : op->getResults()) {
      if (isa<TensorType>(result.getType())) {
        return true;
      }
    }
  }
  return false;
}

/// Step 1: Group ops and build block-level dependency graph (enhanced).
/// computeBlocks: output, block_id → ComputeBlock
/// succs/preds: output, block_id → successor/predecessor block_id list
/// blockEdges: output, srcId → dstId → source ops (cross-block SSA/memory
/// dependency sources)
static void groupAndBuildGraph(
    Block *block, const CVPipeline::MemoryDependenceGraph &memGraph,
    DenseMap<int, ComputeBlock> &computeBlocks,
    DenseMap<int, SmallVector<int>> &succs,
    DenseMap<int, SmallVector<int>> &preds,
    DenseMap<int, DenseMap<int, SmallVector<Operation *>>> &blockEdges) {
  // Walk all ops in the body block and its nested regions, group by
  // block_id
  block->walk([&](Operation *op) {
    if (op->hasTrait<OpTrait::IsTerminator>()) {
      return;
    }
    auto optId = CVPipeline::getOpBlockId(op);
    if (!optId.has_value()) {
      return;
    }
    int bid = *optId;
    if (!computeBlocks.contains(bid)) {
      // find new block_id, create new ComputeBlock
      computeBlocks[bid] = {bid, CVPipeline::getOpCoreType(op), {}};
    }
    computeBlocks[bid].ops.push_back(op);
  });

  if (computeBlocks.empty()) {
    return;
  }

  // Build dependency graph between ComputeBlocks via DependencyHelper,
  // covering both SSA operands and memory execution dependencies.
  CVPipeline::DependencyHelper helper(memGraph);
  DenseSet<std::pair<int, int>> seenEdges;
  DenseSet<std::tuple<int, int, Operation *>> seenDeps;
  for (auto &kv : computeBlocks) {
    int curId = kv.first;
    for (Operation *op : kv.second.ops) {
      helper.forEachSource(op, [&](Operation *src) {
        Operation *ancestor = CVPipeline::getAncestorInBlock(src, block);
        if (!ancestor) {
          return WalkResult::advance(); // not in this block
        }
        auto ancIdOpt = CVPipeline::getOpBlockId(ancestor);
        if (!ancIdOpt.has_value()) {
          return WalkResult::advance();
        }
        int ancId = *ancIdOpt;
        if (ancId == curId) {
          return WalkResult::advance(); // same ComputeBlock internal edge
        }

        // Record the source op that triggers this cross-block edge
        // (dedup: outer op and its nested subOps report the same source)
        if (seenDeps.insert({ancId, curId, src}).second) {
          blockEdges[ancId][curId].push_back(src);
        }

        if (!seenEdges.insert({ancId, curId}).second) {
          return WalkResult::advance();
        }
        succs[ancId].push_back(curId);
        preds[curId].push_back(ancId);
        return WalkResult::advance();
      });
    }
  }
}

/// Find the first CUBE predecessor of a given block.
static std::optional<int>
findCubePred(int blockId, const DenseMap<int, ComputeBlock> &computeBlocks,
             const DenseMap<int, SmallVector<int>> &preds) {
  auto it = preds.find(blockId);
  if (it == preds.end()) {
    return std::nullopt;
  }
  for (int p : it->second) {
    if (computeBlocks.lookup(p).coreType == CVPipeline::CoreType::CUBE_ONLY) {
      return p;
    }
  }
  return std::nullopt;
}

/// Collect all transitive dependencies of \p startOp via SSA use-def chains,
/// memory execution predecessors, and ops nested in regions. Results are
/// accumulated into \p collected (including \p startOp itself).
static void collectAllDeps(Operation *startOp,
                           const CVPipeline::MemoryDependenceGraph &memGraph,
                           SmallPtrSetImpl<Operation *> &collected) {
  collected.insert(startOp);
  CVPipeline::DependencyHelper helper(memGraph);
  SmallVector<Operation *> worklist = {startOp};
  while (!worklist.empty()) {
    Operation *cur = worklist.pop_back_val();
    helper.forEachSource(cur, [&](Operation *src) {
      if (collected.insert(src).second)
        worklist.push_back(src);
      return WalkResult::advance();
    });
  }
}

/// Check if Cube depends on CubePre via bufferization::ToTensorOp defined in
/// CubePre. \p edgeSrcOps are the dependency source ops on the CubePre → Cube
/// edges. Returns the to_tensor ops in CubePre that Cube depends on.
static SmallVector<Operation *>
findToTensorDeps(const ComputeBlock *cubePre,
                 ArrayRef<Operation *> edgeSrcOps) {
  SmallVector<Operation *> result;
  if (!cubePre) {
    return result;
  }
  DenseSet<Operation *> cubePreOpsSet(cubePre->ops.begin(), cubePre->ops.end());

  for (Operation *srcOp : edgeSrcOps) {
    if (isa<bufferization::ToTensorOp>(srcOp) &&
        cubePreOpsSet.contains(srcOp)) {
      result.push_back(srcOp);
    }
  }
  return result;
}

/// Clone ops from CubePre into Cube at the front of Cube's ops.
/// Selected ops must be in CubePre and are in `toClone`.
static void
cloneOpCrossCubeDep(int cubePreId, int cubeId,
                    const SmallPtrSet<Operation *, 16> &toClone,
                    const DenseMap<int, ComputeBlock> &computeBlocks,
                    CVPipeline::ComputeBlockIdManager &bm) {
  const auto &cubePreBlock = computeBlocks.at(cubePreId);
  const auto &cubeBlock = computeBlocks.at(cubeId);

  if (cubeBlock.ops.empty()) {
    return;
  }

  Operation *insertBefore = cubeBlock.ops.front();
  OpBuilder builder(insertBefore);
  IRMapping mapper;
  for (Operation *op : cubePreBlock.ops) {
    if (!toClone.contains(op)) {
      continue;
    }
    Operation *cloned = builder.clone(*op, mapper);
    cloned->walk(
        [&](Operation *innerOp) { bm.updateBlockId(innerOp, cubeId); });
  }

  // Remap Cube's original ops' operands: replace references to old CubePre
  // values with the corresponding cloned values now in Cube.
  for (Operation *op : cubeBlock.ops) {
    if (toClone.contains(op)) {
      continue;
    }
    for (auto &operand : op->getOpOperands()) {
      if (Value mapped = mapper.lookupOrNull(operand.get())) {
        operand.set(mapped);
      }
    }
  }
}

/// Step 2: Collect VECTOR_ONLY candidates that have tensor results
/// and have at least one edge (incoming or outgoing) to a CUBE block.
static SmallVector<int>
collectVectorCandidates(const DenseMap<int, ComputeBlock> &computeBlocks,
                        const DenseMap<int, SmallVector<int>> &succs,
                        const DenseMap<int, SmallVector<int>> &preds) {
  auto hasCubeNeighbor = [&](const DenseMap<int, SmallVector<int>> &neighbors,
                             int blockId) {
    auto it = neighbors.find(blockId);
    return it != neighbors.end() && llvm::any_of(it->second, [&](int id) {
             return computeBlocks.lookup(id).coreType ==
                    CVPipeline::CoreType::CUBE_ONLY;
           });
  };

  SmallVector<int> vecCandidates;
  for (auto &kv : computeBlocks) {
    if (kv.second.coreType != CVPipeline::CoreType::VECTOR_ONLY)
      continue;
    if (!hasTensorResult(kv.second))
      continue;

    // Must have a successor or predecessor that is CUBE
    bool hasCubeEdge =
        hasCubeNeighbor(succs, kv.first) || hasCubeNeighbor(preds, kv.first);
    if (!hasCubeEdge)
      continue;

    if (!llvm::is_contained(vecCandidates, kv.first))
      vecCandidates.push_back(kv.first);
  }
  return vecCandidates;
}

/// Step 3: Find a pair of adjacent VECTOR blocks (predV → succV).
/// Returns {predVId, succVId} if a pair is found, otherwise nullopt.
static std::optional<std::pair<int, int>>
findAdjacentVectorPair(ArrayRef<int> vecCandidates,
                       const DenseMap<int, SmallVector<int>> &succs) {
  for (int candId : vecCandidates) {
    auto it = succs.find(candId);
    if (it == succs.end())
      continue;
    for (int succId : it->second) {
      if (llvm::is_contained(vecCandidates, succId)) {
        return std::make_pair(candId, succId);
      }
    }
  }
  return std::nullopt;
}

static void markSubBlock(const DenseMap<int, ComputeBlock> &computeBlocks,
                         int predVId, int succVId) {
  for (int id : {predVId, succVId}) {
    auto it = computeBlocks.find(id);
    if (it == computeBlocks.end())
      continue;
    for (Operation *op : it->second.ops) {
      int curId = CVPipeline::getOpBlockId(op).value_or(id);
      CVPipeline::setSubBlockId(op, curId);
    }
  }
}

/// Step 4: Try to directly merge succV into predV.
/// Returns true if merge succeeded.
static bool tryDirectMerge(ArrayRef<Operation *> opsToMerge,
                           const CVPipeline::MemoryDependenceGraph &memGraph,
                           int predVId, int succVId,
                           const DenseMap<int, ComputeBlock> &computeBlocks,
                           CVPipeline::ComputeBlockIdManager &bm) {
  if (willCreateCycle(opsToMerge, memGraph, predVId, bm))
    return false;
  LOG_DEBUG("Successfully direct merge: " << succVId << " -> " << predVId);

  markSubBlock(computeBlocks, predVId, succVId);
  for (Operation *op : opsToMerge)
    bm.updateBlockId(op, predVId);
  return true;
}

/// Step 5e: Collect ops in CubePre that need to be cloned to Cube to break
/// the cycle. Returns the set of ops to clone (empty if none).
static SmallPtrSet<Operation *, 16>
collectOpsToBreakCycle(int cubePreId,
                       const DenseMap<int, ComputeBlock> &computeBlocks,
                       const SmallPtrSet<Operation *, 16> &allPredecessors) {
  auto cbIt = computeBlocks.find(cubePreId);

  SmallPtrSet<Operation *, 16> opsToClone;
  for (Operation *op : allPredecessors) {
    if (llvm::is_contained(cbIt->second.ops, op))
      opsToClone.insert(op);
  }
  return opsToClone;
}

/// Step 5: Try cross-Cube clone to break the cycle, then merge.
/// Returns true if merge succeeded after clone.
static bool tryCrossCubeCloneMerge(
    Block *block, const DenseMap<int, ComputeBlock> &computeBlocks,
    const DenseMap<int, SmallVector<int>> &preds,
    const DenseMap<int, DenseMap<int, SmallVector<Operation *>>> &blockEdges,
    int predVId, int succVId, ArrayRef<Operation *> opsToMerge,
    const CVPipeline::MemoryDependenceGraph &memGraph,
    CVPipeline::ComputeBlockIdManager &bm) {
  // 5a. Find succV's CUBE predecessor (Cube)
  std::optional<int> cubeId = findCubePred(succVId, computeBlocks, preds);
  if (!cubeId) {
    LOG_DEBUG("MergeComputeBlock: succV "
              << succVId
              << " has no CUBE predecessor, skipping cross-Cube clone");
    return false;
  }

  // 5b. Find Cube's CUBE predecessor (CubePre)
  std::optional<int> cubePreId = findCubePred(*cubeId, computeBlocks, preds);
  if (!cubePreId) {
    LOG_DEBUG("MergeComputeBlock: Cube "
              << *cubeId
              << " has no CUBE predecessor, skipping cross-Cube clone");
    return false;
  }

  // 5c. Check if Cube depends on CubePre via to_tensor
  SmallVector<Operation *> toTensorOps;
  auto edgeIt = blockEdges.find(*cubePreId);
  if (edgeIt != blockEdges.end()) {
    auto dstIt = edgeIt->second.find(*cubeId);
    if (dstIt != edgeIt->second.end()) {
      toTensorOps =
          findToTensorDeps(&computeBlocks.at(*cubePreId), dstIt->second);
    }
  }
  if (toTensorOps.empty()) {
    LOG_DEBUG("MergeComputeBlock: Cube("
              << *cubeId << ") depends on CubePre(" << *cubePreId
              << ") but not via to_tensor, skipping");
    return false;
  }

  // 5d. Trace back all transitive dependencies
  SmallPtrSet<Operation *, 16> allPredecessors;
  for (Operation *toTensor : toTensorOps) {
    collectAllDeps(toTensor, memGraph, allPredecessors);
  }

  // 5e. Filter to ops in CubePre and clone to Cube
  SmallPtrSet<Operation *, 16> opsToClone =
      collectOpsToBreakCycle(*cubePreId, computeBlocks, allPredecessors);

  for (auto op : opsToClone)
    LOG_DEBUG("Cloning op: " << *op);
  cloneOpCrossCubeDep(*cubePreId, *cubeId, opsToClone, computeBlocks, bm);

  // 5f. Re-check cycle after cloning
  if (willCreateCycle(opsToMerge, memGraph, predVId, bm)) {
    LOG_DEBUG("MergeComputeBlock: still creates cycle after cross-Cube clone "
              "for VECTOR "
              << predVId << " -> " << succVId);
    return false;
  }

  LOG_DEBUG("MergeComputeBlock: Successfully merge after cross-Cube clone: "
            << succVId << " -> " << predVId);
  markSubBlock(computeBlocks, predVId, succVId);
  for (Operation *op : opsToMerge)
    bm.updateBlockId(op, predVId);
  return true;
}

/// Core merge logic for one scf::ForOp body Block.
static bool tryMergeInBlock(Block *block, CVPipeline::ComputeBlockIdManager &bm,
                            const CVPipeline::MemoryDependenceGraph &memGraph) {
  bool mergedAny = false;
  while (true) {
    // Step 1: Group and build enhanced dependency graph
    /// computeBlocks: block_id → ComputeBlock
    DenseMap<int, ComputeBlock> computeBlocks;
    /// succs/preds: block_id → successor/predecessor block‘s blockid list
    DenseMap<int, SmallVector<int>> succs;
    DenseMap<int, SmallVector<int>> preds;
    /// blockEdges: srcId → dstId → source ops (cross-block SSA/memory deps)
    DenseMap<int, DenseMap<int, SmallVector<Operation *>>> blockEdges;
    groupAndBuildGraph(block, memGraph, computeBlocks, succs, preds,
                       blockEdges);
    if (computeBlocks.empty())
      return mergedAny;

    // Step 2: Collect VECTOR candidates
    SmallVector<int> vecCandidates =
        collectVectorCandidates(computeBlocks, succs, preds);
    if (vecCandidates.size() < 2) {
      LOG_DEBUG("MergeComputeBlock: vecCandidates.size() < 2, skipping");
      return mergedAny;
    }

    // Step 3: Find a pair of adjacent VECTOR blocks
    auto pairOpt = findAdjacentVectorPair(vecCandidates, succs);
    if (!pairOpt) {
      LOG_DEBUG("MergeComputeBlock: No adjacent VECTOR pair found, skipping");
      return mergedAny;
    }
    int predVId = pairOpt->first;
    int succVId = pairOpt->second;

    LOG_DEBUG("Found VECTOR pair: predV=" << predVId << ", succV=" << succVId);

    SmallVector<Operation *> opsToMerge = computeBlocks.at(succVId).ops;

    // Step 4: Try direct merge
    if (tryDirectMerge(opsToMerge, memGraph, predVId, succVId, computeBlocks,
                       bm)) {
      mergedAny = true;
      continue;
    }

    // Step 5: Try cross-Cube clone merge
    if (!tryCrossCubeCloneMerge(block, computeBlocks, preds, blockEdges,
                                predVId, succVId, opsToMerge, memGraph, bm))
      return mergedAny;
    // continue to try next pair
    mergedAny = true;
  }
}

namespace {

class MergeComputeBlockPass
    : public PassWrapper<MergeComputeBlockPass, OperationPass<ModuleOp>> {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(MergeComputeBlockPass)

  MergeComputeBlockPass() = default;

  StringRef getArgument() const override { return "merge-compute-block"; }

  StringRef getDescription() const override {
    return "Merge adjacent vector compute blocks between/around CUBE blocks";
  }

  void runOnOperation() override {
    ModuleOp module = getOperation();

    if (CVPipeline::hasFallbackAttr(module)) {
      return;
    }

    // Record that this pass ran. Stays false unless a merge actually
    // succeeds; the following ReorderOpsByBlockIdPass reads this marker.
    module->setAttr(CVPipeline::kMergeComputeBlockApplied,
                    BoolAttr::get(&getContext(), false));

    bool shouldRun = false;
    for (auto funcOp : module.getOps<func::FuncOp>()) {
      if (llvm::is_contained(kEnableMergeComputeBlockKernels,
                             funcOp.getSymName())) {
        LOG_DEBUG(
            "Enable MergeComputeBlock for kernel: " << funcOp.getSymName());
        shouldRun = true;
        break;
      }
    }
    if (!shouldRun) {
      return;
    }

    BufferCountManager bufMgr(module);
    int intraBufCount =
        bufMgr.getBufferCountByType(BufferCountManager::DepType::IntraCore);
    int interCoreBufCount =
        bufMgr.getBufferCountByType(BufferCountManager::DepType::InterCore);
    if (intraBufCount < 3 || interCoreBufCount < 2) {
      LOG_DEBUG("MergeComputeBlock disabled: intraBufCount < 3 or "
                "interCoreBufCount < 2");
      return;
    }

    LOG_DEBUG("Before MergeComputeBlockPass: " << *module);

    SmallVector<Block *> blocksToProcess;
    llvm::LogicalResult walkResult =
        CVPipeline::SplitIf::walkMainLoop(module, [&](Operation *loop) {
          blocksToProcess.push_back(&loop->getRegion(0).front());
          return success();
        });
    if (failed(walkResult)) {
      return;
    }

    auto &aa = getAnalysis<AliasAnalysis>();
    CVPipeline::MemoryDependenceGraph memGraph(module, aa);
    CVPipeline::ComputeBlockIdManager bm(module);
    bool mergedAny = false;
    for (Block *block : blocksToProcess) {
      LOG_DEBUG("try merge in LoopBlock");
      mergedAny = tryMergeInBlock(block, bm, memGraph) || mergedAny;
    }
    if (mergedAny) {
      // A merge actually happened; let the following
      // ReorderOpsByBlockIdPass know it should run.
      module->setAttr(CVPipeline::kMergeComputeBlockApplied,
                      BoolAttr::get(&getContext(), true));
    }

    LOG_DEBUG("After MergeComputeBlockPass: " << *module);
  }
};

} // namespace

// ============================================================================
// Pass Registration
// ============================================================================

namespace mlir {
namespace triton {

std::unique_ptr<OperationPass<ModuleOp>> createMergeComputeBlockPass() {
  return std::make_unique<MergeComputeBlockPass>();
}

void registerMergeComputeBlockPass() {
  PassRegistration<MergeComputeBlockPass> reg;
}

} // namespace triton
} // namespace mlir
