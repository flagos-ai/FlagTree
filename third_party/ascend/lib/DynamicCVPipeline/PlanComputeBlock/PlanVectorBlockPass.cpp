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

#include "DynamicCVPipeline/PlanComputeBlock/ComputeBlockIdManager.h"
#include "ascend/include/DynamicCVPipeline/Common/SyncWall.h"
#include "ascend/include/DynamicCVPipeline/Common/Utils.h"
#include "ascend/include/DynamicCVPipeline/PlanComputeBlock/Common.h"
#include "ascend/include/DynamicCVPipeline/PlanComputeBlock/Passes.h"
#include "bishengir/Dialect/Annotation/IR/Annotation.h"
#include "mlir/Analysis/AliasAnalysis.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/Block.h"
#include "mlir/IR/Operation.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Support/WalkResult.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/LogicalResult.h"
#include "llvm/Support/raw_ostream.h"
#include <algorithm>
#include <memory>

static constexpr const char *DEBUG_TYPE = "plan-vector-block";
#define LOG_DEBUG(...)                                                         \
  LLVM_DEBUG(llvm::dbgs() << " [" << DEBUG_TYPE << "] " << __VA_ARGS__)

using namespace mlir;
using namespace triton;
using namespace CVPipeline;

static bool isFusableOp(Operation *op) {
  if (CVPipeline::isSyncOp(op)) {
    return false;
  }
  if (isVectorSimpleOpOrCf(op)) {
    // skip terminators
    if (op->getBlock()->mightHaveTerminator() &&
        op == op->getBlock()->getTerminator()) {
      return false;
    }
    return true;
  }
  return false;
}

static void
passAndCollectCandidates(Operation *nowOp, DenseMap<Operation *, int> &indegree,
                         SmallVector<Operation *> &candidates,
                         DenseMap<Operation *, bool> &visited,
                         const CVPipeline::MemoryDependenceGraph &memGraph,
                         ComputeBlockIdManager &bm) {
  LOG_DEBUG("Bypassing non-fusable op " << *nowOp << "\nnow candidates size: "
                                        << candidates.size() << "\n");
  DependencyHelper depHelper{memGraph};
  depHelper.forEachUserInSameBlock(nowOp, [&](Operation *user) {
    if (!bm.isSameBlock(user, nowOp)) {
      indegree[user]--;
    }

    if (!bm.isWholeCubeReady(user, indegree)) {
      return;
    }

    if (visited[user]) {
      return;
    }

    if (isFusableOp(user)) {
      visited[user] = true;
      candidates.push_back(user);
      return;
    }

    for (auto *cubeop : bm.getOpsInSameBlock(user)) {
      if (!visited[cubeop]) {
        visited[cubeop] = true;
        passAndCollectCandidates(cubeop, indegree, candidates, visited,
                                 memGraph, bm);
      }
    }
  });

  LOG_DEBUG("After bypassing, candidates size: " << candidates.size() << "\n");
}

static void byPassNonFusable(DenseMap<Operation *, int> &indegree,
                             SmallVector<Operation *> &candidates,
                             DenseMap<Operation *, bool> &visited,
                             const CVPipeline::MemoryDependenceGraph &memGraph,
                             ComputeBlockIdManager &bm) {
  // for every non-fusable candidates, bypass it.
  for (auto [op, _] : indegree) {
    if (!bm.isWholeCubeReady(op, indegree) || isFusableOp(op) || visited[op]) {
      continue;
    }
    for (auto cubeop : bm.getOpsInSameBlock(op)) {
      if (!visited[cubeop]) {
        visited[cubeop] = true;
        passAndCollectCandidates(cubeop, indegree, candidates, visited,
                                 memGraph, bm);
      }
    }
  }
}

static void
updateCandidates(Operation *nextFused, SmallVector<Operation *> &candidates,
                 DenseMap<Operation *, int> &indegree,
                 DenseMap<Operation *, bool> &visited,
                 const CVPipeline::MemoryDependenceGraph &memGraph) {
  // 1. Already fuse with nextFused, so remove it from candidates
  for (auto it = candidates.begin(); it != candidates.end(); it++) {
    if (*it == nextFused) {
      it = candidates.erase(it);
      break;
    }
  }

  // 2. Add new candidates whose indegree becomes 0 after fusing nextFused.
  DependencyHelper depHelper{memGraph};
  depHelper.forEachUserInSameBlock(nextFused, [&](Operation *user) {
    if (!visited[user]) {
      indegree[user]--;
      if (indegree[user] == 0 && isFusableOp(user)) {
        visited[user] = true;
        candidates.push_back(user);
      }
    }
  });
}

void findCandidates(DenseMap<Operation *, int> &indegree,
                    SmallVector<Operation *> &candidates,
                    DenseMap<Operation *, bool> &visited,
                    const CVPipeline::MemoryDependenceGraph &memGraph,
                    ComputeBlockIdManager &bm) {
  // 1. if no candidate, try to bypass non-fusable
  if (candidates.empty()) {
    byPassNonFusable(indegree, candidates, visited, memGraph, bm);
  }
  // 2. find candidates whose indegree is 0 and not visited, add them to
  // candidates and mark visited
  for (auto &[op, degree] : indegree) {
    if (degree == 0 && isFusableOp(op) && !visited[op]) {
      visited[op] = true;
      candidates.push_back(op);
    }
  }
}

static SmallVector<Operation *>
findOpsAdjacentToCube(Block *block, const SmallVector<Operation *> &fuseGroup,
                      DenseMap<Operation *, bool> &visited,
                      const CVPipeline::MemoryDependenceGraph &memGraph,
                      CVPipeline::ComputeBlockIdManager &bm) {
  SmallVector<Operation *> toProcess;
  DependencyHelper depHelper{memGraph};
  // If the nonFusable is contorl, we default first considering.
  // Otheriwe, the smaller blockId, means the matmul is more front in the IR
  // order.
  std::optional<int> minBlockId = std::nullopt;

  for (Operation *op : fuseGroup) {
    depHelper.forEachUserInSameBlock(op, [&](Operation *userInBlock) {
      if (block->mightHaveTerminator() &&
          userInBlock == block->getTerminator()) {
        return WalkResult::advance();
      }
      if (!isFusableOp(userInBlock) && !visited[userInBlock]) {
        auto newBlockId = bm.getBlockIdByOp(userInBlock);
        if (!minBlockId.has_value() || newBlockId < minBlockId) {
          minBlockId = newBlockId;
        }
      }
      return WalkResult::advance();
    });
  }

  if (!minBlockId.has_value()) {
    return {};
  }

  for (Operation *op : fuseGroup) {

    depHelper.forEachUserInSameBlock(op, [&](Operation *userInBlock) {
      if (block->mightHaveTerminator() &&
          userInBlock == block->getTerminator()) {
        return WalkResult::advance();
      }
      if (!isFusableOp(userInBlock) && !visited[userInBlock]) {
        auto newBlockId = bm.getBlockIdByOp(userInBlock);
        if (newBlockId == minBlockId) {
          toProcess.push_back(op);
          return WalkResult::interrupt();
        }
      }
      return WalkResult::advance();
    });
  }
  return toProcess;
}

static SetVector<Operation *>
collectKeepOpsToCube(Block *block, SmallVector<Operation *> toProcess,
                     const SmallVector<Operation *> &fuseGroup,
                     const CVPipeline::MemoryDependenceGraph &memGraph) {
  SetVector<Operation *> keepOps;
  DependencyHelper depHelper{memGraph};
  while (!toProcess.empty()) {
    Operation *op = toProcess.front();
    toProcess.erase(toProcess.begin());
    if (keepOps.contains(op)) {
      continue;
    }
    keepOps.insert(op);

    depHelper.forEachSource<DependencyHelper::SourceMode::AcrossIterArg>(
        op, [&](Operation *source) {
          if (!keepOps.contains(source) &&
              llvm::is_contained(fuseGroup, source)) {
            toProcess.push_back(source);
          }
        });
  }

  // special case: annotation.mark always follows the defining op
  for (auto op : fuseGroup) {
    auto markOp = llvm::dyn_cast<annotation::MarkOp>(op);
    if (!markOp) {
      continue;
    }

    auto src = markOp.getSrc();
    auto definingOp = src.getDefiningOp();
    if (definingOp && keepOps.contains(definingOp)) {
      keepOps.insert(markOp);
    }
  }

  return keepOps;
}

static bool isTensorOperation(Operation *op) {
  for (auto result : op->getResults()) {
    if (isa<RankedTensorType>(result.getType())) {
      return true;
    }
  }
  for (auto operand : op->getOperands()) {
    if (isa<RankedTensorType>(operand.getType())) {
      return true;
    }
  }
  return false;
}

static bool hasTensorOperation(const SmallVector<Operation *> &fuseGroup) {
  for (auto op : fuseGroup) {
    if (isTensorOperation(op)) {
      return true;
    }
  }
  return false;
}

static void
collectAllUsersInFuseGroup(Operation *op,
                           const SmallVector<Operation *> &fuseGroup,
                           SetVector<Operation *> &toRemove) {
  if (toRemove.contains(op) || !llvm::is_contained(fuseGroup, op)) {
    return;
  }
  toRemove.insert(op);
  for (auto user : op->getUsers()) {
    collectAllUsersInFuseGroup(user, fuseGroup, toRemove);
  }
}

static void
collectAllDependenciesInFuseGroup(Operation *op,
                                  const SmallVector<Operation *> &fuseGroup,
                                  SetVector<Operation *> &toRemove) {
  if (toRemove.contains(op) || !llvm::is_contained(fuseGroup, op)) {
    return;
  }
  toRemove.insert(op);
  op->walk([&](Operation *nestedOp) {
    for (auto operand : nestedOp->getOperands()) {
      if (auto definingOp = operand.getDefiningOp()) {
        collectAllDependenciesInFuseGroup(definingOp, fuseGroup, toRemove);
      }
    }
  });
}

static SmallVector<Operation *>
extractWholeFuseGroup(Block *block,
                      const SmallVector<Operation *> &nowFuseGroup,
                      ComputeBlockIdManager &bm) {
  SmallVector<Operation *> toProcess(nowFuseGroup.begin(), nowFuseGroup.end());

  if (!hasTensorOperation(nowFuseGroup)) {
    LOG_DEBUG("no tensor operation in nowFuseGroup.....\n");
    return toProcess;
  }

  SetVector<Operation *> toRemove;
  auto *terminator = block->getTerminator();
  if (llvm::isa_and_present<scf::YieldOp>(terminator) &&
      isa<scf::ForOp, scf::WhileOp>(block->getParentOp())) {
    for (auto *op : nowFuseGroup) {
      auto walkResult = op->walk(
          [block, terminator, &bm, &nowFuseGroup](Operation *nestedOp) {
            for (auto operand : nestedOp->getOperands()) {
              auto *defOp = getLoopCarriedDefOp(operand, block);
              if (defOp && bm.getBlockIdByOp(defOp) == -1 &&
                  !llvm::is_contained(nowFuseGroup, defOp)) {
                return WalkResult::interrupt();
              }
            }
            return WalkResult::advance();
          });
      if (walkResult.wasInterrupted()) {
        collectAllUsersInFuseGroup(op, nowFuseGroup, toRemove);
      }
    }
  }

  for (auto op : toRemove) {
    LOG_DEBUG("Removing op when refining: " << *op << "\n");
    toProcess.erase(std::remove(toProcess.begin(), toProcess.end(), op),
                    toProcess.end());
  }

  if (!toProcess.empty() && !hasTensorOperation(toProcess)) {
    LOG_DEBUG("no tensor operation in toProcess.....\n");
    return {};
  }
  return toProcess;
}

static void evictAndRestoreState(
    Block *block, const SetVector<Operation *> &keepOps,
    SmallVector<Operation *> &fuseGroup, DenseMap<Operation *, bool> &visited,
    SmallVector<Operation *> &candidates, DenseMap<Operation *, int> &indegree,
    const CVPipeline::MemoryDependenceGraph &memGraph) {
  // 1. Collect ops to remove
  SmallVector<Operation *> toRemove;
  for (Operation *op : fuseGroup) {
    if (!keepOps.contains(op)) {
      toRemove.push_back(op);
    }
  }
  fuseGroup.assign(keepOps.begin(), keepOps.end());

  DependencyHelper depHelper{memGraph};
  // 2. Restore indegree for successors and reset visited for removed ops
  for (Operation *op : toRemove) {
    depHelper.forEachUserInSameBlock(
        op, [&](Operation *user) { indegree[user]++; });
    visited[op] = false;
  }

  // 3. Reset visited for current candidates so they can re-enter
  for (auto cand : candidates) {
    visited[cand] = false;
  }

  // 4. Rebuild candidates from (old candidates + removed ops), zero indegree
  // only
  SmallVector<Operation *> pool(candidates.begin(), candidates.end());
  pool.append(toRemove.begin(), toRemove.end());
  candidates.clear();
  // Need to find source again
  for (Operation *op : pool) {
    if (indegree[op] == 0 && !visited[op]) {
      visited[op] = true;
      candidates.push_back(op);
    }
  }
}

void refineFuseGroup(Block *block, SmallVector<Operation *> &nowFuseGroup,
                     DenseMap<Operation *, bool> &visited,
                     SmallVector<Operation *> &candidates,
                     DenseMap<Operation *, int> &indegree,
                     const CVPipeline::MemoryDependenceGraph &memGraph,
                     ComputeBlockIdManager &bm) {
  // 1.Find ops in fuse group whose next node is a non-fusable (CUBE-only) op
  auto toProcess =
      findOpsAdjacentToCube(block, nowFuseGroup, visited, memGraph, bm);

  // 2. If no cube adjacent op, extract toProcess from fuseGroup using fallback
  // rules
  if (toProcess.empty()) {
    LOG_DEBUG("No Cube adjacent op, extracting toProcess from fuseGroup.\n");
    toProcess = extractWholeFuseGroup(block, nowFuseGroup, bm);
  }

  // 3. If still empty after extraction, no op will be cut
  if (toProcess.empty()) {
    LOG_DEBUG("No op will be cut after extraction.\n");
    findCandidates(indegree, candidates, visited, memGraph, bm);
    if (candidates.empty()) {
      // Even if cut these ops, and add them into next search, they will be cut
      // again and lead to dead cycle.
      //  the scenario like this:
      // v1->v2->yield.
      // So after findCandidates, no more new ops, need to fuse nowFuseGroup.
      return;
    }
  }

  // 4. Collect keepOps transitively (data + memory + loop-carried deps)
  auto keepOps = collectKeepOpsToCube(block, toProcess, nowFuseGroup, memGraph);

  // 5. Remove non-kept ops from fuseGroup and restore BFS state
  evictAndRestoreState(block, keepOps, nowFuseGroup, visited, candidates,
                       indegree, memGraph);
  LOG_DEBUG("After cutting, kept " << keepOps.size() << "\n");
}

static SmallVector<Operation *>
findOpsAdjacentFromCube(Block *block, const SmallVector<Operation *> &fuseGroup,
                        DenseMap<Operation *, bool> &visited,
                        const CVPipeline::MemoryDependenceGraph &memGraph,
                        CVPipeline::ComputeBlockIdManager &bm) {
  SmallVector<Operation *> toProcess;
  DependencyHelper depHelper{memGraph};
  std::optional<int> minBlockId = std::nullopt;

  for (Operation *op : fuseGroup) {
    depHelper.forEachSourceInSameBlock(op, [&](Operation *defInBlock) {
      if (!isFusableOp(defInBlock) && !visited[defInBlock]) {
        auto newBlockId = bm.getBlockIdByOp(defInBlock);
        if (!minBlockId.has_value() || newBlockId < minBlockId) {
          minBlockId = newBlockId;
        }
      }
    });
  }

  if (!minBlockId.has_value()) {
    return {};
  }

  for (Operation *op : fuseGroup) {
    depHelper.forEachSourceInSameBlock(op, [&](Operation *defInBlock) {
      if (!isFusableOp(defInBlock) && !visited[defInBlock]) {
        auto newBlockId = bm.getBlockIdByOp(defInBlock);
        if (newBlockId == minBlockId) {
          toProcess.push_back(op);
          return WalkResult::interrupt();
        }
      }
      return WalkResult::advance();
    });
  }
  return toProcess;
}

SetVector<Operation *>
collectKeepOpsFromCube(Block *block, SmallVector<Operation *> toProcess,
                       const SmallVector<Operation *> &fuseGroup,
                       const CVPipeline::MemoryDependenceGraph &memGraph) {
  SetVector<Operation *> keepOps;
  DependencyHelper depHelper{memGraph};
  while (!toProcess.empty()) {
    Operation *op = toProcess.front();
    toProcess.erase(toProcess.begin());
    if (keepOps.contains(op)) {
      continue;
    }
    keepOps.insert(op);

    // Add all users to process
    depHelper.forEachUserInSameBlock(op, [&](Operation *userInblock) {
      if (!keepOps.contains(userInblock) &&
          llvm::is_contained(fuseGroup, userInblock)) {
        toProcess.push_back(userInblock);
      }
    });
  }
  return keepOps;
}

void reverseRefineFuseGroup(Block *block,
                            SmallVector<Operation *> &nowFuseGroup,
                            DenseMap<Operation *, bool> &visited,
                            SmallVector<Operation *> &candidates,
                            DenseMap<Operation *, int> &indegree,
                            const CVPipeline::MemoryDependenceGraph &memGraph,
                            ComputeBlockIdManager &bm) {
  // 1. Find ops in fuse group whose pre node is a non-fusable op
  auto toProcess =
      findOpsAdjacentFromCube(block, nowFuseGroup, visited, memGraph, bm);
  // 2. If no op from cube, extract toProcess from fuseGroup using fallback
  // rules
  if (toProcess.empty()) {
    LOG_DEBUG("No op from cube, extracting toProcess from fuseGroup.\n");
    return;
  }
  // 3. Collect keepOps transitively (data + memory + loop-carried deps)
  auto keepOps =
      collectKeepOpsFromCube(block, toProcess, nowFuseGroup, memGraph);
  // 4. Remove non-kept ops from fuseGroup and restore BFS state
  evictAndRestoreState(block, keepOps, nowFuseGroup, visited, candidates,
                       indegree, memGraph);
  LOG_DEBUG("After reverse cutting, kept " << keepOps.size() << "\n");
}

// Main function to plan vector block id for one block
llvm::LogicalResult
planVectorBlockId(Block *block,
                  const CVPipeline::MemoryDependenceGraph &memGraph,
                  ComputeBlockIdManager &bm) {
  // 1. topo initialize
  llvm::DenseMap<Operation *, int> indegree;
  llvm::SmallVector<Operation *> queue;
  llvm::DenseMap<Operation *, bool> visited; // has been visited in search
  initializeIndegreeForBlock(block, indegree, DependencyHelper{memGraph}, bm);

  // Source-order sync walls of this block; used to keep every vector group on
  // a single side of each sync, otherwise the ReorderOpsByBlockId fence adds
  // both (group -> sync) and (sync -> group) edges and scheduling fails.
  SyncWall wall(block);

  // 2. initialize visited and find initial candidates
  block->walk([&](Operation *op) {
    if (op->getBlock() == block) {
      visited[op] = false;
      if (isFusableOp(op) && indegree[op] == 0) {
        visited[op] = true;
        queue.push_back(op);
      }
    }
  });
  findCandidates(indegree, queue, visited, memGraph, bm);

  // 3. find fuse group follow topo order
  llvm::SmallVector<Operation *> nowFuseGroup;
  while (!queue.empty()) {
    auto nextFused = queue.front();
    if (nextFused) {
      // fused one && update candidates
      nowFuseGroup.push_back(nextFused);
      updateCandidates(nextFused, queue, indegree, visited, memGraph);
    }
    if (queue.empty() || nextFused == nullptr) {
      LLVM_DEBUG({
        LOG_DEBUG("Prepare to check this group: \n");
        for (auto op : nowFuseGroup) {
          LOG_DEBUG("prepare: " << *op << "\n");
        }
      });
      // finish one group, assign block id and start next iteration
      // Cut error operations before assigning block id
      refineFuseGroup(block, nowFuseGroup, visited, queue, indegree, memGraph,
                      bm);

      reverseRefineFuseGroup(block, nowFuseGroup, visited, queue, indegree,
                             memGraph, bm);
      LOG_DEBUG("Group after cutting: \n");
      for (auto op : nowFuseGroup) {
        LOG_DEBUG("fuseing: " << *op << "\n");
      }
      // Assign one fresh block id per sync-free segment. A topo wave of
      // mutually independent ready ops may span several segments separated by
      // syncs; splitting here guarantees no block_id group straddles a sync.
      llvm::DenseMap<unsigned, llvm::SmallVector<Operation *>> segmentGroups;
      for (auto *op : nowFuseGroup) {
        segmentGroups[wall.segmentOf(op)].push_back(op);
      }
      for (auto &segGroup : segmentGroups) {
        if (llvm::failed(bm.markOpsWithNewId(segGroup.second))) {
          return llvm::failure();
        }
      }
      nowFuseGroup.clear();
      // reset queue
      findCandidates(indegree, queue, visited, memGraph, bm);
    }
  }
  return llvm::success();
}

namespace {

class PlanVectorBlockPass
    : public PassWrapper<PlanVectorBlockPass, OperationPass<ModuleOp>> {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(PlanVectorBlockPass)

  PlanVectorBlockPass() = default;
  void runOnOperation() override;

  [[nodiscard]] llvm::StringRef getArgument() const final {
    return "plan-vector-block";
  }
};

void PlanVectorBlockPass::runOnOperation() {
  // 1. Build memory dependence graph
  auto moduleOp = getOperation();

  if (CVPipeline::hasFallbackAttr(moduleOp)) {
    return;
  }

  auto &aa = getAnalysis<AliasAnalysis>();
  auto memDepGraph = MemoryDependenceGraph(moduleOp, aa);
  auto bm = ComputeBlockIdManager(moduleOp);

  // 2. search blocks in topo order and assign block id for each block
  auto result =
      moduleOp.walk<WalkOrder::PreOrder>([&](Block *block) -> WalkResult {
        if (bm.shouldInheritFromParent(block, CoreType::VECTOR_ONLY)) {
          LOG_DEBUG("Inherit from parentOp: " << block->getParentOp() << "\n");
          if (llvm::failed(bm.inheritFromParent(block))) {
            block->getParentOp()->emitError()
                << "[" << DEBUG_TYPE
                << "] Sub-blocks failed to inherit block id from parent op";
            return WalkResult::interrupt();
          }
          return WalkResult::advance();
        }
        if (llvm::failed(planVectorBlockId(block, memDepGraph, bm))) {
          return WalkResult::interrupt();
        }
        return WalkResult::advance();
      });
  if (result.wasInterrupted()) {
    LOG_DEBUG("Failed to plan vector block id for block\n");
    CVPipeline::setFallbackAttr(moduleOp, CVPipeline::ERRCODE_FAILED);
  }
}

} // namespace

namespace mlir::triton {

std::unique_ptr<OperationPass<ModuleOp>> createPlanVectorBlockPass() {
  return std::make_unique<PlanVectorBlockPass>();
}
} // namespace mlir::triton
