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

#include <queue>
#include <utility>

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/iterator.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/LogicalResult.h"
#include "llvm/Support/raw_ostream.h"

#include "DynamicCVPipeline/Common/CycleDetector.h"
#include "mlir/Analysis/AliasAnalysis.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/Block.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/Visitors.h"
#include "mlir/Pass/Pass.h"

#include "ascend/include/DynamicCVPipeline/Common/MemoryEffectsTracker.h"
#include "ascend/include/DynamicCVPipeline/Common/SyncWall.h"
#include "ascend/include/DynamicCVPipeline/Common/Utils.h"
#include "ascend/include/DynamicCVPipeline/ComputeBlockOpt/Common.h"
#include "ascend/include/DynamicCVPipeline/PlanComputeBlock/Common.h"
#include "ascend/include/DynamicCVPipeline/PlanComputeBlock/PlanCubeBlockPass.h"

#include "bishengir/Dialect/Annotation/IR/Annotation.h"
#include "bishengir/Dialect/HIVM/IR/HIVM.h"

using namespace mlir;
using namespace triton;
using namespace CVPipeline;

static constexpr const char *DEBUG_TYPE = "plan-cube-block";
#define LOG_DEBUG(...)                                                         \
  LLVM_DEBUG(llvm::dbgs() << " [" << DEBUG_TYPE << "] " << __VA_ARGS__)

static bool isMatmulOp(Operation *op) { return isa<linalg::MatmulOp>(op); }

static bool isCubeSimpleOpOrCf(Operation *op) {
  if (op->getBlock()->mightHaveTerminator() &&
      op == op->getBlock()->getTerminator()) {
    return false;
  }
  return !isSyncOp(op) && getCoreTypeOfSimpleOpOrCf(op) == CoreType::CUBE_ONLY;
}

namespace {

class SeedRegionPlanner {
  SmallVector<Operation *> seeds;
  Block *block;
  const DependencyHelper &depHelper;
  ComputeBlockIdManager &bm;
  llvm::DenseSet<Operation *> &assigned;
  llvm::SmallVectorImpl<Operation *> &group;
  const SyncWall &wall;
  bool willCreateCycle(Operation *op);
  bool isEligible(Operation *op);
  bool tryAddToGroup(Operation *op, Operation *from);

public:
  SeedRegionPlanner(SmallVector<Operation *> seeds, Block *block,
                    const DependencyHelper &depHelper,
                    llvm::DenseSet<Operation *> &assigned,
                    llvm::SmallVectorImpl<Operation *> &group,
                    ComputeBlockIdManager &bm, const SyncWall &wall)
      : seeds(seeds), block(block), depHelper(depHelper), assigned(assigned),
        group(group), bm(bm), wall(wall) {
    for (auto sd : seeds) {
      group.push_back(sd);
    }
  }

  void run();
};

} // namespace

bool SeedRegionPlanner::willCreateCycle(Operation *op) {
  auto *block = op->getBlock();
  llvm::DenseSet<mlir::Operation *> okSet(group.begin(), group.end());
  okSet.insert(op);

  DependencyCycleDetector dfs = {block, depHelper, okSet, bm};
  return dfs.detectCycle();
}

/**
 * Checks if an operation is eligible to be added to a Cube group.
 * Eligibility depends on it being a cube op, not yet assigned, not a compute
 * op, and not creating a cycle in the dependence graph.
 */
bool SeedRegionPlanner::isEligible(Operation *op) {
  if (!isCubeSimpleOpOrCf(op) || assigned.contains(op) || isMatmulOp(op)) {
    return false;
  }
  return !willCreateCycle(op);
}

bool SeedRegionPlanner::tryAddToGroup(Operation *op, Operation *from) {
  if (!op || llvm::is_contained(group, op) || op->getBlock() != block ||
      !isEligible(op)) {
    return false;
  }
  // Never let the group straddle a sync: a dependency edge must not bridge a
  // synchronization point, otherwise the group would appear on both sides of
  // the barrier and the fence built by ReorderOpsByBlockId would cycle.
  if (from && wall.hasSyncBetween(from, op)) {
    return false;
  }
  group.push_back(op);
  return true;
}

void SeedRegionPlanner::run() {
  size_t head = 0;
  while (head < group.size()) {
    Operation *currOp = group[head++];
    depHelper.forEachSource<DependencyHelper::SourceMode::AcrossIterArg>(
        currOp,
        [&, this](Operation *source) { tryAddToGroup(source, currOp); });
  }
}

namespace {

/**
 * Processes remaining unassigned cube operations by following the topological
 * order.
 */
class TopologicalPartitionPlanner {
  Block *block;
  unsigned nonAssignedCubeCnt = 0;
  llvm::DenseMap<Operation *, int> indegree;
  llvm::DenseSet<Operation *> &assigned;
  const DependencyHelper &depHelper;
  ComputeBlockIdManager &bm;
  const SyncWall &wall;
  llvm::DenseSet<Operation *> newassigned;
  llvm::DenseSet<Operation *> bypassVisited;
  std::queue<Operation *> queue;

  void removeNonCubeOpsRecursively(Operation *op);
  llvm::LogicalResult removeReadyNonCubeOps();

  bool shouldSkip(Operation *op) {
    return !isCubeSimpleOpOrCf(op) || assigned.contains(op);
  };
  bool canExpandTo(Operation *op);
  void dumpQueueAndIndegreeInfo();
  llvm::LogicalResult populateQueueWithReadyOps();
  llvm::SmallVector<Operation *> createNewGroupFromQueue();

public:
  TopologicalPartitionPlanner(Block *block,
                              llvm::DenseSet<Operation *> &assigned,
                              const DependencyHelper &depHelper,
                              ComputeBlockIdManager &bm, const SyncWall &wall)
      : block(block), assigned(assigned), depHelper(depHelper), bm(bm),
        wall(wall) {
    initializeIndegreeForBlock(block, indegree, depHelper, bm);

    block->walk([&](Operation *op) {
      if (op->getBlock() == block && isCubeSimpleOpOrCf(op) &&
          !assigned.contains(op)) {
        nonAssignedCubeCnt++;
      }
    });
  }
  llvm::LogicalResult run();
};

} // namespace

// Recursively bypass non-cube ops: decrement indegree of users; collect
// newly-exposed cube ops
void TopologicalPartitionPlanner::removeNonCubeOpsRecursively(Operation *op) {
  LOG_DEBUG("\tRemoved non-cube:" << *op << "\n");
  bypassVisited.insert(op);
  depHelper.forEachUserInSameBlock(op, [&](Operation *user) {
    if (!indegree.contains(user) || bm.isSameBlock(user, op)) {
      return;
    }
    LOG_DEBUG("Sub indegree to " << *user << " from " << *op << "new degree =  "
                                 << indegree[user] - 1 << "\n");
    indegree[user]--;
    if (!bm.isWholeCubeReady(user, indegree) || bypassVisited.contains(user) ||
        !shouldSkip(user)) {
      return;
    }
    for (auto *passop : bm.getOpsInSameBlock(user)) {
      if (!bypassVisited.contains(passop)) {
        removeNonCubeOpsRecursively(passop);
      }
    }
  });
}

/**
 * Logic to bypass non-cube operations that are ready to be executed.
 * This unblocks downstream cube operations in the topological sort.
 */
llvm::LogicalResult TopologicalPartitionPlanner::removeReadyNonCubeOps() {
  auto indegreeBefore = indegree;
  size_t beforeVisitedSize = bypassVisited.size();
  for (auto &p : indegree) {
    Operation *op = p.first;
    if (!shouldSkip(op) || !bm.isWholeCubeReady(op, indegree) ||
        bypassVisited.contains(op)) {
      continue;
    }
    for (auto *passOp : bm.getOpsInSameBlock(op)) {
      if (!bypassVisited.contains(passOp)) {
        removeNonCubeOpsRecursively(passOp);
      }
    }
  }
  if (indegreeBefore == indegree && beforeVisitedSize == bypassVisited.size()) {
    LLVM_DEBUG({
      if (Operation *parentOp = block->getParentOp()) {
        LOG_DEBUG("PlanCubeBlock cannot make progress while scheduling "
                  "cube operations in: "
                  << *parentOp);
      }
      dumpQueueAndIndegreeInfo();
    });
    return llvm::failure();
  }
  return llvm::success();
}

// Expansion condition: op must be CUBE_ONLY, indegree == 0 and all its
// dependency ops are CUBE_ONLY
bool TopologicalPartitionPlanner::canExpandTo(Operation *op) {
  if (!isCubeSimpleOpOrCf(op) || assigned.contains(op)) {
    return false;
  }
  auto it = indegree.find(op);
  return it->second == 0;
}

// Encountered error. Need to print failure reason, so no need for LLVM_DEBUG
void TopologicalPartitionPlanner::dumpQueueAndIndegreeInfo() {
  // simply print a debug header in a new line
  auto errs = []() -> llvm::raw_ostream & {
    return llvm::errs() << "\n[" << DEBUG_TYPE << "] ";
  };

  errs() << "failed to make progress while planning cube blocks.";
  errs() << "remaining cube count: " << nonAssignedCubeCnt;

  errs() << "ready queue";
  if (queue.empty()) {
    llvm::errs() << " is empty.";
  } else {
    llvm::errs() << ":\n";
    while (!queue.empty()) {
      Operation *op = queue.front();
      queue.pop();
      llvm::errs() << "  " << *op << "\n";
    }
  }

  errs() << "remaining unassigned cube ops:\n";
  bool foundRemainingCube = false;
  for (auto &p : indegree) {
    Operation *op = p.first;
    if (!op || op->getBlock() != block || !isCubeSimpleOpOrCf(op) ||
        assigned.contains(op) || newassigned.contains(op)) {
      continue;
    }
    foundRemainingCube = true;
    llvm::errs() << "  indegree=" << p.second << " op=" << *op << "\n";
  }
  if (!foundRemainingCube) {
    llvm::errs() << "  <none>\n";
  }
}

llvm::LogicalResult TopologicalPartitionPlanner::populateQueueWithReadyOps() {
  for (auto [op, indegree] : indegree) {
    if (indegree < 0) {
      op->emitError("Indegree cannot be negative");
      return llvm::failure();
    }
    if (indegree == 0 && !newassigned.contains(op) && isCubeSimpleOpOrCf(op) &&
        !assigned.contains(op)) {
      queue.push(op);
    }
  }
  return llvm::success();
}

llvm::SmallVector<Operation *>
TopologicalPartitionPlanner::createNewGroupFromQueue() {
  llvm::SmallVector<Operation *> group;
  while (!queue.empty()) {
    auto *currOp = queue.front();
    queue.pop();

    newassigned.insert(currOp);
    group.push_back(currOp);
    nonAssignedCubeCnt--;

    depHelper.forEachUserInSameBlock(currOp, [&](Operation *user) {
      if (!newassigned.contains(user)) {
        auto &userInDegree = indegree[user];
        userInDegree--;
        LOG_DEBUG("Sub indegree to " << *user << " from " << *currOp
                                     << "new degree = " << userInDegree
                                     << "\n");
        if (canExpandTo(user)) {
          queue.push(user);
        }
      }
    });
  }
  return group;
}

llvm::LogicalResult TopologicalPartitionPlanner::run() {
  while (nonAssignedCubeCnt > 0) {
    if (failed(populateQueueWithReadyOps())) {
      return llvm::failure();
    }

    if (queue.empty()) {
      if (failed(removeReadyNonCubeOps())) {
        return llvm::failure();
      }
      continue;
    }

    auto group = createNewGroupFromQueue();
    // A wave of mutually-independent ready cube ops may span several segments
    // separated by syncs. Split by segment so no block_id group straddles a
    // synchronization op; each segment gets its own fresh id.
    llvm::DenseMap<unsigned, llvm::SmallVector<Operation *>> segmentGroups;
    for (auto *op : group) {
      segmentGroups[wall.segmentOf(op)].push_back(op);
    }
    for (auto &segGroup : segmentGroups) {
      if (llvm::failed(bm.markOpsWithNewId(segGroup.second))) {
        return llvm::failure();
      }
    }
  }

  return llvm::success();
}

static SmallVector<Operation *> collectMatmulOps(Block *block) {
  SmallVector<Operation *> computeOps;
  for (Operation &op : *block) {
    if (isMatmulOp(&op)) {
      computeOps.push_back(&op);
    }
  }
  return computeOps;
}

static void fuseMarkOpToDef(Block *block, ComputeBlockIdManager &bm,
                            const DependencyHelper &depHelper,
                            const SyncWall &wall) {
  for (auto *op : llvm::make_pointer_range(block->getOperations())) {
    if (getOpCoreType(op) != CUBE_ONLY) {
      continue;
    }
    auto markOp = llvm::dyn_cast<annotation::MarkOp>(op);
    if (!markOp) {
      continue;
    }
    auto *defOp = markOp.getSrc().getDefiningOp();
    if (!defOp) {
      continue;
    }
    // Merging the mark into the def's group would make the group straddle a
    // sync if the mark sits on the other side of one; keep them separate then.
    if (!wall.sameSegment(markOp, defOp)) {
      continue;
    }

    auto defBlockId = bm.getBlockIdByOp(defOp);
    if (defBlockId == -1) {
      continue;
    }

    auto currGroup = bm.getOpsInSameBlock(defOp);
    llvm::DenseSet<Operation *> newGroup{currGroup.begin(), currGroup.end()};

    if (newGroup.contains(markOp)) {
      continue;
    }
    newGroup.insert(markOp);

    DependencyCycleDetector dfs{block, depHelper, newGroup, bm};
    if (!dfs.detectCycle()) {
      bm.updateBlockId(markOp, defBlockId);
    }
  }
}

static bool checkValidInputSeed(Operation *op) {
  // keep unify to OpClassifer
  return isa<linalg::TransposeOp, bufferization::ToTensorOp, linalg::FillOp,
             tensor::EmptyOp, linalg::BroadcastOp, tensor::ExpandShapeOp,
             arith::ExtFOp>(op);
}
static bool checkValidUserSeed(Operation *op, linalg::MatmulOp &preDot) {
  // keep unify to OpClassifer
  if (isa<hivm::StoreOp, bufferization::MaterializeInDestinationOp,
          ViewLikeOpInterface, tensor::ExtractSliceOp>(op)) {
    return true;
  }
  // add matmul link by L0C
  if (auto nextDot = dyn_cast<linalg::MatmulOp>(op)) {
    if (nextDot.getDpsInits()[0] == *preDot->getResults().begin()) {
      preDot = nextDot;
      return true;
    }
    return false;
  }
  return false;
}

static SmallVector<Operation *>
matchSeed(Operation *dotOp, ComputeBlockIdManager &bm,
          const MemoryDependenceGraph &memGraph) {
  // match inputs
  SmallVector<Operation *> ret;
  ret.push_back(dotOp);
  for (Value operand : dotOp->getOperands()) {
    Operation *def = operand.getDefiningOp();
    if (!def)
      continue;
    if (checkValidInputSeed(def) && isCubeSimpleOpOrCf(def) &&
        dotOp->getBlock() == def->getBlock() && bm.getBlockIdByOp(def) == -1) {
      if (CVPipeline::isOnlyDirectlyUse(def, dotOp, memGraph)) {
        ret.push_back(def);
      }
    }
  }
  // match outputs
  Operation *nowOp = dotOp;
  linalg::MatmulOp linkMatmul = llvm::dyn_cast<linalg::MatmulOp>(dotOp);
  while (nowOp->hasOneUse()) {
    auto user = *nowOp->getUsers().begin();
    if (user->getBlock() != dotOp->getBlock() || !isCubeSimpleOpOrCf(user) ||
        bm.getBlockIdByOp(user) != -1) {
      break;
    }
    if (checkValidUserSeed(user, linkMatmul)) {
      nowOp = user;
      ret.push_back(user);
    } else {
      break;
    }
  }
  return ret;
}

/**
 * Main entry point: Process a single block by grouping operations into
 * execution blocks using BFS and topological traversal.
 */
static llvm::LogicalResult
processBlockWithCubeBFS(Block *block, const DependencyHelper &depHelper,
                        ComputeBlockIdManager &bm) {
  llvm::DenseSet<Operation *> assigned;
  auto allDots = collectMatmulOps(block);

  // Source-order sync walls of this block; used to keep every block_id group
  // on a single side of each synchronization op.
  SyncWall wall(block);

  // Phase 1: Add helper ops (transpose, load/store, ptr etc.) to cube block of
  // related matmul
  for (auto *dot : allDots) {
    if (assigned.contains(dot)) {
      continue;
    }
    auto temBlockId = bm.getNextId();
    llvm::SmallVector<Operation *> dotSeeds =
        matchSeed(dot, bm, depHelper.memGraph);
    // Drop seeds straddling a sync relative to the dot: they belong to their
    // own segment and Phase 2 hands them a separate id, so the seed group can
    // never appear on both sides of the barrier.
    llvm::SmallVector<Operation *> prunedSeeds;
    for (auto *seed : dotSeeds) {
      if (wall.sameSegment(seed, dot)) {
        prunedSeeds.push_back(seed);
      }
    }
    llvm::erase_if(dotSeeds, [&](Operation *seed) {
      return !wall.sameSegment(seed, dot);
    });
    if (willCreateCycle(dotSeeds, depHelper.memGraph, temBlockId, bm)) {
      LOG_DEBUG("Cube Seed already have a cycle!!");
      for (auto seed : dotSeeds) {
        LOG_DEBUG("Seed: " << *seed << "\n");
      }
      return llvm::failure();
    }
    llvm::SmallVector<Operation *> newGroup;
    SeedRegionPlanner regionPlanner{dotSeeds, block, depHelper, assigned,
                                    newGroup, bm,    wall};
    regionPlanner.run();

    for (auto *op : newGroup) {
      assigned.insert(op);
    }
    if (llvm::failed(bm.markOpsWithNewId(newGroup))) {
      return llvm::failure();
    }
  }

  // Phase 2: Handle remaining Cube Ops following Topo order
  TopologicalPartitionPlanner topoPlanner{block, assigned, depHelper, bm, wall};
  if (failed(topoPlanner.run())) {
    return failure();
  }
  fuseMarkOpToDef(block, bm, depHelper, wall);
  return llvm::success();
}

void mlir::triton::PlanCubeBlockPass::runOnOperation() {
  auto moduleOp = getOperation();

  if (CVPipeline::hasFallbackAttr(moduleOp)) {
    return;
  }

  LOG_DEBUG("Input mlir:\n" << moduleOp << "\n==========\n");

  auto &aa = getAnalysis<AliasAnalysis>();
  MemoryDependenceGraph memGraph{moduleOp, aa};
  DependencyHelper depHelper{memGraph};
  auto bm = ComputeBlockIdManager(moduleOp);

  // We do not need to skip linalg blocks since they do not have core types and
  // do not contain matmul
  auto result = moduleOp.walk<WalkOrder::PreOrder>([&](Block *block) {
    if (bm.shouldInheritFromParent(block, CoreType::CUBE_ONLY)) {
      if (llvm::failed(bm.inheritFromParent(block))) {
        block->getParentOp()->emitError()
            << "[" << DEBUG_TYPE
            << "] Sub-blocks failed to inherit block id from parent op";
        return WalkResult::interrupt();
      }
      return WalkResult::advance();
    }

    if (llvm::failed(processBlockWithCubeBFS(block, depHelper, bm))) {
      return WalkResult::interrupt();
    }
    return WalkResult::advance();
  });
  if (result.wasInterrupted()) {
    CVPipeline::setFallbackAttr(moduleOp, CVPipeline::ERRCODE_FAILED);
  }
  LOG_DEBUG("Output mlir:\n" << moduleOp << "\n==========\n");
}

std::unique_ptr<OperationPass<ModuleOp>>
mlir::triton::createPlanCubeBlockPass() {
  return std::make_unique<PlanCubeBlockPass>();
}
