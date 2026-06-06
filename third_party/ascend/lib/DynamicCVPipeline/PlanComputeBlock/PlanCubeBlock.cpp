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
#include "llvm/Support/Debug.h"
#include "llvm/Support/LogicalResult.h"
#include "llvm/Support/raw_ostream.h"

#include "mlir/Analysis/AliasAnalysis.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/Block.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Operation.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Support/WalkResult.h"

#include "ascend/include/DynamicCVPipeline/PlanComputeBlock/PlanCubeBlockPass.h"

#include "DynamicCVPipeline/Common/MemoryEffectsTracker.h"
#include "DynamicCVPipeline/PlanComputeBlock/Common.h"
#include "DynamicCVPipeline/PlanComputeBlock/ComputeBlockIdManager.h"

using namespace mlir;
using namespace triton;
using namespace CVPipeline;

static constexpr const char *DEBUG_TYPE = "PlanCubeBlock";
#define LOG_DEBUG(...) LLVM_DEBUG(llvm::dbgs() << " [" << DEBUG_TYPE << "] " << __VA_ARGS__)

static bool isMatmulOp(Operation *op)
{
    return isa<linalg::MatmulOp>(op);
}

namespace {

class SeedRegionPlanner {
    Operation *seed;
    Block *block;
    const MemoryDependenceGraph &memGraph;
    llvm::DenseSet<Operation *> &assigned;
    llvm::SmallVectorImpl<Operation *> &group;
    bool willCreateCycle(Operation *op);
    bool isEligible(Operation *op);
    bool tryAddToGroup(Operation *op);
    void addSourcesToGroup();
    void addUsersToGroup();

public:
    SeedRegionPlanner(Operation *seed,
                      Block *block,
                      const MemoryDependenceGraph &memGraph,
                      llvm::DenseSet<Operation *> &assigned,
                      llvm::SmallVectorImpl<Operation *> &group)
        : seed(seed), block(block), memGraph(memGraph), assigned(assigned), group(group)
    {
        group.push_back(seed);
    }

    void run();
};

} // namespace


namespace {

struct DependencyCycleDetector {
    llvm::DenseSet<mlir::Operation *> &okSet;
    llvm::DenseSet<mlir::Operation *> visited;
    const MemoryDependenceGraph &memGraph;
    Block *block;
    void clear() { visited.clear(); }
    bool operator()(Operation *cur);
    bool dfs(Operation *cur) { return (*this)(cur); };

    DependencyCycleDetector(Block *block,
                            const MemoryDependenceGraph &memGraph,
                            llvm::DenseSet<mlir::Operation *> &okSet)
        : block(block), memGraph(memGraph), okSet(okSet)
    {}
};

} // namespace

bool DependencyCycleDetector::operator()(Operation *cur)
{
    if (okSet.contains(cur)) {
        return true;
    }
    if (!visited.insert(cur).second) {
        return false;
    }

    SmallVector<Operation *> allusers;
    allusers.append(cur->getUsers().begin(), cur->getUsers().end());
    for (auto *memUser : memGraph.getExecAfter(cur)) {
        allusers.push_back(memUser);
    }
    for (auto *user : allusers) {
        auto *userInBlock = getAncestorInBlock(user, block);
        auto &bm = ComputeBlockIdManager::getInstance();
        if (bm.getBlockIdByOp(userInBlock) == -1) {
            if (dfs(userInBlock)) {
                return true;
            }
        } else {
            for (auto *nx : bm.getOpsByBlockId(bm.getBlockIdByOp(userInBlock))) {
                if (dfs(nx)) {
                    return true;
                }
            }
        }
    }
    return false;
}

bool SeedRegionPlanner::willCreateCycle(Operation *op)
{
    auto *block = op->getBlock();
    llvm::DenseSet<mlir::Operation *> okSet(group.begin(), group.end());
    okSet.insert(op);

    DependencyCycleDetector dfs = {block, memGraph, okSet};

    // DFS from every result in okSet
    for (mlir::Operation *okOp : okSet) {
        SmallVector<Operation *> allusers;
        allusers.append(okOp->getUsers().begin(), okOp->getUsers().end());
        for (auto *memUser : memGraph.getExecAfter(okOp)) {
            allusers.push_back(memUser);
        }
        for (auto *user : allusers) {
            auto *userInBlock = getAncestorInBlock(user, block);
            if (okSet.contains(userInBlock)) {
                continue;
            }
            auto &bm = ComputeBlockIdManager::getInstance();
            if (bm.getBlockIdByOp(userInBlock) == -1) {
                dfs.clear();
                if (dfs(userInBlock)) {
                    return true;
                }
                continue;
            }
            auto opsUsedBlockId = bm.getOpsByBlockId(bm.getBlockIdByOp(userInBlock));
            for (auto *userOp : opsUsedBlockId) {
                dfs.clear();
                if (dfs(userOp)) {
                    return true;
                }
            }
        }
    }
    return false;
}

/**
 * Checks if an operation is eligible to be added to a Cube group.
 * Eligibility depends on it being a cube op, not yet assigned, not a compute op,
 * and not creating a cycle in the dependence graph.
 */
bool SeedRegionPlanner::isEligible(Operation *op)
{
    if (!isCubeOp(op) || assigned.contains(op) || isMatmulOp(op)) {
        return false;
    }
    return !willCreateCycle(op);
}

bool SeedRegionPlanner::tryAddToGroup(Operation *op)
{
    if (!op || llvm::is_contained(group, op) || op->getBlock() != block || !isEligible(op)) {
        return false;
    }
    group.push_back(op);
    return true;
}

void SeedRegionPlanner::addSourcesToGroup()
{
    size_t head = 0;
    while (head < group.size()) {
        Operation *currOp = group[head++];

        // Check data operands
        for (Value iop : currOp->getOperands()) {
            if (auto *def = iop.getDefiningOp()) {
                tryAddToGroup(def);
            }
            // Check loop-carried dependencies (SCF ForOp iter_args)
            if (auto barg = dyn_cast<BlockArgument>(iop)) {
                if (barg.getOwner() == block && isa<scf::ForOp>(block->getParentOp()) && barg.getArgNumber() > 0) {
                    auto *yieldOp = barg.getOwner()->getTerminator();
                    if (auto *yieldedValDef = yieldOp->getOperand(barg.getArgNumber() - 1).getDefiningOp()) {
                        tryAddToGroup(yieldedValDef);
                    }
                }
            }
        }

        // Check memory dependencies (RAW/WAW/WAR)
        for (auto *def : memGraph.getMemDefs(currOp)) {
            tryAddToGroup(def);
        }
    }
}

void SeedRegionPlanner::addUsersToGroup()
{
    llvm::SmallVector<Operation *> queue {seed};
    llvm::DenseSet<Operation *> forwardVisited;
    forwardVisited.insert(seed);

    unsigned qIdx = 0;
    while (qIdx < queue.size()) {
        Operation *currOp = queue[qIdx++];
        SmallVector<Operation *> allUsers;
        for (auto *u : currOp->getUsers()) {
            allUsers.push_back(u);
        }

        for (auto *u : memGraph.getMemUsers(currOp)) {
            allUsers.push_back(u);
        }

        for (auto *userOp : allUsers) {
            auto *userInBlock = getAncestorInBlock(userOp, block);
            if (tryAddToGroup(userInBlock)) {
                queue.push_back(userInBlock);
            }
        }
    }
}

/**
 * Performs a BFS to expand a group from a seed operation (usually a Dot/Compute op).
 * It explores both operands (backward) and users (forward).
 */
void SeedRegionPlanner::run()
{
    addSourcesToGroup();
    addUsersToGroup();
}

namespace {

/**
 * Processes remaining unassigned cube operations by following the topological order.
 */
class TopologicalPartitionPlanner {
    Block *block;
    unsigned nonAssignedCubeCnt = 0;
    llvm::DenseMap<Operation *, int> indegree;
    llvm::DenseSet<Operation *> &assigned;
    const MemoryDependenceGraph &memGraph;
    llvm::DenseSet<Operation *> newassigned;
    llvm::DenseSet<Operation *> bypassVisited;
    std::queue<Operation *> queue;

    void removeNonCubeOpsRecursively(Operation *op);
    llvm::LogicalResult removeReadyNonCubeOps();
    bool shouldSkip(Operation *op) { return !isCubeOp(op) || assigned.contains(op); };
    bool canExpandTo(Operation *op);
    void dumpQueueAndIndegreeInfo();
    llvm::LogicalResult populateQueueWithReadyOps();
    llvm::SmallVector<Operation *> createNewGroupFromQueue();

public:
    TopologicalPartitionPlanner(Block *block,
                                llvm::DenseSet<Operation *> &assigned,
                                const MemoryDependenceGraph &memGraph)
        : block(block), assigned(assigned), memGraph(memGraph)
    {
        initializeIndegreeForBlock(block, indegree, memGraph);

        block->walk([&](Operation *op) {
            if (op->getBlock() == block && isCubeOp(op) && !assigned.contains(op)) {
                nonAssignedCubeCnt++;
            }
        });
    }

    llvm::LogicalResult run();
};

} // namespace

// Recursively bypass non-cube ops: decrement indegree of users; collect newly-exposed cube ops
void TopologicalPartitionPlanner::removeNonCubeOpsRecursively(Operation *op)
{
    LOG_DEBUG("\tRemoved non-cube:" << *op << "\n");
    bypassVisited.insert(op);
    auto *block = op->getBlock();
    SmallVector<Operation *> allusers;
    allusers.append(op->getUsers().begin(), op->getUsers().end());
    auto &bm = ComputeBlockIdManager::getInstance();
    for (auto *memUser : memGraph.getExecAfter(op)) {
        allusers.push_back(memUser);
    }
    for (auto *user : allusers) {
        auto *userInBlock = getAncestorInBlock(user, block);
        if (!userInBlock || !indegree.contains(userInBlock) ||
            ComputeBlockIdManager::getInstance().isSameBlock(userInBlock, op)) {
            continue;
        }
        LOG_DEBUG("Sub indegree to " << *userInBlock << " from " << *op << "new degree =  " << indegree[userInBlock] - 1
                                     << "\n");
        indegree[userInBlock]--;
        if (!bm.isWholeCubeReady(userInBlock, indegree) || bypassVisited.contains(userInBlock) ||
            !shouldSkip(userInBlock)) {
            continue;
        }
        auto &bm = ComputeBlockIdManager::getInstance();
        auto blockId = bm.getBlockIdByOp(userInBlock);
        if (blockId == -1) {
            removeNonCubeOpsRecursively(userInBlock);
            continue;
        }
        for (auto *passop : bm.getOpsByBlockId(blockId)) {
            if (!bypassVisited.contains(passop)) {
                removeNonCubeOpsRecursively(passop);
            }
        }
    }
}

static bool mapsAreDiff(const llvm::DenseMap<Operation *, int> &a, const llvm::DenseMap<Operation *, int> &b)
{
    if (a.size() != b.size()) {
        return true;
    }
    return llvm::any_of(a, [&b](std::pair<Operation *, int> aIter) {
        auto bIter = b.find(aIter.first);
        return bIter == b.end() || bIter->second != aIter.second;
    });
}

/**
 * Logic to bypass non-cube operations that are ready to be executed.
 * This unblocks downstream cube operations in the topological sort.
 */
llvm::LogicalResult TopologicalPartitionPlanner::removeReadyNonCubeOps()
{
    auto &bm = ComputeBlockIdManager::getInstance();
    auto indegreeBefore = indegree;
    size_t beforeVisitedSize = bypassVisited.size();
    for (auto &p : indegree) {
        Operation *op = p.first;
        if (shouldSkip(op) && bm.isWholeCubeReady(op, indegree) && !bypassVisited.contains(op)) {
            int blockId = bm.getBlockIdByOp(op);
            if (blockId == -1) {
                removeNonCubeOpsRecursively(op);
            } else {
                for (auto *passOp : bm.getOpsByBlockId(blockId)) {
                    if (!bypassVisited.contains(passOp)) {
                        removeNonCubeOpsRecursively(passOp);
                    }
                }
            }
        }
    }
    if (!mapsAreDiff(indegreeBefore, indegree) && beforeVisitedSize == bypassVisited.size()) {
        if (Operation *parentOp = block->getParentOp()) {
            parentOp->emitError("PlanCubeBlock cannot make progress while scheduling cube operations");
        }
        dumpQueueAndIndegreeInfo();
        return llvm::failure();
    }
    return llvm::success();
}

// Expansion condition: op must be CUBE_ONLY, indegree == 0 and all its dependency ops are CUBE_ONLY
bool TopologicalPartitionPlanner::canExpandTo(Operation *op)
{
    if (!isCubeOp(op) || assigned.contains(op)) {
        return false;
    }
    auto it = indegree.find(op);
    return it->second == 0;
}

// Encountered error. Need to print failure reason, so no need for LLVM_DEBUG
void TopologicalPartitionPlanner::dumpQueueAndIndegreeInfo()
{
    // simply print a debug header in a new line
    auto errs = []() -> llvm::raw_ostream& {
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
        if (!op || op->getBlock() != block || !CVPipeline::isCubeOp(op) || assigned.contains(op) ||
            newassigned.contains(op)) {
            continue;
        }
        foundRemainingCube = true;
        llvm::errs() << "  indegree=" << p.second << " op=" << *op << "\n";
    }
    if (!foundRemainingCube) {
        llvm::errs() << "  <none>\n";
    }
}

llvm::LogicalResult TopologicalPartitionPlanner::populateQueueWithReadyOps()
{
    for (auto [op, indegree] : indegree) {
        if (indegree < 0) {
            op->emitError("Indegree cannot be negative");
            return llvm::failure();
        }
        if (indegree == 0 && !newassigned.contains(op) && isCubeOp(op) && !assigned.contains(op)) {
            queue.push(op);
        }
    }
    return llvm::success();
}

llvm::SmallVector<Operation *> TopologicalPartitionPlanner::createNewGroupFromQueue()
{
    llvm::SmallVector<Operation *> group;
    while (!queue.empty()) {
        auto *currOp = queue.front();
        queue.pop();

        newassigned.insert(currOp);
        group.push_back(currOp);
        nonAssignedCubeCnt--;

        for (auto *user : llvm::concat<Operation *>(currOp->getUsers(), memGraph.getExecAfter(currOp))) {
            auto *userInBlock = getAncestorInBlock(user, block);
            if (userInBlock && !newassigned.contains(userInBlock)) {
                auto &userInDegree = indegree[userInBlock];
                userInDegree--;
                LOG_DEBUG("Sub indegree to " << *userInBlock << " from " << *currOp << "new degree = " << userInDegree
                                             << "\n");
                if (canExpandTo(userInBlock)) {
                    queue.push(userInBlock);
                }
            }
        }
    }
    return group;
}

llvm::LogicalResult TopologicalPartitionPlanner::run()
{
    auto &bm = ComputeBlockIdManager::getInstance();
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
        if (llvm::failed(bm.markOpsWithNewId(group))) {
            return llvm::failure();
        }
    }

    return llvm::success();
}

static SmallVector<Operation *> collectMatmulOps(Block *block)
{
    SmallVector<Operation *> computeOps;
    for (Operation &op : *block) {
        if (isMatmulOp(&op)) {
            computeOps.push_back(&op);
        }
    }
    return computeOps;
}

/**
 * Main entry point: Process a single block by grouping operations into
 * execution blocks using BFS and topological traversal.
 */
static llvm::LogicalResult processBlockWithCubeBFS(Block *block, const MemoryDependenceGraph &memGraph)
{
    llvm::DenseSet<Operation *> assigned;
    auto allDots = collectMatmulOps(block);

    // Phase 1: Add helper ops (transpose, load/store, ptr etc.) to cube block of related matmul
    for (auto *dot : allDots) {
        if (assigned.contains(dot)) {
            continue;
        }

        llvm::SmallVector<Operation *> newGroup;
        SeedRegionPlanner regionPlanner {dot, block, memGraph, assigned, newGroup};
        regionPlanner.run();

        for (auto *op : newGroup) {
            assigned.insert(op);
        }
        if (llvm::failed(ComputeBlockIdManager::getInstance().markOpsWithNewId(newGroup))) {
          return llvm::failure();
        }
    }

    // Phase 2: Handle remaining Cube Ops following Topo order
    TopologicalPartitionPlanner topoPlanner {block, assigned, memGraph};
    return topoPlanner.run();
}

void mlir::triton::PlanCubeBlockPass::runOnOperation()
{
    LOG_DEBUG("\n--- Step 2: Partitioning compute blocks for cube operations --->\n");
    auto moduleOp = getOperation();
    auto &aa = getAnalysis<AliasAnalysis>();
    auto memGraph = MemoryDependenceGraph(moduleOp, aa);

    // We do not need to skip linalg blocks since they do not have core types and do not contain matmul
    auto result = moduleOp.walk([&](Block *block) {
      if (llvm::failed(processBlockWithCubeBFS(block, memGraph))) {
        return WalkResult::interrupt();
      }
      return WalkResult::advance();
    });
    if (result.wasInterrupted()) {
      signalPassFailure();
    }
    LOG_DEBUG("\n--- Step 2: end --->\n");
}

std::unique_ptr<OperationPass<ModuleOp>> mlir::triton::createPlanCubeBlockPass()
{
    return std::make_unique<PlanCubeBlockPass>();
}
