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

#include <string_view>
#include <utility>

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/iterator.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/LogicalResult.h"
#include "llvm/Support/raw_ostream.h"

#include "mlir/Analysis/AliasAnalysis.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/Block.h"
#include "mlir/IR/Operation.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Support/WalkResult.h"

#include "ascend/include/DynamicCVPipeline/Common/MemoryEffectsTracker.h"
#include "ascend/include/DynamicCVPipeline/PlanComputeBlock/Common.h"
#include "ascend/include/DynamicCVPipeline/PlanComputeBlock/ReorderOpsByBlockId.h"

#include "DynamicCVPipeline/Common/Utils.h"
#include "DynamicCVPipeline/PlanComputeBlock/ComputeBlockIdManager.h"
#include "TritonToUnstructure/OffsetAnalysis.h"
#include "triton/Dialect/Triton/IR/Dialect.h"

using namespace mlir;
static constexpr const char *DEBUG_TYPE = "ReorderOpsByBlockIdPass";
#define LOG_DEBUG(...) LLVM_DEBUG(llvm::dbgs() << " [" << DEBUG_TYPE << "] " << __VA_ARGS__)

using namespace triton;
using namespace CVPipeline;

namespace {

// A dependency DAG of both SSA and memory of the ops
struct BlockOpGraph {
    Block *block;
    ArrayRef<Operation *> ops;
    DenseMap<Operation *, unsigned> opIndex;               // op → position in ops
    DenseMap<Operation *, SmallVector<Operation *>> preds; // op → its defs
    DenseMap<Operation *, SmallVector<Operation *>> succs; // op → its uses
    BlockOpGraph(ArrayRef<Operation *> allOps, Block *block, const MemoryDependenceGraph &memGraph);
};

// Helper class to manage edges in OpGraph, mainly to reduce congitive complexity of the build function
struct EdgeHelper {
    BlockOpGraph &graph;
    DenseSet<std::pair<Operation *, Operation *>> seen;
    Block *block;

    // find the ancestor directly in the block, and in opIndex; return nullptr if either fails
    Operation *resolveToBlockOp(Operation *op);

    template <bool IsMemory = false> void addEdge(Operation *pred, Operation *succ);

    template <bool IsMemory = false> void addEdgeToUser(Operation *op, Operation *user)
    {
        if (graph.opIndex.contains(user)) {
            return; // same-level use, already covered by the def-side loop
        }
        Operation *ancestor = resolveToBlockOp(user);
        addEdge<IsMemory>(op, ancestor);
    };

    EdgeHelper(BlockOpGraph &g, Block *block) : graph(g), block(block) {};
};

} // namespace

Operation *EdgeHelper::resolveToBlockOp(Operation *op)
{
    if (graph.opIndex.contains(op)) {
        return op;
    }
    Operation *ancestor = getAncestorInBlock(op, block);
    if (!ancestor || !graph.opIndex.contains(ancestor)) {
        return nullptr;
    }
    return ancestor;
}

template <bool IsMemory> void EdgeHelper::addEdge(Operation *pred, Operation *succ)
{
    if (!pred || !succ || pred == succ) {
        return;
    }
    if (seen.insert({pred, succ}).second) {
        LOG_DEBUG("Adding " << (IsMemory ? "memory " : "") << "edge from " << *pred << " to " << *succ << "\n");
        graph.succs[pred].push_back(succ);
        graph.preds[succ].push_back(pred);
    }
};

BlockOpGraph::BlockOpGraph(ArrayRef<Operation *> allOps, Block *block, const MemoryDependenceGraph &memGraph)
    : block(block), ops(allOps)
{
    for (unsigned i = 0; i < allOps.size(); ++i) {
        opIndex[allOps[i]] = i;
        preds[allOps[i]]; // ensure every node has an entry
        succs[allOps[i]];
    }

    EdgeHelper edges(*this, block);

    for (Operation *op : allOps) {
        LOG_DEBUG("Processing op: " << *op << "\n");
        // Edges from operand defs (including defs nested inside other ops).
        for (Value const operand : op->getOperands()) {
            Operation *defOp = operand.getDefiningOp();
            if (!defOp) {
                continue;
            }
            Operation *def = edges.resolveToBlockOp(defOp);
            edges.addEdge(def, op);
        }

        // Edges from uses that live inside nested regions of another block-level op.
        for (Value const result : op->getResults()) {
            for (Operation *user : result.getUsers()) {
                edges.addEdgeToUser(op, user);
            }
        }

        for (auto *memDef : memGraph.getExecBefore(op)) {
            Operation *def = edges.resolveToBlockOp(memDef);
            edges.addEdge<true>(def, op);
        }

        for (auto *memUser : memGraph.getExecAfter(op)) {
            edges.addEdgeToUser<true>(op, memUser);
        }
    }
}

static llvm::FailureOr<DenseMap<Operation *, int>> collectBlockIds(ArrayRef<Operation *> allOps)
{
    DenseMap<Operation *, int> opBlockId;
    for (Operation *op : allOps) {
        if (llvm::failed(verifyOpBlockId(op))) {
            return llvm::failure();
        }
        auto blockIdAttrRes = getOpBlockId(op);
        int64_t blockId =
            blockIdAttrRes.has_value() ? blockIdAttrRes.value() : ComputeBlockIdManager::getInstance().getNextId();
        opBlockId[op] = blockId;
    }
    return opBlockId;
}

namespace {

// Helper structure to hold the group-level graph data.
struct GroupAdjacencyGraph {
    Block *block;
    SmallVector<int> groupIds;
    SmallVector<SmallVector<unsigned>> succs;
    SmallVector<unsigned> inDeg;
    GroupAdjacencyGraph(const BlockOpGraph &g, const DenseMap<Operation *, int> &opBlockId);
    llvm::FailureOr<SmallVector<int>> computeTopologicalOrder();
};

} // namespace

/**
 * Step 1: Build the group-level dependency graph from operator-level edges.
 * Maps individual operations to their respective groups and identifies
 * dependencies between those groups.
 */
GroupAdjacencyGraph::GroupAdjacencyGraph(const BlockOpGraph &g, const DenseMap<Operation *, int> &opBlockId)
    : block(g.block)
{
    // 1. Collect distinct group IDs while preserving the first-appearance order.
    DenseSet<int> seenIds;
    for (Operation *op : g.ops) {
        int id = opBlockId.at(op);
        if (seenIds.insert(id).second) {
            groupIds.push_back(id);
        }
    }

    unsigned n = groupIds.size();
    succs.resize(n);
    inDeg.assign(n, 0);

    // Map group ID to its index in the groupIds vector for fast lookup.
    DenseMap<int, unsigned> groupPos;
    for (unsigned i = 0; i < n; ++i) {
        groupPos[groupIds[i]] = i;
    }

    // 2. Build group-level edges. Use a set to avoid duplicate edges between groups.
    DenseSet<std::pair<unsigned, unsigned>> addedEdges;
    for (Operation *op : g.ops) {
        unsigned fromIdx = groupPos[opBlockId.at(op)];

        for (Operation *succ : g.succs.at(op)) {
            unsigned toIdx = groupPos[opBlockId.at(succ)];
            // Ignore intra-group dependencies and duplicate inter-group edges.
            if (fromIdx != toIdx && addedEdges.insert({fromIdx, toIdx}).second) {
                succs[fromIdx].push_back(toIdx);
                inDeg[toIdx]++;
            }
        }
    }

    // Logging the constructed group graph.
    LOG_DEBUG("Group-level edges:\n");
    for (unsigned i = 0; i < n; ++i) {
        LOG_DEBUG("  Group " << groupIds[i] << " -> ");
        for (unsigned succIdx : succs[i]) {
            LOG_DEBUG(groupIds[succIdx] << " ");
        }
        LOG_DEBUG("\n");
    }
}

/**
 * Step 2: Perform a topological sort (Kahn's Algorithm) on the group graph.
 * Returns the group IDs in an order that satisfies all dependencies.
 */
llvm::FailureOr<SmallVector<int>> GroupAdjacencyGraph::computeTopologicalOrder()
{
    SmallVector<int> result;
    SmallVector<unsigned> ready; // Nodes with in-degree 0.
    unsigned n = groupIds.size();

    for (unsigned i = 0; i < n; ++i) {
        if (inDeg[i] == 0) {
            ready.push_back(i);
        }
    }

    while (!ready.empty()) {
        auto cur = ready.pop_back_val();

        result.push_back(groupIds[cur]);

        for (unsigned succIdx : succs[cur]) {
            if (--inDeg[succIdx] == 0) {
                ready.push_back(succIdx);
            }
        }
    }

    LOG_DEBUG("Group order: ");
    for (int id : result) {
        LOG_DEBUG(id << " ");
    }
    LOG_DEBUG("\n");

    if (result.size() == n) {
        return result;
    }
    Operation *op = block->getParentOp();
    constexpr std::string_view kErrorPrefix = "Failed to compute topological order for ";
    if (!op) {
        llvm::errs() << kErrorPrefix << "an unknown block that is not contained in an op";
        return llvm::failure();
    }
    size_t regionIdx = 0;
    bool found = false;
    for (auto [i, region] : llvm::enumerate(op->getRegions())) {
        for (auto &possibleBlock : region.getBlocks()) {
            if (&possibleBlock == block) {
                regionIdx = i;
            }
        }
    }
    op->emitError(kErrorPrefix) << "block in region " << regionIdx;
    return llvm::failure();
}

// Stable sort ops based on their group orders
static llvm::FailureOr<SmallVector<Operation *>> buildReorderedOps(const BlockOpGraph &graph,
                                                                   const DenseMap<Operation *, int> &opBlockId)
{
    SmallVector<Operation *> reordered;
    GroupAdjacencyGraph adjacencyGraph {graph, opBlockId};
    auto groupOrderResult = adjacencyGraph.computeTopologicalOrder();
    if (llvm::failed(groupOrderResult)) {
        return llvm::failure();
    }

    for (int const blockId : groupOrderResult.value()) {
        for (Operation *op : graph.ops) {
            if (opBlockId.at(op) == blockId) {
                reordered.push_back(op);
            }
        }
    }
    return reordered;
}

// Reorder the ops in the mlir representation
static void applyReorder(Block &block, ArrayRef<Operation *> reordered)
{
    Operation *terminator = block.mightHaveTerminator() ? block.getTerminator() : nullptr;
    for (Operation *op : reordered) {
        op->moveBefore(&block, block.end());
    }

    if (terminator) {
        terminator->moveBefore(&block, block.end());
    }
}

static llvm::LogicalResult reorderOpsInBlock(Block &block, const MemoryDependenceGraph &memGraph)
{
    const auto allOps = llvm::to_vector(llvm::make_pointer_range(block.without_terminator()));

    const BlockOpGraph graph {allOps, &block, memGraph};
    llvm::FailureOr<DenseMap<Operation *, int>> opBlockIdOpt = collectBlockIds(allOps);
    if (failed(opBlockIdOpt)) {
        return failure();
    }

    auto &opBlockId = *opBlockIdOpt;
    LOG_DEBUG("Initial opBlockIds:\n");
    for (Operation *op : allOps) {
        LOG_DEBUG("  Op: " << *op << ", opBlockId = " << opBlockId[op] << "\n");
    }

    const auto reorderedRes = buildReorderedOps(graph, opBlockId);
    if (failed(reorderedRes)) {
        return failure();
    }

    applyReorder(block, reorderedRes.value());

    return llvm::success();
}

void ReorderOpsByBlockIdPass::runOnOperation()
{
    LOG_DEBUG("\n=== Pass: TuningOpSeq ===\n");
    OpBuilder const builder(&getContext());

    auto moduleOp = getOperation();
    auto &aa = getAnalysis<AliasAnalysis>();
    auto memGraph = MemoryDependenceGraph(moduleOp, aa);
    moduleOp.walk([&](Block *block) {
        auto *parentOp = block->getParentOp();
        if (!parentOp ||
            // whitelist ops to reorder
            !(isa<func::FuncOp>(parentOp) || isa<scf::SCFDialect>(parentOp->getDialect()))) {
            return WalkResult::skip();
        }
        if (llvm::failed(reorderOpsInBlock(*block, memGraph))) {
            signalPassFailure();
        }
        return WalkResult::advance();
    });

    LOG_DEBUG("=== Pass TuningOpSeq complete ===\n");
    ComputeBlockIdManager::getInstance().reset();
}

std::unique_ptr<OperationPass<ModuleOp>> mlir::triton::createReorderOpsByBlockIdPass()
{
    return std::make_unique<ReorderOpsByBlockIdPass>();
}
