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

#include "ascend/include/DynamicCVPipeline/SeparateMemoryFromCompute/AddMultiBufferToGMLoadInternal.h"

namespace gmload {

// ============================================================================
// Analysis: dependency chain computation
// ============================================================================

/// Given an op that may be nested inside a region (e.g., inside scf.if),
/// walk up the parent chain and return the ancestor op that lives directly
/// in `scopeBlock`. Returns nullptr if no such ancestor exists.
Operation *getAncestorInBlock(Operation *nestedOp, Block *scopeBlock)
{
    while (nestedOp) {
        if (nestedOp->getBlock() == scopeBlock)
            return nestedOp;
        nestedOp = nestedOp->getParentOp();
    }
    return nullptr;
}

/// Add `op` and all its transitive SSA operand-producers within `scopeBlock`
/// to `visited`. Newly discovered ops are also appended to `worklist`.
void backwardTrace(Operation *sourceOp, Block *scopeBlock, llvm::SetVector<Operation *> &visited,
                   SmallVectorImpl<Operation *> &worklist)
{
    for (Value operand : sourceOp->getOperands()) {
        auto *defOp = operand.getDefiningOp();
        if (!defOp || defOp->getBlock() != scopeBlock)
            continue;
        if (visited.insert(defOp))
            worklist.push_back(defOp);
    }
}

/// Process worklist with backward SSA trace until empty.
/// Used in Phase 1 and Phase 3 of computeLoadChain.
void backwardTraceFromWorklist(llvm::SetVector<Operation *> &visited, SmallVectorImpl<Operation *> &worklist,
                               Block *scopeBlock)
{
    while (!worklist.empty()) {
        Operation *currentOp = worklist.pop_back_val();
        backwardTrace(currentOp, scopeBlock, visited, worklist);
    }
}

/// Forward trace from every memref::AllocOp in visited set.
/// Returns newly discovered ops that use alloc results.
/// Used in Phase 2 of computeLoadChain.
SmallVector<Operation *> forwardTraceFromAllocs(llvm::SetVector<Operation *> &visited, Block *scopeBlock)
{
    SmallVector<Value> valueWorklist;
    for (Operation *currentOp : visited) {
        if (isa<memref::AllocOp>(currentOp)) {
            for (Value result : currentOp->getResults())
                valueWorklist.push_back(result);
        }
    }

    SmallVector<Operation *> newlyDiscoveredOps;
    while (!valueWorklist.empty()) {
        Value currentValue = valueWorklist.pop_back_val();
        for (Operation *user : currentValue.getUsers()) {
            Operation *ancestor = getAncestorInBlock(user, scopeBlock);
            if (!ancestor)
                continue;
            if (!visited.insert(ancestor))
                continue;
            newlyDiscoveredOps.push_back(ancestor);
            for (Value result : ancestor->getResults())
                valueWorklist.push_back(result);
        }
    }
    return newlyDiscoveredOps;
}

/// Capture free variables used in regions of ops in visited set.
/// Performs backward trace for each discovered free variable definition.
/// Used in Phase 4 of computeLoadChain.
void captureRegionFreeVars(llvm::SetVector<Operation *> &visited, SmallVectorImpl<Operation *> &worklist,
                           Block *scopeBlock)
{
    SmallVector<Operation *> freeVarWorklist;
    SmallVector<Operation *> initialChain(visited.begin(), visited.end());

    // First pass: collect free variables from all region-containing ops
    for (Operation *chainOp : initialChain) {
        if (chainOp->getNumRegions() == 0)
            continue;
        chainOp->walk([&](Operation *nested) {
            if (nested == chainOp)
                return;
            for (Value operand : nested->getOperands()) {
                auto *defOp = operand.getDefiningOp();
                if (!defOp || defOp->getBlock() != scopeBlock)
                    continue;
                if (visited.insert(defOp))
                    freeVarWorklist.push_back(defOp);
            }
        });
    }

    // Second pass: backward trace and recursively capture nested region free vars
    worklist.assign(freeVarWorklist.begin(), freeVarWorklist.end());
    while (!worklist.empty()) {
        Operation *currentOp = worklist.pop_back_val();
        backwardTrace(currentOp, scopeBlock, visited, worklist);
        if (currentOp->getNumRegions() == 0)
            continue;
        currentOp->walk([&](Operation *nested) {
            if (nested == currentOp)
                return;
            for (Value operand : nested->getOperands()) {
                auto *defOp = operand.getDefiningOp();
                if (!defOp || defOp->getBlock() != scopeBlock)
                    continue;
                if (visited.insert(defOp)) {
                    freeVarWorklist.push_back(defOp);
                    worklist.push_back(defOp);
                }
            }
        });
    }
}

/// Compute the full dependency chain of `markedOp` within `scopeBlock`.
///   Phase 1  Backward SSA trace from markedOp.
///   Phase 2  Forward trace from every memref::AllocOp found in Phase 1.
///   Phase 3  Backward SSA trace from newly discovered ops.
///   Phase 4  Capture free variables of chain-op regions.
SmallVector<Operation *> computeLoadChain(Operation *markedOp, Block *scopeBlock)
{
    llvm::SetVector<Operation *> visited;
    SmallVector<Operation *> worklist;

    // Phase 1: backward SSA trace from markedOp
    visited.insert(markedOp);
    worklist.push_back(markedOp);
    backwardTraceFromWorklist(visited, worklist, scopeBlock);

    // Phase 2: forward trace from every memref.alloc in the chain
    SmallVector<Operation *> newlyDiscoveredOps = forwardTraceFromAllocs(visited, scopeBlock);

    // Phase 3: backward SSA trace from newly discovered ops
    worklist.assign(newlyDiscoveredOps.begin(), newlyDiscoveredOps.end());
    backwardTraceFromWorklist(visited, worklist, scopeBlock);

    // Phase 4: capture free variables of chain-op regions
    captureRegionFreeVars(visited, worklist, scopeBlock);

    // Sort by block position (topological order)
    SmallVector<Operation *> sorted(visited.begin(), visited.end());
    llvm::sort(sorted, [](Operation *leftOp, Operation *rightOp) { return leftOp->isBeforeInBlock(rightOp); });
    return sorted;
}

/// Find the memref::AllocOp in the chain that backs the marked op.
memref::AllocOp findBackingAlloc(Operation *markedOp, ArrayRef<Operation *> chain)
{
    for (Operation *chainOp : chain) {
        if (auto alloc = dyn_cast<memref::AllocOp>(chainOp)) {
            if (markedOp->getNumOperands() > 0 && markedOp->getOperand(0).getDefiningOp() == alloc)
                return alloc;
        }
    }
    for (Operation *chainOp : chain) {
        if (auto alloc = dyn_cast<memref::AllocOp>(chainOp))
            return alloc;
    }
    return nullptr;
}

// ============================================================================
// Step 1a: Generic marked-op collection (scan all IR)
// ============================================================================

/// Scan the entire module IR for ops carrying the `kBufferableMarker`
/// attribute and compute the dependency chain for each.
SmallVector<MarkedLoad> collectMarkedOps(ModuleOp module)
{
    SmallVector<MarkedLoad> results;

    module->walk([&](Operation *op) {
        if (!op->hasAttr(kBufferableMarker))
            return;
        if (op->getNumResults() == 0)
            return;

        Block *scope = op->getBlock();
        auto chain = computeLoadChain(op, scope);

        auto alloc = findBackingAlloc(op, chain);
        if (!alloc)
            return;

        results.push_back({op, std::move(chain), alloc});
    });

    return results;
}

// ============================================================================
// Step 1b: Group by enclosing loop construct
// ============================================================================

/// Try to determine the loop trip count at compile time.
std::optional<int64_t> getConstantTripCount(scf::ForOp forOp)
{
    auto lowerBoundConst = forOp.getLowerBound().getDefiningOp<arith::ConstantOp>();
    auto upperBoundConst = forOp.getUpperBound().getDefiningOp<arith::ConstantOp>();
    auto stepConst = forOp.getStep().getDefiningOp<arith::ConstantOp>();
    if (!lowerBoundConst || !upperBoundConst || !stepConst)
        return std::nullopt;

    auto lowerBoundAttr = dyn_cast<IntegerAttr>(lowerBoundConst.getValue());
    auto upperBoundAttr = dyn_cast<IntegerAttr>(upperBoundConst.getValue());
    auto stepAttr = dyn_cast<IntegerAttr>(stepConst.getValue());
    if (!lowerBoundAttr || !upperBoundAttr || !stepAttr)
        return std::nullopt;

    int64_t lowerBoundValue = lowerBoundAttr.getInt();
    int64_t upperBoundValue = upperBoundAttr.getInt();
    int64_t stepValue = stepAttr.getInt();
    if (stepValue < 1)
        return std::nullopt;

    // stepValue >= 1 guaranteed by check above
    int64_t tripCount = (upperBoundValue - lowerBoundValue + stepValue - 1) / stepValue;
    return tripCount > 0 ? std::optional<int64_t>(tripCount) : std::nullopt;
}

/// Group marked loads by their nearest enclosing scf::ForOp.
/// Each marked load becomes its own independent LoadGroup so that every GM
/// load has its own producer flags, counters, and slot targets.  Shared
/// address-computation ops are handled correctly by the consumer skip analysis
/// without forcing multiple loads into the same multi-buffer state machine.
void groupByEnclosingForOp(SmallVector<MarkedLoad> &markedOps, SmallVectorImpl<ForBufferCtx> &contexts)
{
    // Bucket loads by the nearest enclosing for loop.
    llvm::DenseMap<Operation *, SmallVector<size_t>> forBuckets;
    SmallVector<scf::ForOp> forOpsInOrder;
    for (size_t markedIndex = 0; markedIndex < markedOps.size(); ++markedIndex) {
        auto forOp = markedOps[markedIndex].markedOp->getParentOfType<scf::ForOp>();
        if (!forOp)
            continue;
        auto [bucketIt, inserted] = forBuckets.try_emplace(forOp.getOperation());
        if (inserted)
            forOpsInOrder.push_back(forOp);
        bucketIt->second.push_back(markedIndex);
    }

    // Each marked load becomes its own independent LoadGroup.
    for (auto forOp : forOpsInOrder) {
        auto &indices = forBuckets[forOp.getOperation()];

        ForBufferCtx context;
        context.forOp = forOp;
        for (size_t markedIndex : indices) {
            LoadGroup group;
            group.loads.push_back(std::move(markedOps[markedIndex]));
            if (!group.loads.empty())
                group.mergedChain = group.loads[0].chain;
            context.groups.push_back(std::move(group));
        }
        contexts.push_back(std::move(context));
    }
}

// ============================================================================
// Consumer skip analysis
// ============================================================================

/// Collect all chain ops and marked ops from the given loads.
void collectChainAndMarkedOps(const SmallVector<MarkedLoad> &loads, llvm::DenseSet<Operation *> &allChainOps,
                              llvm::DenseSet<Operation *> &markedOps)
{
    for (const auto &markedLoad : loads) {
        for (Operation *chainOp : markedLoad.chain)
            allChainOps.insert(chainOp);
        markedOps.insert(markedLoad.markedOp);
    }
}

/// Check if a result value is used by yield or by non-chain ops in forBody.
bool isResultUsedExternally(Value result, Operation *yieldOp, const llvm::DenseSet<Operation *> &allChainOps,
                            Block *forBody)
{
    // Check if result is yielded
    for (Value yieldedValue : yieldOp->getOperands()) {
        if (yieldedValue == result)
            return true;
    }
    // Check if result is used by non-chain ops
    for (Operation *user : result.getUsers()) {
        Operation *ancestor = getAncestorInBlock(user, forBody);
        if (ancestor && !allChainOps.contains(ancestor))
            return true;
    }
    return false;
}

/// Collect all operands from an op, including those from nested regions.
SmallVector<Value, kInitialRefCapacity> collectOperandsIncludingRegions(Operation *op)
{
    SmallVector<Value, kInitialRefCapacity> refs;
    for (Value operand : op->getOperands())
        refs.push_back(operand);
    if (op->getNumRegions() > 0) {
        op->walk([&](Operation *nested) {
            if (nested == op)
                return;
            for (Value operand : nested->getOperands())
                refs.push_back(operand);
        });
    }
    return refs;
}

/// Return the set of ops that must be skipped (not cloned) in the consumer
llvm::DenseSet<Operation *> computeSkipInConsumer(const SmallVector<MarkedLoad> &loads, Block *forBody)
{
    llvm::DenseSet<Operation *> allChainOps;
    llvm::DenseSet<Operation *> markedOps;
    collectChainAndMarkedOps(loads, allChainOps, markedOps);

    // Phase 1: chain ops whose results are directly used by non-chain ops or
    // yielded. These must be cloned so non-chain consumers have valid SSA defs.
    llvm::DenseSet<Operation *> needed;
    Operation *yieldOp = forBody->getTerminator();
    for (Operation *chainOp : allChainOps) {
        if (markedOps.contains(chainOp))
            continue;
        bool isNeeded = llvm::any_of(chainOp->getResults(), [&](Value result) {
            return isResultUsedExternally(result, yieldOp, allChainOps, forBody);
        });
        if (isNeeded)
            needed.insert(chainOp);
    }

    // Phase 2: backward propagation through operands and region free variables.
    SmallVector<Operation *> worklist(needed.begin(), needed.end());
    while (!worklist.empty()) {
        Operation *currentOp = worklist.pop_back_val();
        SmallVector<Value, kInitialRefCapacity> operands = collectOperandsIncludingRegions(currentOp);
        for (Value operand : operands) {
            auto *defOp = operand.getDefiningOp();
            if (!defOp || defOp->getBlock() != forBody)
                continue;
            if (!allChainOps.contains(defOp) || markedOps.contains(defOp))
                continue;
            if (needed.insert(defOp).second)
                worklist.push_back(defOp);
        }
    }

    // Compute skip set as difference: allChainOps - needed
    llvm::DenseSet<Operation *> skip;
    for (Operation *chainOp : allChainOps) {
        if (!needed.contains(chainOp))
            skip.insert(chainOp);
    }
    return skip;
}

} // namespace gmload
