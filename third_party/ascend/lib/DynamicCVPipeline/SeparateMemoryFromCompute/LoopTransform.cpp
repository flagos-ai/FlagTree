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
// Dead iter_arg elimination
// ============================================================================

/// Return true if the iter_arg at `iterArgIdx` in `forOp` is dead
///   (a) its loop result has no uses outside the loop, AND
///   (b) its block argument is only used in a chain of side-effect-free ops
///       whose results ultimately feed only the loop yield.
bool isDeadIterArg(scf::ForOp forOp, unsigned iterArgIdx)
{
    if (!forOp.getResult(iterArgIdx).use_empty()) {
        return false;
    }

    Value blockArg = forOp.getBody()->getArgument(iterArgIdx + kForBodyIterArgOffset);
    Block *forBody = forOp.getBody();
    Operation *yieldOp = forBody->getTerminator();

    llvm::SmallSetVector<Value, kInitialRefCapacity> frontier;
    frontier.insert(blockArg);

    while (!frontier.empty()) {
        Value currentValue = frontier.pop_back_val();
        for (OpOperand &use : currentValue.getUses()) {
            Operation *user = use.getOwner();
            Operation *ancestor = getAncestorInBlock(user, forBody);
            if (!ancestor) {
                continue;
            }
            if (ancestor == yieldOp) {
                continue;
            }
            if (!mlir::isMemoryEffectFree(ancestor)) {
                return false;
            }
            for (Value result : ancestor->getResults()) {
                frontier.insert(result);
            }
        }
    }
    return true;
}

llvm::DenseSet<unsigned> collectDeadIterArgs(scf::ForOp forOp, unsigned candidateCount)
{
    llvm::DenseSet<unsigned> deadSet;
    for (unsigned iterArgIdx = 0; iterArgIdx < candidateCount; ++iterArgIdx) {
        if (isDeadIterArg(forOp, iterArgIdx)) {
            deadSet.insert(iterArgIdx);
        }
    }
    return deadSet;
}

SmallVector<Value> collectLiveIterArgInits(scf::ForOp forOp, const llvm::DenseSet<unsigned> &deadSet)
{
    unsigned numIter = forOp.getNumResults();
    SmallVector<Value> liveInits;
    liveInits.reserve(numIter - deadSet.size());
    for (unsigned iterArgIdx = 0; iterArgIdx < numIter; ++iterArgIdx) {
        if (!deadSet.contains(iterArgIdx)) {
            liveInits.push_back(forOp.getInitArgs()[iterArgIdx]);
        }
    }
    return liveInits;
}

scf::ForOp createForWithLiveIterArgs(OpBuilder &builder, scf::ForOp forOp, ArrayRef<Value> liveInits)
{
    builder.setInsertionPoint(forOp);
    auto newFor = builder.create<scf::ForOp>(forOp.getLoc(), forOp.getLowerBound(), forOp.getUpperBound(),
                                             forOp.getStep(), liveInits);
    newFor->setAttrs(forOp->getAttrs());
    return newFor;
}

IRMapping buildPrunedIterArgMapping(scf::ForOp oldForOp, scf::ForOp newForOp, const llvm::DenseSet<unsigned> &deadSet)
{
    unsigned numIter = oldForOp.getNumResults();
    Block *oldBody = oldForOp.getBody();
    Block *newBody = newForOp.getBody();

    IRMapping mapping;
    mapping.map(oldBody->getArgument(kForInductionVarArgIndex), newForOp.getInductionVar());
    unsigned liveArgIdx = 0;
    for (unsigned iterArgIdx = 0; iterArgIdx < numIter; ++iterArgIdx) {
        Value oldArg = oldBody->getArgument(iterArgIdx + kForBodyIterArgOffset);
        if (deadSet.contains(iterArgIdx)) {
            mapping.map(oldArg, oldForOp.getInitArgs()[iterArgIdx]);
            continue;
        }
        mapping.map(oldArg, newBody->getArgument(liveArgIdx++ + kForBodyIterArgOffset));
    }
    return mapping;
}

void clonePrunedForBody(OpBuilder &builder, scf::ForOp oldForOp, scf::ForOp newForOp, IRMapping &mapping)
{
    builder.setInsertionPointToStart(newForOp.getBody());
    for (auto &op : oldForOp.getBody()->without_terminator()) {
        builder.clone(op, mapping);
    }
}

SmallVector<Value> collectLiveYieldOperands(scf::ForOp forOp, const llvm::DenseSet<unsigned> &deadSet,
                                            IRMapping &mapping)
{
    unsigned numIter = forOp.getNumResults();
    auto oldYield = cast<scf::YieldOp>(forOp.getBody()->getTerminator());
    SmallVector<Value> liveYield;
    liveYield.reserve(numIter - deadSet.size());
    for (unsigned iterArgIdx = 0; iterArgIdx < numIter; ++iterArgIdx) {
        if (!deadSet.contains(iterArgIdx)) {
            liveYield.push_back(mapping.lookupOrDefault(oldYield.getOperand(iterArgIdx)));
        }
    }
    return liveYield;
}

void replaceLiveForResults(scf::ForOp oldForOp, scf::ForOp newForOp, const llvm::DenseSet<unsigned> &deadSet)
{
    unsigned numIter = oldForOp.getNumResults();
    unsigned liveArgIdx = 0;
    for (unsigned iterArgIdx = 0; iterArgIdx < numIter; ++iterArgIdx) {
        if (!deadSet.contains(iterArgIdx)) {
            oldForOp.getResult(iterArgIdx).replaceAllUsesWith(newForOp.getResult(liveArgIdx++));
        }
    }
}

/// Rebuild `forOp` dropping all iter_args that satisfy `isDeadIterArg`.
/// Only the first `candidateCount` iter_args are considered for pruning.
/// Returns the pruned ForOp, or `forOp` itself when nothing was pruned.
FailureOr<scf::ForOp> pruneDeadIterArgs(OpBuilder &builder, scf::ForOp forOp, unsigned candidateCount)
{
    unsigned numIter = forOp.getNumResults();
    if (candidateCount > numIter) {
        forOp.emitError() << "[" << DEBUG_TYPE << "] candidate iter_arg count " << candidateCount
                          << " exceeds total iter_arg count " << numIter;
        return failure();
    }

    llvm::DenseSet<unsigned> deadSet = collectDeadIterArgs(forOp, candidateCount);
    if (deadSet.empty()) {
        return forOp;
    }

    SmallVector<Value> liveInits = collectLiveIterArgInits(forOp, deadSet);
    scf::ForOp newFor = createForWithLiveIterArgs(builder, forOp, liveInits);
    IRMapping mapping = buildPrunedIterArgMapping(forOp, newFor, deadSet);

    clonePrunedForBody(builder, forOp, newFor, mapping);
    SmallVector<Value> liveYield = collectLiveYieldOperands(forOp, deadSet, mapping);
    builder.create<scf::YieldOp>(forOp.getBody()->getTerminator()->getLoc(), liveYield);

    replaceLiveForResults(forOp, newFor, deadSet);
    forOp.erase();
    return newFor;
}

/// Erase side-effect-free ops in `body` whose results are entirely unused.
/// Propagates: erasing an op may expose its operand-defining ops as newly dead.
/// Only considers ops directly in `body` (not nested regions).
void eraseDeadBodyOps(Block *body)
{
    SmallVector<Operation *> worklist;
    llvm::SmallDenseSet<Operation *, kInitialRefCapacity> queued;
    auto enqueueIfDead = [&body, &queued, &worklist](Operation *candidateOp) {
        if (!candidateOp || candidateOp->getBlock() != body || candidateOp == body->getTerminator()) {
            return;
        }
        if (!queued.insert(candidateOp).second) {
            return;
        }
        worklist.push_back(candidateOp);
    };

    for (auto &op : *body) {
        if (&op == body->getTerminator()) {
            continue;
        }
        if (mlir::isMemoryEffectFree(&op) &&
            llvm::all_of(op.getResults(), [](Value result) { return result.use_empty(); })) {
            enqueueIfDead(&op);
        }
    }

    while (!worklist.empty()) {
        Operation *currentOp = worklist.pop_back_val();
        queued.erase(currentOp);
        if (!mlir::isMemoryEffectFree(currentOp)) {
            continue;
        }
        if (!llvm::all_of(currentOp->getResults(), [](Value result) { return result.use_empty(); })) {
            continue;
        }

        SmallVector<Operation *, kInitialRefCapacity> operandDefs;
        for (Value operand : currentOp->getOperands()) {
            auto *defOp = operand.getDefiningOp();
            if (defOp && defOp->getBlock() == body) {
                operandDefs.push_back(defOp);
            }
        }
        currentOp->erase();

        for (Operation *defOp : operandDefs) {
            if (mlir::isMemoryEffectFree(defOp) &&
                llvm::all_of(defOp->getResults(), [](Value result) { return result.use_empty(); })) {
                enqueueIfDead(defOp);
            }
        }
    }
}

// ============================================================================
// Producer address projection helpers
// ============================================================================

/// Return true if `value` is defined outside `forOp` (i.e. loop-invariant).
bool isLoopInvariant(Value value, scf::ForOp forOp)
{
    if (auto blockArg = dyn_cast<BlockArgument>(value)) {
        return blockArg.getOwner() != forOp.getBody();
    }
    Operation *defOp = value.getDefiningOp();
    Operation *parent = defOp->getParentOp();
    while (parent) {
        if (parent == forOp.getOperation()) {
            return false;
        }
        parent = parent->getParentOp();
    }
    return true;
}

/// If the yield value for iter-arg `iterArg` is of the form
/// arith.addi(iterArg, delta) where `delta` is loop-invariant,
/// return true and set `delta`.
bool getLinearIterArgDelta(Value iterArg, Value yieldVal, scf::ForOp forOp, Value &delta)
{
    auto addOp = yieldVal.getDefiningOp<arith::AddIOp>();
    if (!addOp) {
        return false;
    }
    Value candidate;
    if (addOp.getLhs() == iterArg) {
        candidate = addOp.getRhs();
    } else if (addOp.getRhs() == iterArg) {
        candidate = addOp.getLhs();
    } else {
        return false;
    }
    if (!isLoopInvariant(candidate, forOp)) {
        return false;
    }
    delta = candidate;
    return true;
}

Value castIndexTo(OpBuilder &builder, Location loc, Value value, Type targetType)
{
    if (value.getType() == targetType) {
        return value;
    }
    return builder.create<arith::IndexCastOp>(loc, targetType, value);
}

Value castToIndex(OpBuilder &builder, Location loc, Value value)
{
    if (value.getType().isIndex()) {
        return value;
    }
    return builder.create<arith::IndexCastOp>(loc, builder.getIndexType(), value);
}

// ============================================================================
// Block-id tagging helper
// ============================================================================

/// Set kBlockIdAttr on `value`'s defining op if both the op and the attribute exist.
void tagWithBlockId(Value value, IntegerAttr blockId)
{
    if (blockId) {
        if (auto *defOp = value.getDefiningOp()) {
            defOp->setAttr(kBlockIdAttr, blockId);
        }
    }
}

// ============================================================================
// Core transformation helpers
// ============================================================================

/// Allocate one memref slot per (depth × load) pair before the loop.
void allocateBufferSlots(OpBuilder &builder, Location loc, scf::ForOp forOp, LoadGroup &group)
{
    builder.setInsertionPoint(forOp);
    int depth = group.depth;
    int numLoads = static_cast<int>(group.loads.size());
    group.bufSlots.resize(depth);
    for (int slotIdx = 0; slotIdx < depth; ++slotIdx) {
        group.bufSlots[slotIdx].resize(numLoads);
        for (int loadIdx = 0; loadIdx < numLoads; ++loadIdx) {
            auto newAlloc = builder.create<memref::AllocOp>(loc, group.loads[loadIdx].allocOp.getType());
            newAlloc->setAttrs(group.loads[loadIdx].allocOp.getOperation()->getAttrs());
            group.bufSlots[slotIdx][loadIdx] = newAlloc;
        }
    }
}

/// Build a new scf.for whose iter_args cover all groups:
///   [original..., (flags_g[depth], prodCounter_g, consCounter_g) for each g].
/// Returns metadata including per-group iter_arg handles.
ExtendedForInfo buildExtendedFor(OpBuilder &builder, Location loc, scf::ForOp forOp, ArrayRef<LoadGroup> groups,
                                 ConstantCache &cache)
{
    builder.setInsertionPoint(forOp);
    int numOrig = static_cast<int>(forOp.getInitArgs().size());
    int depth = groups.empty() ? 0 : groups[0].depth;

    Value falseVal = cache.getFalse(builder, loc);
    Value zeroIndex = cache.getIndex(builder, loc, 0);

    SmallVector<Value> inits;
    inits.reserve(numOrig + static_cast<int>(groups.size()) * (depth + kLoadGroupCounterIterArgCount));
    for (Value initArg : forOp.getInitArgs()) {
        inits.push_back(initArg);
    }
    for (auto &group : groups) {
        for (int slotIdx = 0; slotIdx < group.depth; ++slotIdx) {
            inits.push_back(falseVal);
        }
        inits.push_back(zeroIndex); // prodCounter
        inits.push_back(zeroIndex); // consCounter
    }

    auto newFor = builder.create<scf::ForOp>(loc, forOp.getLowerBound(), forOp.getUpperBound(), forOp.getStep(), inits);
    newFor->setAttrs(forOp->getAttrs());
    Block *oldBody = forOp.getBody();
    Block *newBody = newFor.getBody();

    IRMapping mapping;
    mapping.map(oldBody->getArgument(kForInductionVarArgIndex), newBody->getArgument(kForInductionVarArgIndex));
    for (int initArgIdx = 0; initArgIdx < numOrig; ++initArgIdx) {
        mapping.map(oldBody->getArgument(initArgIdx + kForBodyIterArgOffset),
                    newBody->getArgument(initArgIdx + kForBodyIterArgOffset));
    }

    if (!newBody->empty()) {
        newBody->getTerminator()->erase();
    }
    builder.setInsertionPointToEnd(newBody);

    SmallVector<GroupIterArgs> groupArgs;
    int argOffset = numOrig;
    for (auto &group : groups) {
        GroupIterArgs groupIterArgs;
        for (int slotIdx = 0; slotIdx < group.depth; ++slotIdx) {
            groupIterArgs.flagArgs.push_back(newBody->getArgument(kForBodyIterArgOffset + argOffset + slotIdx));
        }
        groupIterArgs.prodCounter =
            newBody->getArgument(kForBodyIterArgOffset + argOffset + group.depth + kLoadGroupProducerCounterOffset);
        groupIterArgs.consCounter =
            newBody->getArgument(kForBodyIterArgOffset + argOffset + group.depth + kLoadGroupConsumerCounterOffset);
        argOffset += group.depth + kLoadGroupCounterIterArgCount;
        groupArgs.push_back(std::move(groupIterArgs));
    }

    return {newFor, oldBody, newBody, std::move(mapping), std::move(groupArgs), falseVal, numOrig, depth};
}

// ============================================================================
// Clone availability checks
// ============================================================================

/// Return true if `value` can be used while cloning into the new body with the
/// current mapping. Values defined outside the old body still dominate; values
/// defined in the old body must already have been cloned and mapped.
bool isAvailableForClone(Value value, Block *oldBody, const IRMapping &mapping)
{
    if (auto blockArg = dyn_cast<BlockArgument>(value)) {
        return blockArg.getOwner() != oldBody || static_cast<bool>(mapping.lookupOrNull(value));
    }

    Operation *defOp = value.getDefiningOp();
    if (!defOp || defOp->getBlock() != oldBody) {
        return true;
    }
    return static_cast<bool>(mapping.lookupOrNull(value));
}

/// Check all direct and region-captured operands before cloning an op out of the
/// old loop body.
bool areOperandsAvailableForClone(Operation *op, Block *oldBody, const IRMapping &mapping)
{
    for (Value operand : collectOperandsIncludingRegions(op)) {
        if (!isAvailableForClone(operand, oldBody, mapping)) {
            return false;
        }
    }
    return true;
}

// ============================================================================
// Body-run utilities
// ============================================================================

std::optional<int32_t> getBlockId(Operation *op)
{
    if (auto bid = op->getAttrOfType<IntegerAttr>(kBlockIdAttr)) {
        return static_cast<int32_t>(bid.getInt());
    }
    return std::nullopt;
}

bool sameRunBlockId(const BodyRun &run, std::optional<int32_t> bid)
{
    if (run.hasBlockId != bid.has_value()) {
        return false;
    }
    return !run.hasBlockId || run.blockId == *bid;
}

SmallVector<BodyRun> collectBodyRuns(Block *body)
{
    SmallVector<BodyRun> runs;
    for (auto &op : body->without_terminator()) {
        std::optional<int32_t> bid = getBlockId(&op);
        if (runs.empty() || !sameRunBlockId(runs.back(), bid)) {
            BodyRun run;
            if (bid) {
                run.hasBlockId = true;
                run.blockId = *bid;
            }
            runs.push_back(std::move(run));
        }
        runs.back().ops.push_back(&op);
    }
    return runs;
}

int findRunContainingOp(ArrayRef<BodyRun> runs, Operation *target)
{
    for (size_t runIdx = 0; runIdx < runs.size(); ++runIdx) {
        for (Operation *op : runs[runIdx].ops) {
            if (op == target) {
                return static_cast<int>(runIdx);
            }
        }
    }
    return -1;
}

int findFirstRunWithBlockId(ArrayRef<BodyRun> runs, int32_t blockId)
{
    for (size_t runIdx = 0; runIdx < runs.size(); ++runIdx) {
        if (runs[runIdx].hasBlockId && runs[runIdx].blockId == blockId) {
            return static_cast<int>(runIdx);
        }
    }
    return -1;
}

void logRepeatedTopLevelBlockRuns(Block *body, llvm::StringRef stage)
{
    llvm::DenseSet<int32_t> closed;
    bool hasCurrent = false;
    int32_t current = 0;

    for (auto &op : body->without_terminator()) {
        std::optional<int32_t> bid = getBlockId(&op);
        if (!bid) {
            if (hasCurrent) {
                closed.insert(current);
            }
            hasCurrent = false;
            continue;
        }

        if (hasCurrent && current == *bid) {
            continue;
        }

        if (hasCurrent) {
            closed.insert(current);
        }

        if (closed.contains(*bid)) {
            LOG_DEBUG("non-consecutive top-level ssbuffer.block_id run after " << stage << ": block_id=" << *bid
                                                                               << "\n");
        }
        current = *bid;
        hasCurrent = true;
    }
}

// ============================================================================
// Core transformation
// ============================================================================

struct LoopInvariantValues {
    Value lowerBound;
    Value step;
    Value tripCount;
};

struct GroupInvariantValues {
    SmallVector<Value> trueFlags;
    SmallVector<Value> indexOnes;
    SmallVector<SmallVector<Value>> slotConsts;
    SmallVector<Value> depthConsts;
};

void allocateBufferSlotsForGroups(OpBuilder &builder, Location loc, scf::ForOp forOp,
                                  SmallVectorImpl<LoadGroup> &groups)
{
    for (auto &group : groups) {
        allocateBufferSlots(builder, loc, forOp, group);
    }
}

LoopInvariantValues emitLoopInvariantValues(OpBuilder &builder, Location loc, scf::ForOp forOp)
{
    IntegerAttr forBlockId = forOp->getAttrOfType<IntegerAttr>(kBlockIdAttr);
    Value lowerBound = forOp.getLowerBound();
    Value step = forOp.getStep();
    Value upperBound = forOp.getUpperBound();

    Value lowerBoundIndex = castToIndex(builder, loc, lowerBound);
    if (lowerBoundIndex != lowerBound) {
        tagWithBlockId(lowerBoundIndex, forBlockId);
    }
    Value stepIndex = castToIndex(builder, loc, step);
    if (stepIndex != step) {
        tagWithBlockId(stepIndex, forBlockId);
    }
    Value upperBoundIndex = castToIndex(builder, loc, upperBound);
    if (upperBoundIndex != upperBound) {
        tagWithBlockId(upperBoundIndex, forBlockId);
    }

    Value range = builder.create<arith::SubIOp>(loc, upperBoundIndex, lowerBoundIndex);
    tagWithBlockId(range, forBlockId);
    Value tripCount = builder.create<arith::CeilDivUIOp>(loc, range, stepIndex);
    tagWithBlockId(tripCount, forBlockId);
    return {lowerBound, step, tripCount};
}

GroupInvariantValues emitGroupInvariantValues(OpBuilder &builder, Location loc, ArrayRef<LoadGroup> groups)
{
    int numGroups = static_cast<int>(groups.size());
    GroupInvariantValues values;
    values.trueFlags.resize(numGroups);
    values.indexOnes.resize(numGroups);
    values.slotConsts.resize(numGroups);
    values.depthConsts.resize(numGroups);

    for (int groupIdx = 0; groupIdx < numGroups; ++groupIdx) {
        const auto &group = groups[groupIdx];
        IntegerAttr blockId =
            group.loads.empty() ? IntegerAttr {} : group.loads[0].markedOp->getAttrOfType<IntegerAttr>(kBlockIdAttr);
        int groupDepth = group.depth;
        values.trueFlags[groupIdx] = builder.create<arith::ConstantOp>(loc, builder.getBoolAttr(true));
        tagWithBlockId(values.trueFlags[groupIdx], blockId);
        values.indexOnes[groupIdx] = builder.create<arith::ConstantIndexOp>(loc, 1);
        tagWithBlockId(values.indexOnes[groupIdx], blockId);

        values.slotConsts[groupIdx].resize(groupDepth);
        for (int slot = 0; slot < groupDepth; ++slot) {
            values.slotConsts[groupIdx][slot] = builder.create<arith::ConstantIndexOp>(loc, slot);
            tagWithBlockId(values.slotConsts[groupIdx][slot], blockId);
        }

        values.depthConsts[groupIdx] = builder.create<arith::ConstantIndexOp>(loc, groupDepth);
        tagWithBlockId(values.depthConsts[groupIdx], blockId);
    }

    return values;
}

FailureOr<SmallVector<Value>> collectLinearIterArgDeltas(const ExtendedForInfo &info, scf::ForOp forOp)
{
    if (info.numOrig < 0) {
        forOp.emitError() << "[" << DEBUG_TYPE << "] original iter_arg count is negative: " << info.numOrig;
        return failure();
    }
    auto numOrig = static_cast<unsigned>(info.numOrig);
    if (info.oldBody->getNumArguments() < numOrig + kForBodyIterArgOffset) {
        forOp.emitError() << "[" << DEBUG_TYPE << "] old loop body has " << info.oldBody->getNumArguments()
                          << " arguments, expected at least " << (numOrig + kForBodyIterArgOffset);
        return failure();
    }

    auto oldYieldOp = cast<scf::YieldOp>(info.oldBody->getTerminator());
    if (oldYieldOp->getNumOperands() < numOrig) {
        forOp.emitError() << "[" << DEBUG_TYPE << "] old loop yield has " << oldYieldOp->getNumOperands()
                          << " operands, expected at least " << numOrig;
        return failure();
    }

    SmallVector<Value> iterArgDeltas(info.numOrig, Value {});
    for (unsigned iterArgIdx = 0; iterArgIdx < numOrig; ++iterArgIdx) {
        Value delta;
        if (getLinearIterArgDelta(info.oldBody->getArgument(iterArgIdx + kForBodyIterArgOffset),
                                  oldYieldOp->getOperand(iterArgIdx), forOp, delta)) {
            iterArgDeltas[iterArgIdx] = delta;
        }
    }
    return iterArgDeltas;
}

LogicalResult replaceOriginalForResults(scf::ForOp oldForOp, scf::ForOp newForOp, int numOrig)
{
    if (numOrig < 0) {
        oldForOp.emitError() << "[" << DEBUG_TYPE << "] original result count is negative: " << numOrig;
        return failure();
    }

    auto numOrigResults = static_cast<unsigned>(numOrig);
    if (oldForOp.getNumResults() < numOrigResults || newForOp.getNumResults() < numOrigResults) {
        oldForOp.emitError() << "[" << DEBUG_TYPE << "] cannot replace " << numOrigResults
                             << " loop results; old loop has " << oldForOp.getNumResults() << ", new loop has "
                             << newForOp.getNumResults();
        return failure();
    }

    for (unsigned resultIdx = 0; resultIdx < numOrigResults; ++resultIdx) {
        oldForOp.getResult(resultIdx).replaceAllUsesWith(newForOp.getResult(resultIdx));
    }
    return success();
}

LogicalResult finalizeMultiBufferLoop(OpBuilder &builder, scf::ForOp newForOp, int numOrig)
{
    if (numOrig < 0) {
        newForOp.emitError() << "[" << DEBUG_TYPE << "] original iter_arg count is negative: " << numOrig;
        return failure();
    }

    FailureOr<scf::ForOp> finalFor = pruneDeadIterArgs(builder, newForOp, static_cast<unsigned>(numOrig));
    if (failed(finalFor)) {
        return failure();
    }

    eraseDeadBodyOps(finalFor->getBody());

    // Debug-only diagnostic: keep this as a log instead of asserting so the
    // pass can still dump the transformed IR for investigation.
    logRepeatedTopLevelBlockRuns(finalFor->getBody(), kGMLoadMultiBufferStage);
    return success();
}

/// Apply GM-load multi-buffer rewriting to one scf.for.
/// allCtxForOps carries the set of all forOps being transformed so that inner
/// forOps (already handled) are not cloned again into the consumer body.
LogicalResult applyMultiBufferToForLoop(ForBufferCtx &context, const llvm::DenseSet<Operation *> &allCtxForOps)
{
    scf::ForOp forOp = context.forOp;
    Location loc = forOp.getLoc();
    OpBuilder builder(forOp);

    allocateBufferSlotsForGroups(builder, loc, forOp, context.groups);

    // Scan parent block once so buildExtendedFor can reuse existing constants.
    ConstantCache cache;
    cache.scan(forOp->getBlock());

    builder.setInsertionPoint(forOp);
    LoopInvariantValues loopValues = emitLoopInvariantValues(builder, loc, forOp);
    GroupInvariantValues groupValues = emitGroupInvariantValues(builder, loc, context.groups);

    auto info = buildExtendedFor(builder, loc, forOp, context.groups, cache);
    FailureOr<SmallVector<Value>> iterArgDeltas = collectLinearIterArgDeltas(info, forOp);
    if (failed(iterArgDeltas)) {
        return failure();
    }

    LogicalResult emitBodyStatus = emitMultiBufferLoopBody(
        builder, loc, context.groups, info, groupValues.indexOnes, groupValues.slotConsts, groupValues.depthConsts,
        allCtxForOps, loopValues.tripCount, loopValues.lowerBound, loopValues.step, groupValues.trueFlags,
        *iterArgDeltas, forOp);
    if (failed(emitBodyStatus)) {
        return failure();
    }

    if (failed(replaceOriginalForResults(forOp, info.newForOp, info.numOrig))) {
        return failure();
    }
    if (failed(finalizeMultiBufferLoop(builder, info.newForOp, info.numOrig))) {
        return failure();
    }
    return success();
}

// ============================================================================
// Cleanup utilities
// ============================================================================

bool isAncestorBlock(Block *ancestor, Block *descendant)
{
    for (Block *currentBlock = descendant; currentBlock;) {
        if (currentBlock == ancestor) {
            return true;
        }
        auto *parentOp = currentBlock->getParentOp();
        if (!parentOp) {
            break;
        }
        currentBlock = parentOp->getBlock();
    }
    return false;
}

void deduplicateConstants(ModuleOp module)
{
    llvm::DenseMap<mlir::Attribute, Value> canonical;
    SmallVector<Operation *> toErase;

    module->walk<mlir::WalkOrder::PreOrder>([&](arith::ConstantOp cst) {
        if (cst->hasAttr(kBlockIdAttr)) {
            return mlir::WalkResult::advance();
        }
        auto [canonicalIt, inserted] = canonical.try_emplace(cst.getValue(), cst.getResult());
        if (!inserted && canonicalIt->second != cst.getResult()) {
            Block *canonBlock = canonicalIt->second.getParentBlock();
            Block *thisBlock = cst->getBlock();
            if (canonBlock == thisBlock || isAncestorBlock(canonBlock, thisBlock)) {
                cst.getResult().replaceAllUsesWith(canonicalIt->second);
                toErase.push_back(cst);
            }
        }
        return mlir::WalkResult::advance();
    });

    for (Operation *op : llvm::reverse(toErase)) {
        op->erase();
    }
}

} // namespace gmload
