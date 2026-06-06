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
// Producer emission helpers
// ============================================================================

static bool canSkipBecauseAllResultsMapped(Operation *op, const IRMapping &mapping)
{
    if (op->getNumResults() == 0) {
        return false;
    }
    return llvm::all_of(op->getResults(),
                        [&](Value result) { return static_cast<bool>(mapping.lookupOrNull(result)); });
}

static LogicalResult materializeProducerOp(OpBuilder &builder, Operation *op, Block *oldBody,
                                           const llvm::DenseSet<Operation *> &producerSourceOps, IRMapping &mapping,
                                           llvm::DenseSet<Operation *> &materializedOps,
                                           llvm::DenseSet<Operation *> &activeOps);

static LogicalResult materializeProducerValue(OpBuilder &builder, Value value, Block *oldBody,
                                              const llvm::DenseSet<Operation *> &producerSourceOps, IRMapping &mapping,
                                              llvm::DenseSet<Operation *> &materializedOps,
                                              llvm::DenseSet<Operation *> &activeOps)
{
    if (mapping.lookupOrNull(value)) {
        return success();
    }

    if (auto blockArg = dyn_cast<BlockArgument>(value)) {
        if (blockArg.getOwner() == oldBody) {
            return failure();
        }
        return success();
    }

    Operation *defOp = value.getDefiningOp();
    if (!defOp) {
        return success();
    }

    if (producerSourceOps.contains(defOp) || defOp->getBlock() == oldBody) {
        return materializeProducerOp(builder, defOp, oldBody, producerSourceOps, mapping, materializedOps, activeOps);
    }

    return success();
}

static LogicalResult materializeProducerOp(OpBuilder &builder, Operation *op, Block *oldBody,
                                           const llvm::DenseSet<Operation *> &producerSourceOps, IRMapping &mapping,
                                           llvm::DenseSet<Operation *> &materializedOps,
                                           llvm::DenseSet<Operation *> &activeOps)
{
    if (materializedOps.contains(op) || canSkipBecauseAllResultsMapped(op, mapping)) {
        return success();
    }

    if (activeOps.contains(op)) {
        return failure();
    }

    if (!producerSourceOps.contains(op) && op->getBlock() == oldBody) {
        if (!mlir::isMemoryEffectFree(op) || op->getNumRegions() != 0) {
            return failure();
        }
    }

    activeOps.insert(op);
    for (Value operand : collectOperandsIncludingRegions(op)) {
        LogicalResult materializeStatus = materializeProducerValue(
            builder, operand, oldBody, producerSourceOps, mapping, materializedOps, activeOps);
        if (failed(materializeStatus)) {
            activeOps.erase(op);
            return failure();
        }
    }
    activeOps.erase(op);

    builder.clone(*op, mapping);
    materializedOps.insert(op);
    return success();
}

static IntegerAttr getGroupBlockId(const LoadGroup &group)
{
    return group.loads.empty() ? IntegerAttr {} : group.loads[0].markedOp->getAttrOfType<IntegerAttr>(kBlockIdAttr);
}

static Value createProducerFillCondition(OpBuilder &builder, Location loc, Value flagArg, Value currentProducerCounter,
                                         Value tripCount, Value falseVal, IntegerAttr blockId)
{
    Value flagEmpty = builder.create<arith::CmpIOp>(loc, arith::CmpIPredicate::eq, flagArg, falseVal);
    tagWithBlockId(flagEmpty, blockId);
    Value prodLtTripCount =
        builder.create<arith::CmpIOp>(loc, arith::CmpIPredicate::ult, currentProducerCounter, tripCount);
    tagWithBlockId(prodLtTripCount, blockId);
    Value cond = builder.create<arith::AndIOp>(loc, flagEmpty, prodLtTripCount);
    tagWithBlockId(cond, blockId);
    return cond;
}

static scf::IfOp createProducerFillIf(OpBuilder &builder, Location loc, Value cond, IntegerAttr blockId)
{
    auto ifOp = builder.create<scf::IfOp>(loc, TypeRange {builder.getI1Type(), builder.getIndexType()}, cond, true);
    if (blockId)
        ifOp->setAttr(kBlockIdAttr, blockId);
    ifOp->setAttr(kLoadStoreAttr, builder.getUnitAttr());
    return ifOp;
}

struct ProducerLoopPosition {
    Type inductionVarType;
    Value projectedInductionVar;
    Value producerCounterInInductionType;
};

static ProducerLoopPosition mapProducerLoopPosition(OpBuilder &builder, Location loc, Block *oldBody,
                                                    Value loopLowerBound, Value loopStep, Value currentProducerCounter,
                                                    IntegerAttr blockId, IRMapping &prodMapping)
{
    Value oldInductionVar = oldBody->getArgument(kForInductionVarArgIndex);
    Type inductionVarType = oldInductionVar.getType();

    Value lowerBound = castIndexTo(builder, loc, loopLowerBound, inductionVarType);
    if (lowerBound != loopLowerBound)
        tagWithBlockId(lowerBound, blockId);
    Value step = castIndexTo(builder, loc, loopStep, inductionVarType);
    if (step != loopStep)
        tagWithBlockId(step, blockId);
    Value producerCounterInInductionType = castIndexTo(builder, loc, currentProducerCounter, inductionVarType);
    tagWithBlockId(producerCounterInInductionType, blockId);
    Value inductionOffset = builder.create<arith::MulIOp>(loc, producerCounterInInductionType, step);
    tagWithBlockId(inductionOffset, blockId);
    Value projectedInductionVar = builder.create<arith::AddIOp>(loc, lowerBound, inductionOffset);
    tagWithBlockId(projectedInductionVar, blockId);
    prodMapping.map(oldInductionVar, projectedInductionVar);

    return {inductionVarType, projectedInductionVar, producerCounterInInductionType};
}

static LogicalResult validateProducerIterArgInputs(scf::ForOp origForOp, Block *oldBody, Block *newBody,
                                                   ArrayRef<Value> iterArgDeltas)
{
    int numOrig = static_cast<int>(origForOp.getInitArgs().size());
    if (numOrig < 0) {
        origForOp.emitError() << "[" << DEBUG_TYPE << "] original iter_arg count is negative: " << numOrig;
        return failure();
    }

    auto numOrigArgs = static_cast<unsigned>(numOrig);
    if (iterArgDeltas.size() < numOrigArgs) {
        origForOp.emitError() << "[" << DEBUG_TYPE << "] iter_arg delta count " << iterArgDeltas.size()
                              << " is smaller than original iter_arg count " << numOrigArgs;
        return failure();
    }
    if (oldBody->getNumArguments() < numOrigArgs + kForBodyIterArgOffset ||
        newBody->getNumArguments() < numOrigArgs + kForBodyIterArgOffset) {
        origForOp.emitError() << "[" << DEBUG_TYPE << "] loop body argument count is smaller than expected "
                              << (numOrigArgs + kForBodyIterArgOffset);
        return failure();
    }
    return success();
}

static LogicalResult mapProducerIterArgs(OpBuilder &builder, Location loc, Block *oldBody, Block *newBody,
                                         scf::ForOp origForOp, Value loopLowerBound, Value loopStep,
                                         Value currentProducerCounter, const SmallVector<Value> &iterArgDeltas,
                                         const ProducerLoopPosition &position, IntegerAttr blockId,
                                         IRMapping &prodMapping)
{
    if (failed(validateProducerIterArgInputs(origForOp, oldBody, newBody, iterArgDeltas))) {
        return failure();
    }

    // producerCounterInInductionType is already cast to inductionVarType; cache casts by type to avoid
    // redundant index_cast ops when multiple iter_args share the same type.
    llvm::DenseMap<Type, Value> producerCounterByType;
    producerCounterByType[position.inductionVarType] = position.producerCounterInInductionType;

    int numOrig = static_cast<int>(origForOp.getInitArgs().size());
    for (int iterArgIdx = 0; iterArgIdx < numOrig; ++iterArgIdx) {
        Value oldArg = oldBody->getArgument(iterArgIdx + kForBodyIterArgOffset);
        Value delta = iterArgDeltas[iterArgIdx];
        if (!delta) {
            prodMapping.map(oldArg, newBody->getArgument(iterArgIdx + kForBodyIterArgOffset));
            continue;
        }

        Type argType = oldArg.getType();
        Value initArg = origForOp.getInitArgs()[iterArgIdx];
        // If initArg == lower bound and delta == step this iter_arg tracks the
        // induction variable exactly; reuse the projected IV.
        if (initArg == loopLowerBound && delta == loopStep && argType == position.inductionVarType) {
            prodMapping.map(oldArg, position.projectedInductionVar);
            continue;
        }

        auto [counterIt, inserted] = producerCounterByType.try_emplace(argType, Value {});
        if (inserted) {
            counterIt->second = castIndexTo(builder, loc, currentProducerCounter, argType);
            tagWithBlockId(counterIt->second, blockId);
        }
        Value argMul = builder.create<arith::MulIOp>(loc, counterIt->second, delta);
        tagWithBlockId(argMul, blockId);
        Value argAdd = builder.create<arith::AddIOp>(loc, initArg, argMul);
        tagWithBlockId(argAdd, blockId);
        prodMapping.map(oldArg, argAdd);
    }
    return success();
}

static void mapProducerSlotBuffers(const LoadGroup &group, int slotIdx, IRMapping &prodMapping)
{
    for (int loadIdx = 0; loadIdx < static_cast<int>(group.loads.size()); ++loadIdx)
        prodMapping.map(group.loads[loadIdx].allocOp->getResult(0), group.bufSlots[slotIdx][loadIdx]);
}

static LogicalResult materializeProducerChain(OpBuilder &builder, Block *oldBody, const LoadGroup &group,
                                              IRMapping &prodMapping)
{
    llvm::DenseSet<Operation *> skipSet;
    for (const MarkedLoad &load : group.loads) {
        skipSet.insert(load.allocOp);
        skipSet.insert(load.markedOp);
    }

    llvm::DenseSet<Operation *> producerSourceOps(group.mergedChain.begin(), group.mergedChain.end());
    llvm::DenseSet<Operation *> materializedOps;
    llvm::DenseSet<Operation *> activeOps;
    for (Operation *op : group.mergedChain) {
        if (skipSet.contains(op)) {
            continue;
        }
        if (failed(materializeProducerOp(builder, op, oldBody, producerSourceOps, prodMapping, materializedOps,
                                         activeOps)))
            return failure();
    }
    return success();
}

static LogicalResult emitProducerFillThenBlock(OpBuilder &builder, Location loc, scf::IfOp ifOp, Block *oldBody,
                                               Block *newBody, const LoadGroup &group, int slotIdx,
                                               Value currentProducerCounter, Value loopLowerBound, Value loopStep,
                                               Value trueFlag, Value indexOne, const SmallVector<Value> &iterArgDeltas,
                                               scf::ForOp origForOp, IntegerAttr blockId)
{
    OpBuilder::InsertionGuard guard(builder);
    builder.setInsertionPointToStart(ifOp.thenBlock());

    IRMapping prodMapping;
    ProducerLoopPosition position = mapProducerLoopPosition(builder, loc, oldBody, loopLowerBound, loopStep,
                                                            currentProducerCounter, blockId, prodMapping);
    LogicalResult mapIterArgStatus = mapProducerIterArgs(
        builder, loc, oldBody, newBody, origForOp, loopLowerBound, loopStep, currentProducerCounter, iterArgDeltas,
        position, blockId, prodMapping);
    if (failed(mapIterArgStatus)) {
        return failure();
    }
    mapProducerSlotBuffers(group, slotIdx, prodMapping);
    if (failed(materializeProducerChain(builder, oldBody, group, prodMapping))) {
        return failure();
    }

    Value prodNext = builder.create<arith::AddIOp>(loc, currentProducerCounter, indexOne);
    tagWithBlockId(prodNext, blockId);
    builder.create<scf::YieldOp>(loc, ValueRange {trueFlag, prodNext});
    return success();
}

static void emitProducerFillElseBlock(OpBuilder &builder, Location loc, scf::IfOp ifOp, Value flagArg,
                                      Value currentProducerCounter)
{
    OpBuilder::InsertionGuard guard(builder);
    builder.setInsertionPointToStart(ifOp.elseBlock());
    builder.create<scf::YieldOp>(loc, ValueRange {flagArg, currentProducerCounter});
}

/// Emit the producer scf.if for one buffer slot.  Returns {flagNext, prodCounterNext}.
FailureOr<std::pair<Value, Value>> emitProducerFillForSlot(OpBuilder &builder, Location loc, Block *oldBody,
                                                           Block *newBody, const LoadGroup &group, int slotIdx,
                                                           Value flagArg, Value currentProducerCounter,
                                                           Value tripCount, Value falseVal, Value loopLowerBound,
                                                           Value loopStep, Value trueFlag, Value indexOne,
                                                           const SmallVector<Value> &iterArgDeltas,
                                                           scf::ForOp origForOp)
{
    IntegerAttr blockId = getGroupBlockId(group);
    Value cond =
        createProducerFillCondition(builder, loc, flagArg, currentProducerCounter, tripCount, falseVal, blockId);
    scf::IfOp ifOp = createProducerFillIf(builder, loc, cond, blockId);
    LogicalResult thenStatus = emitProducerFillThenBlock(
        builder, loc, ifOp, oldBody, newBody, group, slotIdx, currentProducerCounter, loopLowerBound, loopStep,
        trueFlag, indexOne, iterArgDeltas, origForOp, blockId);
    if (failed(thenStatus)) {
        return failure();
    }
    emitProducerFillElseBlock(builder, loc, ifOp, flagArg, currentProducerCounter);

    return std::make_pair(ifOp.getResult(0), ifOp.getResult(1));
}

// ============================================================================
// Consumer emission helpers
// ============================================================================

/// Emit slot-selection for one load: one ToTensorOp per slot, followed by a
/// (depth-1)-level comparison/select chain that picks the slot matching
/// `target`.  Sets mapping[markedOp->result] = selected value.
Value emitLoadSlotSelection(OpBuilder &builder, Location loc, const MarkedLoad &load, ArrayRef<Value> slotBufs,
                            Value target, int depth, IRMapping &mapping, ArrayRef<Value> slotConsts)
{
    IntegerAttr blockId = load.markedOp->getAttrOfType<IntegerAttr>(kBlockIdAttr);
    Type tensorTy = load.markedOp->getResult(0).getType();

    SmallVector<Value> slotVals(depth);
    for (int slotIdx = 0; slotIdx < depth; ++slotIdx) {
        auto toTensor = builder.create<bufferization::ToTensorOp>(loc, tensorTy, slotBufs[slotIdx], true, true);
        if (blockId)
            toTensor->setAttr(kBlockIdAttr, blockId);
        slotVals[slotIdx] = toTensor;
    }

    Value result = slotVals[depth - 1];
    for (int slotIdx = depth - 2; slotIdx >= 0; --slotIdx) {
        Value eq = builder.create<arith::CmpIOp>(loc, arith::CmpIPredicate::eq, target, slotConsts[slotIdx]);
        tagWithBlockId(eq, blockId);
        result = builder.create<arith::SelectOp>(loc, eq, slotVals[slotIdx], result);
        tagWithBlockId(result, blockId);
    }

    mapping.map(load.markedOp->getResult(0), result);
    return result;
}

/// Emit the scf.if that clears slot `slotIdx`'s flag when it is the consumed
/// slot (`target == slotIdx`).  Returns the updated flag value.
Value emitSlotFlagClear(OpBuilder &builder, Location loc, Value target, Value slotConst, Value flagNext, Value falseVal,
                        IntegerAttr blockId)
{
    Value isConsumed = builder.create<arith::CmpIOp>(loc, arith::CmpIPredicate::eq, target, slotConst);
    tagWithBlockId(isConsumed, blockId);

    auto ifClear = builder.create<scf::IfOp>(loc, TypeRange {builder.getI1Type()}, isConsumed, true);
    if (blockId)
        ifClear->setAttr(kBlockIdAttr, blockId);
    ifClear->setAttr(kLoadStoreAttr, builder.getUnitAttr());
    {
        OpBuilder::InsertionGuard guard(builder);
        builder.setInsertionPointToStart(ifClear.thenBlock());
        builder.create<scf::YieldOp>(loc, ValueRange {falseVal});
    }
    {
        OpBuilder::InsertionGuard guard(builder);
        builder.setInsertionPointToStart(ifClear.elseBlock());
        builder.create<scf::YieldOp>(loc, ValueRange {flagNext});
    }
    return ifClear.getResult(0);
}

// ============================================================================

namespace {

struct MultiBufferLoopBodyState {
    SmallVector<SmallVector<Value>> flagFinals;
    SmallVector<Value> consCounterNexts;
    SmallVector<Value> prodCounterFinals;
    SmallVector<SmallVector<Value>> flagNexts;
    SmallVector<Value> targets;
    SmallVector<char> prefixEmitted;
    SmallVector<char> finalEmitted;

    explicit MultiBufferLoopBodyState(int numGroups)
        : flagFinals(numGroups), consCounterNexts(numGroups), prodCounterFinals(numGroups), flagNexts(numGroups),
          targets(numGroups), prefixEmitted(numGroups, 0), finalEmitted(numGroups, 0)
    {
    }
};

static void computeConsumerCloneSets(ArrayRef<LoadGroup> groups, Block *oldBody,
                                     llvm::DenseSet<Operation *> &allChainOps,
                                     llvm::DenseSet<Operation *> &retainedChainOpSet)
{
    // A retained chain op is a chain op that must still be cloned into the
    // consumer body because non-chain body code uses it.  Retained ops are
    // emitted in their original top-level run, not in the load group that
    // discovered them.
    for (const LoadGroup &group : groups) {
        llvm::DenseSet<Operation *> skipSet = computeSkipInConsumer(group.loads, oldBody);
        for (Operation *op : group.mergedChain) {
            allChainOps.insert(op);
            if (!skipSet.contains(op))
                retainedChainOpSet.insert(op);
        }
    }
}

static bool shouldSkipOldBodyOp(Operation *op, const llvm::DenseSet<Operation *> &allCtxForOps,
                                const llvm::DenseSet<Operation *> &allChainOps,
                                const llvm::DenseSet<Operation *> &retainedChainOpSet)
{
    if (allCtxForOps.contains(op)) {
        return true;
    }
    return allChainOps.contains(op) && !retainedChainOpSet.contains(op);
}

struct ProducerFillState {
    SmallVector<Value> flagNexts;
    Value producerCounter;
};

static FailureOr<ProducerFillState> emitProducerFillsForGroup(
    OpBuilder &builder, Location loc, LoadGroup &group, int groupIdx, ExtendedForInfo &info, Block *oldBody,
    Value tripCount, Value loopLowerBound, Value loopStep, const SmallVector<Value> &groupTrueFlagVals,
    const SmallVector<Value> &groupIndexOneVals, const SmallVector<Value> &iterArgDeltas, scf::ForOp origForOp)
{
    int groupDepth = group.depth;
    Value currentProducerCounter = info.groupArgs[groupIdx].prodCounter;
    SmallVector<Value> flagNexts(groupDepth);
    for (int slotIdx = 0; slotIdx < groupDepth; ++slotIdx) {
        FailureOr<std::pair<Value, Value>> producerResult = emitProducerFillForSlot(
            builder, loc, oldBody, info.newBody, group, slotIdx, info.groupArgs[groupIdx].flagArgs[slotIdx],
            currentProducerCounter, tripCount, info.falseVal, loopLowerBound, loopStep, groupTrueFlagVals[groupIdx],
            groupIndexOneVals[groupIdx], iterArgDeltas, origForOp);
        if (failed(producerResult)) {
            return failure();
        }

        auto [flagNext, prodCounterNext] = *producerResult;
        flagNexts[slotIdx] = flagNext;
        currentProducerCounter = prodCounterNext;
    }

    return ProducerFillState {std::move(flagNexts), currentProducerCounter};
}

static Value emitConsumerSlotTarget(OpBuilder &builder, Location loc, ExtendedForInfo &info, int groupIdx,
                                    Value groupDepthConst, IntegerAttr blockId)
{
    Value target = builder.create<arith::RemUIOp>(loc, info.groupArgs[groupIdx].consCounter, groupDepthConst);
    tagWithBlockId(target, blockId);
    return target;
}

static void emitLoadSlotSelectionsForGroup(OpBuilder &builder, Location loc, LoadGroup &group, int groupIdx,
                                           int groupDepth, Value target, ExtendedForInfo &info,
                                           const SmallVector<SmallVector<Value>> &allGroupSlotConsts)
{
    for (int loadIdx = 0; loadIdx < static_cast<int>(group.loads.size()); ++loadIdx) {
        SmallVector<Value> slotBufs(groupDepth);
        for (int slotIdx = 0; slotIdx < groupDepth; ++slotIdx)
            slotBufs[slotIdx] = group.bufSlots[slotIdx][loadIdx];
        emitLoadSlotSelection(builder, loc, group.loads[loadIdx], slotBufs, target, groupDepth, info.mapping,
                              allGroupSlotConsts[groupIdx]);
    }
}

static LogicalResult validateGroupEmissionInputs(ArrayRef<LoadGroup> groups, ExtendedForInfo &info, int groupIdx,
                                                 ArrayRef<Value> groupTrueFlagVals, ArrayRef<Value> groupIndexOneVals,
                                                 ArrayRef<Value> allGroupDepthConsts,
                                                 ArrayRef<SmallVector<Value>> allGroupSlotConsts, scf::ForOp origForOp)
{
    if (groupIdx < 0 || static_cast<size_t>(groupIdx) >= groups.size() ||
        static_cast<size_t>(groupIdx) >= info.groupArgs.size() ||
        static_cast<size_t>(groupIdx) >= groupTrueFlagVals.size() ||
        static_cast<size_t>(groupIdx) >= groupIndexOneVals.size() ||
        static_cast<size_t>(groupIdx) >= allGroupDepthConsts.size() ||
        static_cast<size_t>(groupIdx) >= allGroupSlotConsts.size()) {
        origForOp.emitError() << "[" << DEBUG_TYPE << "] load group index " << groupIdx
                              << " is out of range for multi-buffer emission";
        return failure();
    }
    return success();
}

static LogicalResult emitProducerAndSelectSlots(
    OpBuilder &builder, Location loc, SmallVectorImpl<LoadGroup> &groups, int groupIdx, ExtendedForInfo &info,
    Block *oldBody, Value tripCount, Value loopLowerBound, Value loopStep,
    const SmallVector<Value> &groupTrueFlagVals, const SmallVector<Value> &groupIndexOneVals,
    const SmallVector<Value> &iterArgDeltas, scf::ForOp origForOp, const SmallVector<Value> &allGroupDepthConsts,
    const SmallVector<SmallVector<Value>> &allGroupSlotConsts, MultiBufferLoopBodyState &state)
{
    LogicalResult validateStatus = validateGroupEmissionInputs(
        groups, info, groupIdx, groupTrueFlagVals, groupIndexOneVals, allGroupDepthConsts, allGroupSlotConsts,
        origForOp);
    if (failed(validateStatus)) {
        return failure();
    }
    if (static_cast<size_t>(groupIdx) >= state.prefixEmitted.size()) {
        origForOp.emitError() << "[" << DEBUG_TYPE << "] load group index " << groupIdx
                              << " exceeds multi-buffer state size " << state.prefixEmitted.size();
        return failure();
    }
    if (state.prefixEmitted[groupIdx]) {
        return success();
    }

    LoadGroup &group = groups[groupIdx];
    int groupDepth = group.depth;
    IntegerAttr blockId = getGroupBlockId(group);

    FailureOr<ProducerFillState> fillState =
        emitProducerFillsForGroup(builder, loc, group, groupIdx, info, oldBody, tripCount, loopLowerBound, loopStep,
                                  groupTrueFlagVals, groupIndexOneVals, iterArgDeltas, origForOp);
    if (failed(fillState)) {
        return failure();
    }

    state.flagNexts[groupIdx] = std::move(fillState->flagNexts);
    state.prodCounterFinals[groupIdx] = fillState->producerCounter;
    Value target = emitConsumerSlotTarget(builder, loc, info, groupIdx, allGroupDepthConsts[groupIdx], blockId);
    state.targets[groupIdx] = target;
    emitLoadSlotSelectionsForGroup(builder, loc, group, groupIdx, groupDepth, target, info, allGroupSlotConsts);

    state.prefixEmitted[groupIdx] = 1;
    return success();
}

static LogicalResult emitConsumerReleaseForGroup(OpBuilder &builder, Location loc, SmallVectorImpl<LoadGroup> &groups,
                                                 int groupIdx, ExtendedForInfo &info,
                                                 const SmallVector<Value> &groupIndexOneVals,
                                                 const SmallVector<SmallVector<Value>> &allGroupSlotConsts,
                                                 MultiBufferLoopBodyState &state)
{
    if (state.finalEmitted[groupIdx]) {
        return success();
    }

    LoadGroup &group = groups[groupIdx];
    int groupDepth = group.depth;
    IntegerAttr blockId = getGroupBlockId(group);

    state.flagFinals[groupIdx].resize(groupDepth);
    for (int slotIdx = 0; slotIdx < groupDepth; ++slotIdx) {
        state.flagFinals[groupIdx][slotIdx] =
            emitSlotFlagClear(builder, loc, state.targets[groupIdx], allGroupSlotConsts[groupIdx][slotIdx],
                              state.flagNexts[groupIdx][slotIdx], info.falseVal, blockId);
    }

    state.consCounterNexts[groupIdx] =
        builder.create<arith::AddIOp>(loc, info.groupArgs[groupIdx].consCounter, groupIndexOneVals[groupIdx]);
    tagWithBlockId(state.consCounterNexts[groupIdx], blockId);
    state.finalEmitted[groupIdx] = 1;
    return success();
}

static LogicalResult emitConsumerReleaseAfterPrefix(
    OpBuilder &builder, Location loc, SmallVectorImpl<LoadGroup> &groups, int groupIdx, ExtendedForInfo &info,
    Block *oldBody, Value tripCount, Value loopLowerBound, Value loopStep, const SmallVector<Value> &groupTrueFlagVals,
    const SmallVector<Value> &groupIndexOneVals, const SmallVector<Value> &iterArgDeltas, scf::ForOp origForOp,
    const SmallVector<Value> &allGroupDepthConsts, const SmallVector<SmallVector<Value>> &allGroupSlotConsts,
    MultiBufferLoopBodyState &state)
{
    if (!state.prefixEmitted[groupIdx]) {
        LogicalResult producerStatus = emitProducerAndSelectSlots(
            builder, loc, groups, groupIdx, info, oldBody, tripCount, loopLowerBound, loopStep, groupTrueFlagVals,
            groupIndexOneVals, iterArgDeltas, origForOp, allGroupDepthConsts, allGroupSlotConsts, state);
        if (failed(producerStatus)) {
            return failure();
        }
    }

    return emitConsumerReleaseForGroup(builder, loc, groups, groupIdx, info, groupIndexOneVals, allGroupSlotConsts,
                                       state);
}

static LogicalResult collectGroupsByOwnerOp(ArrayRef<LoadGroup> groups, Block *oldBody,
                                            llvm::DenseMap<Operation *, SmallVector<int>> &groupsByOwnerOp)
{
    for (int groupIdx = 0; groupIdx < static_cast<int>(groups.size()); ++groupIdx) {
        if (groups[groupIdx].loads.empty()) {
            continue;
        }

        Operation *owner = getAncestorInBlock(groups[groupIdx].loads[0].markedOp, oldBody);
        if (!owner) {
            return failure();
        }
        groupsByOwnerOp[owner].push_back(groupIdx);
    }
    return success();
}

static LogicalResult emitProducerHooksForOwnerGroups(
    OpBuilder &builder, Location loc, ArrayRef<int> ownerGroups, SmallVectorImpl<LoadGroup> &groups,
    ExtendedForInfo &info, Block *oldBody, Value tripCount, Value loopLowerBound, Value loopStep,
    const SmallVector<Value> &groupTrueFlagVals, const SmallVector<Value> &groupIndexOneVals,
    const SmallVector<Value> &iterArgDeltas, scf::ForOp origForOp, const SmallVector<Value> &allGroupDepthConsts,
    const SmallVector<SmallVector<Value>> &allGroupSlotConsts, MultiBufferLoopBodyState &state)
{
    for (int groupIdx : ownerGroups) {
        LogicalResult producerStatus = emitProducerAndSelectSlots(
            builder, loc, groups, groupIdx, info, oldBody, tripCount, loopLowerBound, loopStep, groupTrueFlagVals,
            groupIndexOneVals, iterArgDeltas, origForOp, allGroupDepthConsts, allGroupSlotConsts, state);
        if (failed(producerStatus)) {
            return failure();
        }
    }
    return success();
}

static LogicalResult emitConsumerReleaseHooksForOwnerGroups(
    OpBuilder &builder, Location loc, ArrayRef<int> ownerGroups, SmallVectorImpl<LoadGroup> &groups,
    ExtendedForInfo &info, Block *oldBody, Value tripCount, Value loopLowerBound, Value loopStep,
    const SmallVector<Value> &groupTrueFlagVals, const SmallVector<Value> &groupIndexOneVals,
    const SmallVector<Value> &iterArgDeltas, scf::ForOp origForOp, const SmallVector<Value> &allGroupDepthConsts,
    const SmallVector<SmallVector<Value>> &allGroupSlotConsts, MultiBufferLoopBodyState &state)
{
    for (int groupIdx : ownerGroups) {
        LogicalResult releaseStatus = emitConsumerReleaseAfterPrefix(
            builder, loc, groups, groupIdx, info, oldBody, tripCount, loopLowerBound, loopStep, groupTrueFlagVals,
            groupIndexOneVals, iterArgDeltas, origForOp, allGroupDepthConsts, allGroupSlotConsts, state);
        if (failed(releaseStatus)) {
            return failure();
        }
    }
    return success();
}

static LogicalResult cloneOldBodyOpIfNeeded(OpBuilder &builder, Operation *op, Block *oldBody, IRMapping &mapping,
                                            const llvm::DenseSet<Operation *> &allCtxForOps,
                                            const llvm::DenseSet<Operation *> &allChainOps,
                                            const llvm::DenseSet<Operation *> &retainedChainOpSet)
{
    if (shouldSkipOldBodyOp(op, allCtxForOps, allChainOps, retainedChainOpSet)) {
        return success();
    }
    if (!areOperandsAvailableForClone(op, oldBody, mapping)) {
        return failure();
    }
    builder.clone(*op, mapping);
    return success();
}

static LogicalResult emitOldBodyWithMultiBufferHooks(
    OpBuilder &builder, Location loc, SmallVectorImpl<LoadGroup> &groups, ExtendedForInfo &info,
    const llvm::DenseMap<Operation *, SmallVector<int>> &groupsByOwnerOp,
    const llvm::DenseSet<Operation *> &allCtxForOps, const llvm::DenseSet<Operation *> &allChainOps,
    const llvm::DenseSet<Operation *> &retainedChainOpSet, Value tripCount, Value loopLowerBound, Value loopStep,
    const SmallVector<Value> &groupTrueFlagVals, const SmallVector<Value> &groupIndexOneVals,
    const SmallVector<Value> &iterArgDeltas, scf::ForOp origForOp, const SmallVector<Value> &allGroupDepthConsts,
    const SmallVector<SmallVector<Value>> &allGroupSlotConsts, MultiBufferLoopBodyState &state)
{
    Block *oldBody = info.oldBody;
    for (auto &op : oldBody->without_terminator()) {
        auto ownerIt = groupsByOwnerOp.find(&op);
        if (ownerIt != groupsByOwnerOp.end()) {
            LogicalResult producerHookStatus = emitProducerHooksForOwnerGroups(
                builder, loc, ownerIt->second, groups, info, oldBody, tripCount, loopLowerBound, loopStep,
                groupTrueFlagVals, groupIndexOneVals, iterArgDeltas, origForOp, allGroupDepthConsts,
                allGroupSlotConsts, state);
            if (failed(producerHookStatus)) {
                return failure();
            }
        }

        LogicalResult cloneStatus =
            cloneOldBodyOpIfNeeded(builder, &op, oldBody, info.mapping, allCtxForOps, allChainOps, retainedChainOpSet);
        if (failed(cloneStatus)) {
            return failure();
        }

        if (ownerIt != groupsByOwnerOp.end()) {
            LogicalResult releaseHookStatus = emitConsumerReleaseHooksForOwnerGroups(
                builder, loc, ownerIt->second, groups, info, oldBody, tripCount, loopLowerBound, loopStep,
                groupTrueFlagVals, groupIndexOneVals, iterArgDeltas, origForOp, allGroupDepthConsts,
                allGroupSlotConsts, state);
            if (failed(releaseHookStatus)) {
                return failure();
            }
        }
    }
    return success();
}

static LogicalResult emitUnownedGroupsAtBodyEnd(
    OpBuilder &builder, Location loc, SmallVectorImpl<LoadGroup> &groups, ExtendedForInfo &info, Block *oldBody,
    Value tripCount, Value loopLowerBound, Value loopStep, const SmallVector<Value> &groupTrueFlagVals,
    const SmallVector<Value> &groupIndexOneVals, const SmallVector<Value> &iterArgDeltas, scf::ForOp origForOp,
    const SmallVector<Value> &allGroupDepthConsts, const SmallVector<SmallVector<Value>> &allGroupSlotConsts,
    MultiBufferLoopBodyState &state)
{
    for (int groupIdx = 0; groupIdx < static_cast<int>(groups.size()); ++groupIdx) {
        if (state.prefixEmitted[groupIdx]) {
            continue;
        }

        LogicalResult producerStatus = emitProducerAndSelectSlots(
            builder, loc, groups, groupIdx, info, oldBody, tripCount, loopLowerBound, loopStep, groupTrueFlagVals,
            groupIndexOneVals, iterArgDeltas, origForOp, allGroupDepthConsts, allGroupSlotConsts, state);
        if (failed(producerStatus)) {
            return failure();
        }
        LogicalResult releaseStatus = emitConsumerReleaseAfterPrefix(
            builder, loc, groups, groupIdx, info, oldBody, tripCount, loopLowerBound, loopStep, groupTrueFlagVals,
            groupIndexOneVals, iterArgDeltas, origForOp, allGroupDepthConsts, allGroupSlotConsts, state);
        if (failed(releaseStatus)) {
            return failure();
        }
    }
    return success();
}

static void emitCombinedYield(OpBuilder &builder, Location loc, ArrayRef<LoadGroup> groups, ExtendedForInfo &info,
                              const MultiBufferLoopBodyState &state)
{
    auto oldYield = cast<scf::YieldOp>(info.oldBody->getTerminator());
    SmallVector<Value> yieldVals;
    yieldVals.reserve(info.numOrig + static_cast<int>(groups.size()) * (info.depth + kLoadGroupCounterIterArgCount));
    for (Value yieldedValue : oldYield.getOperands()) {
        yieldVals.push_back(info.mapping.lookupOrDefault(yieldedValue));
    }
    for (int groupIdx = 0; groupIdx < static_cast<int>(groups.size()); ++groupIdx) {
        for (int slotIdx = 0; slotIdx < groups[groupIdx].depth; ++slotIdx) {
            yieldVals.push_back(state.flagFinals[groupIdx][slotIdx]);
        }
        yieldVals.push_back(state.prodCounterFinals[groupIdx]);
        yieldVals.push_back(state.consCounterNexts[groupIdx]);
    }
    builder.create<scf::YieldOp>(loc, yieldVals);
}

} // namespace

/// Emit the full loop body with producer prefetch, consumer slot selection,
/// cloned compute, slot release, and a combined scf.yield.
LogicalResult emitMultiBufferLoopBody(OpBuilder &builder, Location loc, SmallVectorImpl<LoadGroup> &groups,
                                      ExtendedForInfo &info, const SmallVector<Value> &groupIndexOneVals,
                                      const SmallVector<SmallVector<Value>> &allGroupSlotConsts,
                                      const SmallVector<Value> &allGroupDepthConsts,
                                      const llvm::DenseSet<Operation *> &allCtxForOps, Value tripCount,
                                      Value loopLowerBound, Value loopStep, const SmallVector<Value> &groupTrueFlagVals,
                                      const SmallVector<Value> &iterArgDeltas, scf::ForOp origForOp)
{
    int numGroups = static_cast<int>(groups.size());
    Block *oldBody = info.oldBody;

    llvm::DenseSet<Operation *> allChainOps;
    llvm::DenseSet<Operation *> retainedChainOpSet;
    computeConsumerCloneSets(groups, oldBody, allChainOps, retainedChainOpSet);

    MultiBufferLoopBodyState state(numGroups);
    llvm::DenseMap<Operation *, SmallVector<int>> groupsByOwnerOp;
    if (failed(collectGroupsByOwnerOp(groups, oldBody, groupsByOwnerOp))) {
        return failure();
    }

    LogicalResult bodyStatus = emitOldBodyWithMultiBufferHooks(
        builder, loc, groups, info, groupsByOwnerOp, allCtxForOps, allChainOps, retainedChainOpSet, tripCount,
        loopLowerBound, loopStep, groupTrueFlagVals, groupIndexOneVals, iterArgDeltas, origForOp,
        allGroupDepthConsts, allGroupSlotConsts, state);
    if (failed(bodyStatus)) {
        return failure();
    }

    LogicalResult unownedStatus = emitUnownedGroupsAtBodyEnd(
        builder, loc, groups, info, oldBody, tripCount, loopLowerBound, loopStep, groupTrueFlagVals, groupIndexOneVals,
        iterArgDeltas, origForOp, allGroupDepthConsts, allGroupSlotConsts, state);
    if (failed(unownedStatus)) {
        return failure();
    }

    emitCombinedYield(builder, loc, groups, info, state);
    return success();
}

} // namespace gmload
