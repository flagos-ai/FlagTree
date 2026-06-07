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

#include "ascend/include/DynamicCVPipeline/SplitDataflow/SeparateCVScope.h"

#include "bishengir/Dialect/HIVM/IR/HIVM.h"
#include "bishengir/Dialect/Scope/IR/Scope.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/Pass/Pass.h"

#include <optional>
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Debug.h"

using namespace mlir;

using namespace mlir::triton;

static constexpr const char *DEBUG_TYPE = "SeparateCVScope";
#define DBGS() (llvm::dbgs() << '[' << DEBUG_TYPE << "] ")

template <typename... Args>
static void logDebug(const Args &...args)
{
    LLVM_DEBUG({
        auto &debugStream = llvm::dbgs();
        debugStream << '[' << DEBUG_TYPE << "] ";
        (debugStream << ... << args);
        debugStream << "\n";
    });
}

static constexpr unsigned kForOpOperandPrefixCount = 3;
static constexpr unsigned kSeenValuesCapacity = 16;

static void debugDumpOperation(StringRef prefix, Operation *op)
{
    LLVM_DEBUG({
        DBGS() << prefix << "\n";
        op->print(llvm::dbgs());
        llvm::dbgs() << "\n";
    });
}

struct CoreTypeInfo {
    SmallVector<StringRef> resultTypes;

    StringRef getResultType(size_t index) const
    {
        if (index < resultTypes.size()) {
            return resultTypes[index];
        }
        return resultTypes.front();
    }

    bool hasResultForScope(StringRef scopeType) const
    {
        for (StringRef t : resultTypes) {
            if (t == scopeType) {
                return true;
            }
        }
        return false;
    }

    bool allResultsMatchScope(StringRef scopeType, unsigned numResults) const
    {
        if (numResults == 0) {
            return getResultType(0) == scopeType;
        }
        for (unsigned i = 0; i < numResults; ++i) {
            if (getResultType(i) != scopeType) {
                return false;
            }
        }
        return true;
    }
};

static std::optional<CoreTypeInfo> parseCoreTypeInfo(Operation *op)
{
    auto attr = op->getAttrOfType<StringAttr>("ssbuffer.core_type");
    if (!attr) {
        return std::nullopt;
    }

    CoreTypeInfo info;
    StringRef raw = attr.getValue();
    SmallVector<StringRef> parts;
    raw.split(parts, ',', -1, false);

    if (parts.empty()) {
        return std::nullopt;
    }

    for (auto &part : parts) {
        part = part.trim();
    }

    info.resultTypes.assign(parts.begin(), parts.end());
    return info;
}

struct ScopeMatchInfo {
    bool matches = false;
    bool allResultsMatch = false;
};

static ScopeMatchInfo getScopeMatchInfo(Operation *op, StringRef scopeType)
{
    auto info = parseCoreTypeInfo(op);
    if (!info) {
        return {};
    }

    unsigned numResults = op->getNumResults();
    bool matches = numResults == 0 ? info->getResultType(0) == scopeType : info->hasResultForScope(scopeType);
    return {matches, info->allResultsMatchScope(scopeType, numResults)};
}

static bool matchesScope(Operation *op, StringRef scopeType)
{
    return getScopeMatchInfo(op, scopeType).matches;
}

static bool isOpAlive(Operation *op)
{
    return op && op->getBlock();
}

static Value buildNeutralValue(OpBuilder &builder, Value oldOperand, Location loc, StringRef scopeType)
{
    if (!oldOperand) {
        return Value();
    }

    Type type = oldOperand.getType();
    Operation *createdOp = nullptr;

    if (auto memrefTy = dyn_cast<MemRefType>(type)) {
        createdOp = builder.create<memref::AllocOp>(loc, memrefTy);
    } else if (auto shapedTy = dyn_cast<ShapedType>(type)) {
        if (shapedTy.hasStaticShape() && !isa<MemRefType>(type)) {
            if (Attribute elemZero = builder.getZeroAttr(shapedTy.getElementType())) {
                auto zeroDense = DenseElementsAttr::get(shapedTy, elemZero);
                auto typedAttr = mlir::cast<TypedAttr>(zeroDense);
                createdOp = builder.create<arith::ConstantOp>(loc, type, typedAttr);
            }
        }
    } else if (Attribute zeroAttr = builder.getZeroAttr(type)) {
        if (auto typedZero = dyn_cast<TypedAttr>(zeroAttr)) {
            createdOp = builder.create<arith::ConstantOp>(loc, type, typedZero);
        }
    }

    return createdOp ? createdOp->getResult(0) : Value();
}

static FailureOr<std::pair<scope::ScopeOp, scope::ScopeOp>> createTwoFullScopes(func::FuncOp funcOp)
{
    if (!llvm::hasSingleElement(funcOp.getBody())) {
        logDebug("createTwoFullScopes failed for func '", funcOp.getName(), "': expected single block");
        funcOp.emitError("SeparateCVScope only supports single-block functions");
        return failure();
    }

    Block &lastBlock = funcOp.getBody().back();
    SmallVector<Operation *> opsToMove;
    for (Operation &op : lastBlock.without_terminator()) {
        opsToMove.push_back(&op);
    }

    if (opsToMove.empty()) {
        return std::make_pair(scope::ScopeOp(), scope::ScopeOp());
    }

    Operation *lastOpToMove = opsToMove.back();
    OpBuilder builder(&lastBlock, ++lastOpToMove->getIterator());

    auto vecScope = builder.create<scope::ScopeOp>(builder.getUnknownLoc(), ArrayRef<Type> {});
    vecScope.getBodyRegion().emplaceBlock();

    Block *vecBlock = &vecScope.getBodyRegion().front();
    OpBuilder vecBuilder(vecBlock, vecBlock->end());

    for (Operation *op : opsToMove) {
        op->remove();
        vecBuilder.insert(op);
    }
    vecBuilder.create<scope::ReturnOp>(builder.getUnknownLoc());

    builder.setInsertionPointAfter(vecScope);
    IRMapping mapping;
    auto cloned = builder.clone(*vecScope.getOperation(), mapping);
    auto cubeScope = cast<scope::ScopeOp>(cloned);

    auto vecAttr = hivm::TCoreTypeAttr::get(builder.getContext(), hivm::TCoreType::VECTOR);
    auto cubeAttr = hivm::TCoreTypeAttr::get(builder.getContext(), hivm::TCoreType::CUBE);

    vecScope->setAttr(hivm::TCoreTypeAttr::name, vecAttr);
    cubeScope->setAttr(hivm::TCoreTypeAttr::name, cubeAttr);

    return std::make_pair(vecScope, cubeScope);
}

static bool isPureForScope(Operation *op, StringRef scopeType)
{
    ScopeMatchInfo matchInfo = getScopeMatchInfo(op, scopeType);
    if (!matchInfo.matches || !matchInfo.allResultsMatch) {
        return false;
    }

    for (Region &region : op->getRegions()) {
        for (Block &block : region) {
            for (Operation &nested : block) {
                if (nested.hasTrait<OpTrait::IsTerminator>()) {
                    continue;
                }
                if (nested.getNumRegions() > 0) {
                    if (!isPureForScope(&nested, scopeType)) {
                        return false;
                    }
                    continue;
                }
                if (!getScopeMatchInfo(&nested, scopeType).matches) {
                    return false;
                }
            }
        }
    }
    return true;
}

enum class PendingActionKind { EraseDirectly, NormalizeControlFlow };

struct PendingAction {
    Operation *op = nullptr;
    PendingActionKind kind;
};

static bool isControlFlowOp(Operation *op)
{
    return isa<scf::ForOp, scf::IfOp, scf::WhileOp, scf::ParallelOp>(op);
}

static void collectActionForOp(Operation *op, StringRef scopeType, SmallVector<PendingAction> &actions)
{
    ScopeMatchInfo matchInfo = getScopeMatchInfo(op, scopeType);
    if (!matchInfo.matches) {
        actions.push_back({op, PendingActionKind::EraseDirectly});
    } else if (!matchInfo.allResultsMatch) {
        actions.push_back({op, PendingActionKind::NormalizeControlFlow});
    }
}

static SmallVector<PendingAction> collectActionsInRegion(Region &region, StringRef scopeType)
{
    SmallVector<PendingAction> actions;
    for (Block &block : region) {
        for (Operation &op : block) {
            if (op.hasTrait<OpTrait::IsTerminator>())
                continue;

            if (op.getNumRegions() > 0) {
                if (isControlFlowOp(&op)) {
                    if (!isPureForScope(&op, scopeType))
                        actions.push_back({&op, PendingActionKind::NormalizeControlFlow});
                } else if (!matchesScope(&op, scopeType)) {
                    actions.push_back({&op, PendingActionKind::EraseDirectly});
                }
                continue;
            }
            collectActionForOp(&op, scopeType, actions);
        }
    }
    return actions;
}

static bool hasLiveUsers(Operation *op)
{
    for (Value result : op->getResults()) {
        for (OpOperand &use : result.getUses()) {
            if (isOpAlive(use.getOwner())) {
                return true;
            }
        }
    }
    return false;
}

static bool needsLoopCarryPreserve(Operation *owner, unsigned slotIndex, StringRef scopeType);

static bool canSkipForwardingUse(OpOperand &use, StringRef scopeType)
{
    Operation *user = use.getOwner();
    auto info = parseCoreTypeInfo(user);
    if (!info) {
        return false;
    }

    if (auto forOp = dyn_cast<scf::ForOp>(user)) {
        unsigned operandIndex = use.getOperandNumber();
        if (operandIndex < kForOpOperandPrefixCount) {
            return false;
        }
        unsigned resultIndex = operandIndex - kForOpOperandPrefixCount;
        return resultIndex < forOp.getNumResults() && info->getResultType(resultIndex) != scopeType;
    }

    if (auto whileOp = dyn_cast<scf::WhileOp>(user)) {
        unsigned slotIdx = use.getOperandNumber();
        if (slotIdx >= whileOp.getNumResults()) {
            return false;
        }
        if (info->getResultType(slotIdx) == scopeType) {
            return false;
        }
        if (needsLoopCarryPreserve(whileOp, slotIdx, scopeType)) {
            return false;
        }
        return true;
    }
    return false;
}

static Operation *findLiveUser(Value value, StringRef scopeType)
{
    for (OpOperand &use : value.getUses()) {
        Operation *user = use.getOwner();
        if (isOpAlive(user) && !canSkipForwardingUse(use, scopeType)) {
            return user;
        }
    }
    return nullptr;
}

static Operation *findNonTermUser(Value value, StringRef scopeType)
{
    for (OpOperand &use : value.getUses()) {
        Operation *user = use.getOwner();
        if (isOpAlive(user) && !canSkipForwardingUse(use, scopeType) && !user->hasTrait<OpTrait::IsTerminator>()) {
            return user;
        }
    }
    return nullptr;
}

enum class UseCheckResult { Skip, Active, Continue };

static UseCheckResult checkConditionUse(OpOperand &use, Operation *owner, unsigned slotIndex,
                                        StringRef scopeType)
{
    auto conditionOp = dyn_cast<scf::ConditionOp>(use.getOwner());
    if (!conditionOp) {
        return UseCheckResult::Skip;
    }

    unsigned idx = use.getOperandNumber();
    if (conditionOp->getParentOp() != owner || idx == 0 || idx - 1 != slotIndex) {
        Operation *parentOp = conditionOp->getParentOp();
        if (parentOp && !matchesScope(parentOp, scopeType)) {
            return UseCheckResult::Continue;
        }
        return UseCheckResult::Active;
    }
    return UseCheckResult::Continue;
}

static UseCheckResult checkYieldUse(OpOperand &use, Operation *owner, unsigned slotIndex, StringRef scopeType,
                                    SmallVector<Value> &worklist)
{
    auto yieldOp = dyn_cast<scf::YieldOp>(use.getOwner());
    if (!yieldOp) {
        return UseCheckResult::Skip;
    }

    Operation *yieldOwner = yieldOp->getParentOp();
    unsigned idx = use.getOperandNumber();

    if (yieldOwner == owner) {
        if (idx != slotIndex && idx < yieldOwner->getNumResults()) {
            worklist.push_back(yieldOwner->getResult(idx));
        }
        return UseCheckResult::Continue;
    }

    if (idx < yieldOwner->getNumResults()) {
        worklist.push_back(yieldOwner->getResult(idx));
        return UseCheckResult::Continue;
    }

    if (matchesScope(yieldOwner, scopeType) || isControlFlowOp(yieldOwner)) {
        return UseCheckResult::Active;
    }
    return UseCheckResult::Continue;
}

static UseCheckResult checkGeneralUse(OpOperand &use, StringRef scopeType, SmallVector<Value> &worklist)
{
    Operation *user = use.getOwner();
    if (user->getNumResults() == 0) {
        if (matchesScope(user, scopeType) || isControlFlowOp(user)) {
            return UseCheckResult::Active;
        }
        return UseCheckResult::Continue;
    }
    for (Value result : user->getResults()) {
        worklist.push_back(result);
    }
    return UseCheckResult::Continue;
}

static bool slotHasActiveUse(Value startValue, Operation *owner, unsigned slotIndex, StringRef scopeType)
{
    SmallVector<Value> worklist {startValue};
    llvm::SmallPtrSet<void *, kSeenValuesCapacity> seenValues;

    while (!worklist.empty()) {
        Value value = worklist.pop_back_val();
        if (!value || !seenValues.insert(value.getAsOpaquePointer()).second) {
            continue;
        }

        for (OpOperand &use : value.getUses()) {
            if (!isOpAlive(use.getOwner())) {
                continue;
            }

            UseCheckResult result;
            if ((result = checkConditionUse(use, owner, slotIndex, scopeType)) != UseCheckResult::Skip ||
                (result = checkYieldUse(use, owner, slotIndex, scopeType, worklist)) != UseCheckResult::Skip ||
                (result = checkGeneralUse(use, scopeType, worklist)) != UseCheckResult::Skip) {
                if (result == UseCheckResult::Active) {
                    return true;
                }
            }
        }
    }
    return false;
}

static bool needsLoopCarryPreserve(Operation *owner, unsigned slotIndex, StringRef scopeType)
{
    if (auto forOp = dyn_cast<scf::ForOp>(owner)) {
        Block &body = forOp.getRegion().front();
        unsigned iterArgIndex = slotIndex + 1;
        if (iterArgIndex >= body.getNumArguments()) {
            return false;
        }
        return slotHasActiveUse(body.getArgument(iterArgIndex), owner, slotIndex, scopeType);
    }

    if (auto whileOp = dyn_cast<scf::WhileOp>(owner)) {
        Block &before = whileOp.getBefore().front();
        Block &after = whileOp.getAfter().front();
        if (slotIndex < before.getNumArguments() &&
            slotHasActiveUse(before.getArgument(slotIndex), owner, slotIndex, scopeType)) {
            return true;
        }
        if (slotIndex < after.getNumArguments() &&
            slotHasActiveUse(after.getArgument(slotIndex), owner, slotIndex, scopeType)) {
            return true;
        }
    }
    return false;
}

static LogicalResult neutralizeYieldInRegion(Operation *op, const CoreTypeInfo &info, StringRef scopeType, Location loc)
{
    if (op->getNumRegions() == 0) {
        return success();
    }

    for (Region &region : op->getRegions()) {
        if (region.empty()) {
            continue;
        }

        for (Block &block : region) {
            auto yieldOp = dyn_cast<scf::YieldOp>(block.getTerminator());
            if (!yieldOp) {
                continue;
            }

            OpBuilder builder(yieldOp);
            for (unsigned i = 0; i < yieldOp.getNumOperands(); ++i) {
                if (info.getResultType(i) == scopeType) {
                    continue;
                }
                if (needsLoopCarryPreserve(op, i, scopeType)) {
                    continue;
                }

                Value oldOperand = yieldOp.getOperand(i);
                if (i < op->getNumResults()) {
                    if (Operation *resultUser = findLiveUser(op->getResult(i), scopeType)) {
                        yieldOp.emitWarning() << "skip neutralizing yield operand #" << i << " for scope " << scopeType
                                              << " because parent result #" << i << " still has live user '"
                                              << resultUser->getName().getStringRef() << "'";
                        continue;
                    }
                }

                Value replacement = buildNeutralValue(builder, oldOperand, loc, scopeType);
                if (!replacement) {
                    logDebug("neutralizeYieldInRegion failed for op '", op->getName().getStringRef(),
                             "' at operand #", i, " in scope ", scopeType);
                    return failure();
                }
                yieldOp.setOperand(i, replacement);
            }
        }
    }
    return success();
}

static LogicalResult neutralizeTerminatorUses(Operation *op, const CoreTypeInfo &info, StringRef scopeType)
{
    for (unsigned i = 0; i < op->getNumResults(); ++i) {
        if (info.getResultType(i) == scopeType) {
            continue;
        }

        Value result = op->getResult(i);
        if (Operation *extraUser = findNonTermUser(result, scopeType)) {
            op->emitWarning() << "skip neutralizing result #" << i << " for scope " << scopeType
                              << " because the value still has live user '" << extraUser->getName().getStringRef()
                              << "'";
            continue;
        }

        SmallVector<OpOperand *> usesToNeutralize;
        for (OpOperand &use : llvm::make_early_inc_range(result.getUses())) {
            if (use.getOwner()->hasTrait<OpTrait::IsTerminator>()) {
                usesToNeutralize.push_back(&use);
            }
        }

        for (OpOperand *use : usesToNeutralize) {
            OpBuilder builder(use->getOwner());
            Value replacement = buildNeutralValue(builder, result, op->getLoc(), scopeType);
            if (!replacement) {
                logDebug("neutralizeTerminatorUses failed for op '", op->getName().getStringRef(),
                         "' at result #", i, " in scope ", scopeType);
                return failure();
            }
            use->set(replacement);
        }
    }
    return success();
}

static LogicalResult executeActions(SmallVector<PendingAction> &actions, StringRef scopeType);

static LogicalResult normalizeRegionOp(Operation *op, StringRef scopeType)
{
    auto infoOpt = parseCoreTypeInfo(op);
    if (!infoOpt) {
        for (Region &region : op->getRegions()) {
            SmallVector<PendingAction> nestedActions = collectActionsInRegion(region, scopeType);
            if (failed(executeActions(nestedActions, scopeType))) {
                return failure();
            }
        }
        return success();
    }

    auto info = *infoOpt;
    Location loc = op->getLoc();

    debugDumpOperation("before normalizeRegionOp", op);

    if (op->getNumRegions() > 0) {
        if (failed(neutralizeYieldInRegion(op, info, scopeType, loc))) {
            logDebug("normalizeRegionOp failed while neutralizing yields for op '",
                     op->getName().getStringRef(), "' in scope ", scopeType);
            return failure();
        }

        for (Region &region : op->getRegions()) {
            SmallVector<PendingAction> nestedActions = collectActionsInRegion(region, scopeType);
            if (failed(executeActions(nestedActions, scopeType))) {
                logDebug("normalizeRegionOp failed while executing nested actions for op '",
                         op->getName().getStringRef(), "' in scope ", scopeType);
                return failure();
            }
        }
    }

    debugDumpOperation("after normalizeRegionOp", op);
    return success();
}

static LogicalResult normalizeNonRegionOp(Operation *op, StringRef scopeType)
{
    auto infoOpt = parseCoreTypeInfo(op);
    if (!infoOpt) {
        logDebug("normalizeNonRegionOp failed for op '", op->getName().getStringRef(),
                 "': missing ssbuffer.core_type in scope ", scopeType);
        op->emitError("missing ssbuffer.core_type");
        return failure();
    }

    debugDumpOperation("before normalizeNonRegionOp", op);

    LogicalResult result = neutralizeTerminatorUses(op, *infoOpt, scopeType);
    if (succeeded(result)) {
        debugDumpOperation("after normalizeNonRegionOp", op);
    }
    return result;
}

static bool regionHasScopeContent(Region &region, StringRef scopeType)
{
    for (Block &block : region) {
        for (Operation &op : block) {
            if (op.hasTrait<OpTrait::IsTerminator>()) {
                continue;
            }

            if (op.getNumRegions() > 0) {
                if (!isControlFlowOp(&op) && matchesScope(&op, scopeType)) {
                    return true;
                }

                for (Region &subRegion : op.getRegions()) {
                    if (regionHasScopeContent(subRegion, scopeType)) {
                        return true;
                    }
                }
                continue;
            }

            if (matchesScope(&op, scopeType)) {
                return true;
            }
        }
    }
    return false;
}

static bool isNormalizedDeadShell(Operation *op, StringRef scopeType)
{
    if (!op || op->getNumRegions() == 0) {
        return false;
    }

    for (Value result : op->getResults()) {
        if (findLiveUser(result, scopeType)) {
            return false;
        }
    }

    for (Region &region : op->getRegions()) {
        if (regionHasScopeContent(region, scopeType)) {
            return false;
        }
    }
    return true;
}

static LogicalResult executeActions(SmallVector<PendingAction> &actions, StringRef scopeType)
{
    for (auto it = actions.rbegin(); it != actions.rend(); ++it) {
        Operation *op = it->op;
        if (!isOpAlive(op)) {
            continue;
        }

        switch (it->kind) {
            case PendingActionKind::EraseDirectly:
                if (!hasLiveUsers(op)) {
                    op->erase();
                }
                break;
            case PendingActionKind::NormalizeControlFlow:
                if (op->getNumRegions() > 0) {
                    if (failed(normalizeRegionOp(op, scopeType))) {
                        logDebug("failed to normalize region op '", op->getName().getStringRef(),
                                 "' for scope ", scopeType);
                        return failure();
                    }
                } else {
                    if (failed(normalizeNonRegionOp(op, scopeType))) {
                        logDebug("failed to normalize non-region op '", op->getName().getStringRef(),
                                 "' for scope ", scopeType);
                        return failure();
                    }
                }
                if (isOpAlive(op) && op->getNumRegions() > 0 && isNormalizedDeadShell(op, scopeType)) {
                    logDebug("erasing dead shell after normalize: '", op->getName().getStringRef(),
                             "' in scope ", scopeType);
                    op->erase();
                }
                break;
        }
    }
    return success();
}

static void removeSsbufferAttrs(Operation *op)
{
    op->removeAttr("ssbuffer.core_type");
}

static void cleanupSsbufferAttrs(Operation *rootOp)
{
    removeSsbufferAttrs(rootOp);
    rootOp->walk([](Operation *op) { removeSsbufferAttrs(op); });
}

static LogicalResult separateScopes(func::FuncOp funcOp)
{
    debugDumpOperation("before SeparateCVScope on func", funcOp.getOperation());

    FailureOr<std::pair<scope::ScopeOp, scope::ScopeOp>> scopes = createTwoFullScopes(funcOp);
    if (failed(scopes)) {
        logDebug("SeparateCVScope failed to create VECTOR/CUBE scopes for func '", funcOp.getName(), "'");
        return failure();
    }

    auto vecScope = scopes->first;
    auto cubeScope = scopes->second;

    auto vecActions = collectActionsInRegion(vecScope.getRegion(), "VECTOR");
    auto cubeActions = collectActionsInRegion(cubeScope.getRegion(), "CUBE");
    if (failed(executeActions(vecActions, "VECTOR")) || failed(executeActions(cubeActions, "CUBE"))) {
        logDebug("SeparateCVScope failed while executing actions for func '", funcOp.getName(), "'");
        return failure();
    }

    cleanupSsbufferAttrs(funcOp);

    debugDumpOperation("after SeparateCVScope on func", funcOp.getOperation());
    return success();
}

// Declare dependent dialects
void mlir::triton::SeparateCVScopePass::getDependentDialects(DialectRegistry &registry) const
{
    registry.insert<arith::ArithDialect, hivm::HIVMDialect, memref::MemRefDialect, scope::ScopeDialect>();
}

void mlir::triton::SeparateCVScopePass::runOnOperation()
{
    auto module = getOperation();
    SmallVector<func::FuncOp> funcOps;
    module.walk([&](func::FuncOp funcOp) { funcOps.push_back(funcOp); });

    debugDumpOperation("before SeparateCVScopePass", module.getOperation());

    for (auto funcOp : funcOps) {
        if (funcOp.getBody().empty()) {
            continue;
        }
        if (failed(separateScopes(funcOp))) {
            logDebug("SeparateCVScopePass failed on func '", funcOp.getName(), "'");
            signalPassFailure();
            return;
        }
    }

    debugDumpOperation("after SeparateCVScopePass", module.getOperation());
}

namespace mlir {
namespace triton {

std::unique_ptr<OperationPass<ModuleOp>> createSeparateCVScopePass()
{
    return std::make_unique<SeparateCVScopePass>();
}

void registerSeparateCVScopePasses()
{
    registerPass([]() -> std::unique_ptr<mlir::Pass> { return createSeparateCVScopePass(); });
}

} // namespace triton
} // namespace mlir