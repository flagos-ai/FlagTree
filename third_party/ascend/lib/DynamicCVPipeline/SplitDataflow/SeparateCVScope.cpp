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

#include <optional>

#include <map>

#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Debug.h"

#include "ascend/include/DynamicCVPipeline/Common/SSBufferManager.h"
#include "ascend/include/DynamicCVPipeline/Common/Utils.h"
#include "ascend/include/DynamicCVPipeline/SplitDataflow/SeparateCVScope.h"
#include "bishengir/Dialect/Annotation/IR/Annotation.h"
#include "bishengir/Dialect/HIVM/IR/HIVM.h"
#include "bishengir/Dialect/Scope/IR/Scope.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/Interfaces/LoopLikeInterface.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/RegionUtils.h"

using namespace mlir;

using namespace mlir::triton;

static constexpr const char *DEBUG_TYPE = "SeparateCVScope";
#define DBGS() (llvm::dbgs() << '[' << DEBUG_TYPE << "] ")

template <typename... Args> static void logDebug(const Args &...args) {
  LLVM_DEBUG({
    auto &debugStream = llvm::dbgs();
    debugStream << '[' << DEBUG_TYPE << "] ";
    (debugStream << ... << args);
    debugStream << "\n";
  });
}

static constexpr unsigned kForOpOperandPrefixCount = 3;
static constexpr unsigned kSeenValuesCapacity = 16;

static void debugDumpOperation(StringRef prefix, Operation *op) {
  LLVM_DEBUG({
    DBGS() << prefix << "\n";
    op->print(llvm::dbgs());
    llvm::dbgs() << "\n";
  });
}

static bool needsLoopCarryPreserve(Operation *owner, unsigned slotIndex,
                                   StringRef scopeType);
static bool regionHasScopeContent(Region &region, StringRef scopeType);

struct CoreTypeInfo {
  SmallVector<StringRef> resultTypes;

  StringRef getResultType(size_t index) const {
    if (index < resultTypes.size()) {
      return resultTypes[index];
    }
    return resultTypes.front();
  }

  bool hasResultForScope(StringRef scopeType) const {
    for (StringRef t : resultTypes) {
      if (t == scopeType) {
        return true;
      }
    }
    return false;
  }

  bool allResultsMatchScope(StringRef scopeType, unsigned numResults) const {
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

static std::optional<CoreTypeInfo> parseCoreTypeInfo(Operation *op) {
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

static ScopeMatchInfo getScopeMatchInfo(Operation *op, StringRef scopeType) {
  auto info = parseCoreTypeInfo(op);
  if (!info) {
    return {};
  }

  unsigned numResults = op->getNumResults();
  bool matches = numResults == 0 ? info->getResultType(0) == scopeType
                                 : info->hasResultForScope(scopeType);
  return {matches, info->allResultsMatchScope(scopeType, numResults)};
}

static bool matchesScope(Operation *op, StringRef scopeType) {
  return getScopeMatchInfo(op, scopeType).matches;
}

static Value buildNeutralValue(OpBuilder &builder, Value oldOperand,
                               Location loc, StringRef scopeType) {
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

static FailureOr<std::pair<scope::ScopeOp, scope::ScopeOp>>
createTwoFullScopes(func::FuncOp funcOp) {
  if (!llvm::hasSingleElement(funcOp.getBody())) {
    logDebug("createTwoFullScopes failed for func '", funcOp.getName(),
             "': expected single block");
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

  auto vecScope =
      builder.create<scope::ScopeOp>(builder.getUnknownLoc(), ArrayRef<Type>{});
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

  auto vecAttr =
      hivm::TCoreTypeAttr::get(builder.getContext(), hivm::TCoreType::VECTOR);
  auto cubeAttr =
      hivm::TCoreTypeAttr::get(builder.getContext(), hivm::TCoreType::CUBE);

  vecScope->setAttr(hivm::TCoreTypeAttr::name, vecAttr);
  cubeScope->setAttr(hivm::TCoreTypeAttr::name, cubeAttr);

  return std::make_pair(vecScope, cubeScope);
}

static bool isPureForScope(Operation *op, StringRef scopeType) {
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

static bool isControlFlowOp(Operation *op) {
  return isa<scf::ForOp, scf::IfOp, scf::WhileOp, scf::ParallelOp>(op);
}

static void collectActionForOp(Operation *op, StringRef scopeType,
                               SmallVector<PendingAction> &actions) {
  ScopeMatchInfo matchInfo = getScopeMatchInfo(op, scopeType);
  if (!matchInfo.matches) {
    actions.push_back({op, PendingActionKind::EraseDirectly});
  } else if (!matchInfo.allResultsMatch) {
    actions.push_back({op, PendingActionKind::NormalizeControlFlow});
  }
}

static SmallVector<PendingAction> collectActionsInRegion(Region &region,
                                                         StringRef scopeType) {
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

static bool hasLiveUsers(Operation *op) {
  for (Value result : op->getResults()) {
    for (OpOperand &use : result.getUses()) {
      if (use.getOwner()) {
        return true;
      }
    }
  }
  return false;
}

// Check whether this use maps to a loop-carried slot that must stay live in the
// target scope.
static bool hasActiveNestedLoopCarryUse(OpOperand &use, StringRef scopeType) {
  auto loopOp = dyn_cast<LoopLikeOpInterface>(use.getOwner());
  if (!loopOp) {
    return false;
  }

  auto inits = loopOp.getInitsMutable();
  if (inits.empty()) {
    return false;
  }

  unsigned operandIndex = use.getOperandNumber();
  unsigned initsBegin = inits.front().getOperandNumber();
  unsigned initsEnd = initsBegin + inits.size();
  if (operandIndex < initsBegin || operandIndex >= initsEnd) {
    return false;
  }

  unsigned resultIndex = operandIndex - initsBegin;
  return resultIndex < loopOp->getNumResults() &&
         needsLoopCarryPreserve(loopOp.getOperation(), resultIndex, scopeType);
}

// Skip pure forwarding uses that only feed out-of-scope loop result slots.
static bool canSkipForwardingUse(OpOperand &use, StringRef scopeType) {
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
    return resultIndex < forOp.getNumResults() &&
           info->getResultType(resultIndex) != scopeType &&
           !needsLoopCarryPreserve(forOp, resultIndex, scopeType);
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
    return llvm::none_of(whileOp->getUsers(), [&](Operation *user) {
      return matchesScope(user, scopeType);
    });
  }
  return false;
}

// Preserve yield-to-result tracking only when loop-carry evidence and
// mixed-scope sibling results require it.
static bool shouldPreserveFromYield(Operation *yieldOwner,
                                    unsigned operandIndex,
                                    StringRef scopeType) {
  bool preserveLoopCarry =
      needsLoopCarryPreserve(yieldOwner, operandIndex, scopeType);
  if (!preserveLoopCarry) {
    return false;
  }

  auto loopOp = dyn_cast<LoopLikeOpInterface>(yieldOwner);
  if (!loopOp) {
    return preserveLoopCarry;
  }

  auto ownerInfo = parseCoreTypeInfo(yieldOwner);
  if (!ownerInfo) {
    return preserveLoopCarry;
  }

  unsigned numResults = yieldOwner->getNumResults();
  return llvm::any_of(
      llvm::seq<unsigned>(0, numResults), [&](unsigned resultIndex) {
        return resultIndex != operandIndex &&
               ownerInfo->getResultType(resultIndex) == scopeType;
      });
}

// Return `user` as a relevant result, honoring the ignoreTerminators flag.
static Operation *acceptUser(Operation *user, bool ignoreTerminators) {
  if (!user || !user->getBlock()) {
    return nullptr;
  }
  if (!ignoreTerminators || !user->hasTrait<OpTrait::IsTerminator>()) {
    return user;
  }
  return nullptr;
}

// Handle a scf.yield use: follow result slots transitively or mark the owner
// live.
static Operation *handleYieldUse(scf::YieldOp yieldOp, OpOperand &use,
                                 StringRef scopeType, bool ignoreTerminators,
                                 SmallVector<Value> &worklist) {
  Operation *yieldOwner = yieldOp->getParentOp();
  unsigned operandIndex = use.getOperandNumber();

  if (operandIndex < yieldOwner->getNumResults()) {
    if (shouldPreserveFromYield(yieldOwner, operandIndex, scopeType)) {
      return yieldOwner;
    }
    worklist.push_back(yieldOwner->getResult(operandIndex));
    return nullptr;
  }

  if (matchesScope(yieldOwner, scopeType) || isControlFlowOp(yieldOwner)) {
    return acceptUser(yieldOwner, ignoreTerminators);
  }
  return nullptr;
}

// True when `op` holds scope content in any of its regions.
static bool controlFlowOpHasScopeContent(Operation *op, StringRef scopeType) {
  for (Region &region : op->getRegions()) {
    if (regionHasScopeContent(region, scopeType)) {
      return true;
    }
  }
  return false;
}

// Returns whether a use is a control-flow gating operand (for/parallel bounds
// or if condition).
static bool isControlFlowGatingUse(OpOperand &use) {
  Operation *user = use.getOwner();
  unsigned operandIndex = use.getOperandNumber();

  if (isa<scf::ForOp>(user)) {
    return operandIndex < kForOpOperandPrefixCount;
  }
  if (auto parallelOp = dyn_cast<scf::ParallelOp>(user)) {
    return operandIndex < kForOpOperandPrefixCount * parallelOp.getNumLoops();
  }
  if (isa<scf::IfOp>(user)) {
    return operandIndex == 0;
  }
  return false;
}

// Classify a single use: return a scope-relevant user, or enqueue transitive
// values.
static Operation *classifyScopeRelevantUse(OpOperand &use, StringRef scopeType,
                                           bool ignoreTerminators,
                                           SmallVector<Value> &worklist) {
  Operation *user = use.getOwner();
  if (!user || !user->getBlock() || canSkipForwardingUse(use, scopeType)) {
    return nullptr;
  }

  if (hasActiveNestedLoopCarryUse(use, scopeType)) {
    return acceptUser(user, ignoreTerminators);
  }

  if (auto conditionOp = dyn_cast<scf::ConditionOp>(user)) {
    Operation *parentOp = conditionOp.getParentOp();
    if (parentOp && matchesScope(parentOp, scopeType)) {
      return acceptUser(parentOp, ignoreTerminators);
    }
    if (parentOp && use.getOperandNumber() == 0 &&
        controlFlowOpHasScopeContent(parentOp, scopeType)) {
      return acceptUser(parentOp, ignoreTerminators);
    }
    return nullptr;
  }

  if (auto yieldOp = dyn_cast<scf::YieldOp>(user)) {
    return handleYieldUse(yieldOp, use, scopeType, ignoreTerminators, worklist);
  }

  if (matchesScope(user, scopeType)) {
    return acceptUser(user, ignoreTerminators);
  }

  if (user->getNumResults() == 0) {
    if (isControlFlowOp(user)) {
      return acceptUser(user, ignoreTerminators);
    }
    return nullptr;
  }

  if (isControlFlowGatingUse(use) &&
      controlFlowOpHasScopeContent(user, scopeType)) {
    return acceptUser(user, ignoreTerminators);
  }

  for (Value result : user->getResults()) {
    worklist.push_back(result);
  }
  return nullptr;
}

static Operation *findScopeRelevantUser(Value startValue, StringRef scopeType,
                                        bool ignoreTerminators) {
  SmallVector<Value> worklist{startValue};
  llvm::SmallPtrSet<void *, kSeenValuesCapacity> seenValues;

  while (!worklist.empty()) {
    Value value = worklist.pop_back_val();
    if (!value || !seenValues.insert(value.getAsOpaquePointer()).second) {
      continue;
    }

    for (OpOperand &use : value.getUses()) {
      if (Operation *found = classifyScopeRelevantUse(
              use, scopeType, ignoreTerminators, worklist)) {
        return found;
      }
    }
  }
  return nullptr;
}

static Operation *findLiveUser(Value value, StringRef scopeType) {
  return findScopeRelevantUser(value, scopeType, false);
}

static Operation *findNonTermUser(Value value, StringRef scopeType) {
  return findScopeRelevantUser(value, scopeType, true);
}

enum class UseCheckResult { Skip, Active, Continue };

static UseCheckResult checkConditionUse(OpOperand &use, Operation *owner,
                                        unsigned slotIndex,
                                        StringRef scopeType) {
  auto conditionOp = dyn_cast<scf::ConditionOp>(use.getOwner());
  if (!conditionOp) {
    return UseCheckResult::Skip;
  }

  unsigned idx = use.getOperandNumber();
  if (conditionOp->getParentOp() != owner || idx == 0 || idx != slotIndex + 1) {
    Operation *parentOp = conditionOp->getParentOp();
    if (parentOp && idx == 0 &&
        controlFlowOpHasScopeContent(parentOp, scopeType)) {
      return UseCheckResult::Active;
    }
    if (parentOp && !matchesScope(parentOp, scopeType)) {
      return UseCheckResult::Continue;
    }
    return UseCheckResult::Active;
  }
  return UseCheckResult::Continue;
}

static UseCheckResult checkYieldUse(OpOperand &use, Operation *owner,
                                    unsigned slotIndex, StringRef scopeType,
                                    SmallVector<Value> &worklist) {
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

static UseCheckResult checkGeneralUse(OpOperand &use, StringRef scopeType,
                                      SmallVector<Value> &worklist) {
  Operation *user = use.getOwner();
  if (hasActiveNestedLoopCarryUse(use, scopeType)) {
    return UseCheckResult::Active;
  }

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

static bool slotHasActiveUse(Value startValue, Operation *owner,
                             unsigned slotIndex, StringRef scopeType) {
  SmallVector<Value> worklist{startValue};
  llvm::SmallPtrSet<void *, kSeenValuesCapacity> seenValues;

  while (!worklist.empty()) {
    Value value = worklist.pop_back_val();
    if (!value || !seenValues.insert(value.getAsOpaquePointer()).second) {
      continue;
    }

    for (OpOperand &use : value.getUses()) {
      if (!use.getOwner()) {
        continue;
      }

      UseCheckResult result;
      if ((result = checkConditionUse(use, owner, slotIndex, scopeType)) !=
              UseCheckResult::Skip ||
          (result = checkYieldUse(use, owner, slotIndex, scopeType,
                                  worklist)) != UseCheckResult::Skip ||
          (result = checkGeneralUse(use, scopeType, worklist)) !=
              UseCheckResult::Skip) {
        if (result == UseCheckResult::Active) {
          return true;
        }
      }
    }
  }
  return false;
}

static bool needsLoopCarryPreserve(Operation *owner, unsigned slotIndex,
                                   StringRef scopeType) {
  if (auto forOp = dyn_cast<scf::ForOp>(owner)) {
    Block &body = forOp.getRegion().front();
    unsigned iterArgIndex = slotIndex + 1;
    if (iterArgIndex >= body.getNumArguments()) {
      return false;
    }
    return slotHasActiveUse(body.getArgument(iterArgIndex), owner, slotIndex,
                            scopeType);
  }

  if (auto whileOp = dyn_cast<scf::WhileOp>(owner)) {
    Block &before = whileOp.getBefore().front();
    Block &after = whileOp.getAfter().front();
    if (slotIndex < before.getNumArguments() &&
        slotHasActiveUse(before.getArgument(slotIndex), owner, slotIndex,
                         scopeType)) {
      return true;
    }
    if (slotIndex < after.getNumArguments() &&
        slotHasActiveUse(after.getArgument(slotIndex), owner, slotIndex,
                         scopeType)) {
      return true;
    }
  }
  return false;
}

// Returns true when the producer op has no result belonging to scopeType .
static bool isProducedByForeignScope(Value operand, StringRef scopeType) {
  Operation *producer = operand.getDefiningOp();
  if (!producer || !producer->hasAttr(CVPipeline::kCoreType)) {
    return false;
  }

  // Only treat as foreign when the producer has NO result in the current scope.
  return !matchesScope(producer, scopeType);
}

static LogicalResult
neutralizeCarriedTerminatorOperand(Operation *op, const CoreTypeInfo &info,
                                   StringRef scopeType, Location loc,
                                   unsigned slotIndex, OpOperand &operand) {
  if (info.getResultType(slotIndex) == scopeType) {
    return success();
  }

  // Preserve values read by an in-loop consumer through an iter_arg.
  if (needsLoopCarryPreserve(op, slotIndex, scopeType)) {
    return success();
  }

  Value oldOperand = operand.get();

  // A foreign-scope producer must not be kept alive solely by the parent
  // result-user check.
  bool isLoopOp = isa<scf::ForOp, scf::WhileOp>(op);
  if ((!isLoopOp || !isProducedByForeignScope(oldOperand, scopeType)) &&
      slotIndex < op->getNumResults()) {
    if (Operation *resultUser =
            findLiveUser(op->getResult(slotIndex), scopeType)) {
      logDebug("skip neutralizing carried operand #", slotIndex, " for scope ",
               scopeType, " because parent result #", slotIndex,
               " still has live user '", resultUser->getName().getStringRef(),
               "'");
      return success();
    }
  }

  OpBuilder builder(operand.getOwner());
  Value replacement = buildNeutralValue(builder, oldOperand, loc, scopeType);
  if (!replacement) {
    logDebug("neutralizeRegionTerminators failed for op '",
             op->getName().getStringRef(), "' at carried operand #", slotIndex,
             " in scope ", scopeType);
    return failure();
  }
  operand.set(replacement);
  return success();
}

static LogicalResult neutralizeRegionTerminators(Operation *op,
                                                 const CoreTypeInfo &info,
                                                 StringRef scopeType,
                                                 Location loc) {
  if (op->getNumRegions() == 0) {
    return success();
  }

  for (Region &region : op->getRegions()) {
    if (region.empty()) {
      continue;
    }

    for (Block &block : region) {
      Operation *terminator = block.getTerminator();
      unsigned carriedOperandOffset = 0;
      if (isa<scf::ConditionOp>(terminator)) {
        // Operand 0 is the while condition; carried args start at operand 1.
        carriedOperandOffset = 1;
      } else if (!isa<scf::YieldOp>(terminator)) {
        continue;
      }

      unsigned numCarriedOperands =
          terminator->getNumOperands() - carriedOperandOffset;
      for (unsigned i = 0; i < numCarriedOperands; ++i) {
        if (failed(neutralizeCarriedTerminatorOperand(
                op, info, scopeType, loc, i,
                terminator->getOpOperand(i + carriedOperandOffset)))) {
          return failure();
        }
      }
    }
  }
  return success();
}

static LogicalResult neutralizeTerminatorUses(Operation *op,
                                              const CoreTypeInfo &info,
                                              StringRef scopeType) {
  for (unsigned i = 0; i < op->getNumResults(); ++i) {
    if (info.getResultType(i) == scopeType) {
      continue;
    }

    Value result = op->getResult(i);
    if (Operation *extraUser = findNonTermUser(result, scopeType)) {
      logDebug("skip neutralizing result #", i, " for scope ", scopeType,
               " because the value still has live user '",
               extraUser->getName().getStringRef(), "'");
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
      Value replacement =
          buildNeutralValue(builder, result, op->getLoc(), scopeType);
      if (!replacement) {
        logDebug("neutralizeTerminatorUses failed for op '",
                 op->getName().getStringRef(), "' at result #", i, " in scope ",
                 scopeType);
        return failure();
      }
      use->set(replacement);
    }
  }
  return success();
}

static LogicalResult executeActions(SmallVector<PendingAction> &actions,
                                    StringRef scopeType);

static Operation *findLiveResultUser(Operation *op, StringRef scopeType) {
  for (Value result : op->getResults()) {
    if (Operation *user = findLiveUser(result, scopeType)) {
      return user;
    }
  }
  return nullptr;
}

static LogicalResult normalizeRegionOp(Operation *op, StringRef scopeType) {
  auto infoOpt = parseCoreTypeInfo(op);
  if (!infoOpt) {
    for (Region &region : op->getRegions()) {
      SmallVector<PendingAction> nestedActions =
          collectActionsInRegion(region, scopeType);
      if (failed(executeActions(nestedActions, scopeType))) {
        return failure();
      }
    }
    return success();
  }

  auto info = *infoOpt;
  Location loc = op->getLoc();

  debugDumpOperation("before normalizeRegionOp", op);

  // Keep a loop intact when its body has no op for this scope but its results
  // are still consumed here; neutralizing carried values would change them.
  if (isa<scf::ForOp, scf::WhileOp>(op) &&
      !controlFlowOpHasScopeContent(op, scopeType)) {
    if (Operation *resultUser = findLiveResultUser(op, scopeType)) {
      logDebug("preserving complete loop '", op->getName().getStringRef(),
               "' in scope ", scopeType,
               " because its result still has live user '",
               resultUser->getName().getStringRef(), "'");
      return success();
    }
  }

  if (op->getNumRegions() > 0) {
    if (failed(neutralizeRegionTerminators(op, info, scopeType, loc))) {
      logDebug(
          "normalizeRegionOp failed while neutralizing terminators for op '",
          op->getName().getStringRef(), "' in scope ", scopeType);
      return failure();
    }

    for (Region &region : op->getRegions()) {
      SmallVector<PendingAction> nestedActions =
          collectActionsInRegion(region, scopeType);
      if (failed(executeActions(nestedActions, scopeType))) {
        logDebug(
            "normalizeRegionOp failed while executing nested actions for op '",
            op->getName().getStringRef(), "' in scope ", scopeType);
        return failure();
      }
    }
  }

  debugDumpOperation("after normalizeRegionOp", op);
  return success();
}

static LogicalResult normalizeNonRegionOp(Operation *op, StringRef scopeType) {
  auto infoOpt = parseCoreTypeInfo(op);
  if (!infoOpt) {
    logDebug("normalizeNonRegionOp failed for op '",
             op->getName().getStringRef(),
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

static bool regionHasScopeContent(Region &region, StringRef scopeType) {
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

static bool isNormalizedDeadShell(Operation *op, StringRef scopeType) {
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

static LogicalResult executeActions(SmallVector<PendingAction> &actions,
                                    StringRef scopeType) {
  for (auto it = actions.rbegin(); it != actions.rend(); ++it) {
    Operation *op = it->op;
    if (!op) {
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
          logDebug("failed to normalize region op '",
                   op->getName().getStringRef(), "' for scope ", scopeType);
          return failure();
        }
      } else {
        if (failed(normalizeNonRegionOp(op, scopeType))) {
          logDebug("failed to normalize non-region op '",
                   op->getName().getStringRef(), "' for scope ", scopeType);
          return failure();
        }
      }
      if (op && op->getNumRegions() > 0 &&
          isNormalizedDeadShell(op, scopeType)) {
        if (hasLiveUsers(op)) {
          logDebug("preserving normalized shell '",
                   op->getName().getStringRef(), "' in scope ", scopeType,
                   " because it still has structural users");
        } else {
          logDebug("erasing dead shell after normalize: '",
                   op->getName().getStringRef(), "' in scope ", scopeType);
          op->erase();
        }
      }
      break;
    }
  }
  return success();
}

static void removeSsbufferAttrs(Operation *op) {
  op->removeAttr("ssbuffer.core_type");
}

static void cleanupSsbufferAttrs(Operation *rootOp) {
  removeSsbufferAttrs(rootOp);
  rootOp->walk([](Operation *op) { removeSsbufferAttrs(op); });
}

static LogicalResult separateScopes(func::FuncOp funcOp) {
  debugDumpOperation("before SeparateCVScope on func", funcOp.getOperation());

  FailureOr<std::pair<scope::ScopeOp, scope::ScopeOp>> scopes =
      createTwoFullScopes(funcOp);
  if (failed(scopes)) {
    logDebug("SeparateCVScope failed to create VECTOR/CUBE scopes for func '",
             funcOp.getName(), "'");
    return failure();
  }

  auto vecScope = scopes->first;
  auto cubeScope = scopes->second;

  auto vecActions = collectActionsInRegion(vecScope.getRegion(), "VECTOR");
  auto cubeActions = collectActionsInRegion(cubeScope.getRegion(), "CUBE");
  if (failed(executeActions(vecActions, "VECTOR")) ||
      failed(executeActions(cubeActions, "CUBE"))) {
    logDebug("SeparateCVScope failed while executing actions for func '",
             funcOp.getName(), "'");
    return failure();
  }

  cleanupSsbufferAttrs(funcOp);

  debugDumpOperation("after SeparateCVScope on func", funcOp.getOperation());
  return success();
}

// Mechanism A: in VECTOR scopes, replace each redundant store→load pair that
// shares an ssbuffer.transfer_id with the stored value, then erase the load.
//
// The scope-clone step duplicates the CUBE-side scalar load into the VECTOR
// scope too; there it reads back a value this scope itself just stored, so the
// load is pure redundancy (not dead code — it has users), and DCE will not
// remove it.
static void replaceRedundantVectorStoreLoad(scope::ScopeOp scopeOp) {
  auto coreTypeAttr =
      scopeOp->getAttrOfType<hivm::TCoreTypeAttr>(hivm::TCoreTypeAttr::name);
  if (!coreTypeAttr || coreTypeAttr.getTcoretype() != hivm::TCoreType::VECTOR) {
    return;
  }

  // Pass 1: collect the value each ssbuffer.transfer_id stores.
  llvm::DenseMap<int64_t, mlir::Value> storedValues;
  scopeOp.walk([&](memref::StoreOp storeOp) {
    auto transferIdAttr =
        storeOp->getAttrOfType<mlir::IntegerAttr>(CVPipeline::kTransferId);
    if (!transferIdAttr) {
      return;
    }
    int64_t tid = transferIdAttr.getInt();
    storedValues[tid] = storeOp.getValue();
  });

  if (storedValues.empty()) {
    return;
  }

  // Pass 2: replace loads of the same transfer_id with the stored value.
  llvm::SmallVector<memref::LoadOp> deadLoads;
  scopeOp.walk([&](memref::LoadOp loadOp) {
    auto transferIdAttr =
        loadOp->getAttrOfType<mlir::IntegerAttr>(CVPipeline::kTransferId);
    if (!transferIdAttr) {
      return;
    }
    int64_t tid = transferIdAttr.getInt();
    auto it = storedValues.find(tid);
    if (it == storedValues.end()) {
      return;
    }
    mlir::Value storeVal = it->second;
    if (storeVal == loadOp.getResult()) {
      return;
    }
    // Drop the volatile annotation the scalar read attached to this load;
    // it only makes sense while the load survives in the CUBE scope.
    for (Operation *user :
         llvm::make_early_inc_range(loadOp.getResult().getUsers())) {
      if (isa<annotation::MarkOp>(user) && user->hasAttr(kMemrefExtVolatile)) {
        user->erase();
      }
    }
    loadOp.replaceAllUsesWith(storeVal);
    deadLoads.push_back(loadOp);
  });
  for (memref::LoadOp loadOp : deadLoads) {
    loadOp->erase();
  }
}

// Declare dependent dialects
void mlir::triton::SeparateCVScopePass::getDependentDialects(
    DialectRegistry &registry) const {
  registry
      .insert<annotation::AnnotationDialect, arith::ArithDialect,
              hivm::HIVMDialect, memref::MemRefDialect, scope::ScopeDialect>();
}

void mlir::triton::SeparateCVScopePass::runOnOperation() {
  auto module = getOperation();

  if (CVPipeline::hasFallbackAttr(module)) {
    return;
  }

  SmallVector<func::FuncOp> funcOps;
  module.walk([&](func::FuncOp funcOp) { funcOps.push_back(funcOp); });

  debugDumpOperation("before SeparateCVScopePass", module.getOperation());

  for (auto funcOp : funcOps) {
    if (funcOp.getBody().empty()) {
      continue;
    }
    if (failed(separateScopes(funcOp))) {
      logDebug("SeparateCVScopePass failed on func '", funcOp.getName(), "'");
      CVPipeline::setFallbackAttr(module, CVPipeline::ERRCODE_FAILED);
      return;
    }
  }

  module.walk([](scope::ScopeOp scopeOp) {
    scopeOp->setAttr(CVPipeline::kHIVMMatmulLimitedInCubeAttr,
                     UnitAttr::get(scopeOp->getContext()));
  });

  // Mechanism A: clean up redundant VECTOR-side store→load pairs.
  module.walk(
      [](scope::ScopeOp scopeOp) { replaceRedundantVectorStoreLoad(scopeOp); });

  debugDumpOperation("after SeparateCVScopePass", module.getOperation());
}

namespace mlir {
namespace triton {

std::unique_ptr<OperationPass<ModuleOp>> createSeparateCVScopePass() {
  return std::make_unique<SeparateCVScopePass>();
}

void registerSeparateCVScopePasses() {
  registerPass([]() -> std::unique_ptr<mlir::Pass> {
    return createSeparateCVScopePass();
  });
}

} // namespace triton
} // namespace mlir
