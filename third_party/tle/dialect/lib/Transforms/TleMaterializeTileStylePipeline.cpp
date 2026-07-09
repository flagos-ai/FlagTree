// MIT License
//
// Copyright (c) 2025 The FlagOS Contributors
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.

#include "tle/dialect/include/IR/Dialect.h"
#include "tle/dialect/include/Transforms/Passes.h"
#include "tle/dialect/include/Transforms/TransformAttrs.h"

#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/IRMapping.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/Transforms/PipeliningUtility.h"
#include "triton/Dialect/TritonNvidiaGPU/IR/Dialect.h"
#include "llvm/ADT/DenseSet.h"

namespace mlir::triton::tle {

#define GEN_PASS_DEF_TRITONTLEMATERIALIZETILESTYLEPIPELINE
#include "tle/dialect/include/Transforms/Passes.h.inc"

namespace {

namespace tt = mlir::triton;
namespace ttg = mlir::triton::gpu;
namespace ttng = mlir::triton::nvidia_gpu;

static constexpr llvm::StringLiteral
    kTleTileStylePipelineAttr("tle.tile_style_pipeline");
static constexpr llvm::StringLiteral
    kTleExplicitTileStylePipelineAttr("tle.explicit_tile_style_pipeline");
static constexpr llvm::StringLiteral
    kTleAsyncTileProducerCountAttr("tle.async_tile_producer_count");
static constexpr bool kEnableExperimentalOneStepLeadMaterialization = true;

struct AsyncTileProducerGroup {
  Value baseMemDesc;
  tt::LoadOp loadOp;
  ttg::LocalAllocOp allocOp;
  SmallVector<ttg::AsyncCopyGlobalToLocalOp> asyncCopyOps;

  bool isLegacyLoadAlloc() const { return loadOp && allocOp; }
  bool isDirectAsyncFamily() const { return !asyncCopyOps.empty(); }
};

struct TileStyleLoopAnalysis {
  SmallVector<AsyncTileProducerGroup> asyncTileProducers;
};

struct ExplicitPipelinePlan {
  SmallVector<AsyncTileProducerGroup> producerGroups;
  SmallVector<Operation *> prefixOps;
  SmallVector<Operation *> restOps;
  SmallVector<Value> carriedValues;
  Operation *firstDotOp = nullptr;
};

struct PrefixMaterializationResult {
  SmallVector<Value> carriedValues;
  SmallVector<Value> tokens;
};

static bool isProducerAllocResult(ArrayRef<AsyncTileProducerGroup> groups,
                                  Value value) {
  return llvm::any_of(groups, [&](AsyncTileProducerGroup group) {
    return group.allocOp && group.allocOp.getResult() == value;
  });
}

static bool isProducerBaseMemDesc(ArrayRef<AsyncTileProducerGroup> groups,
                                  Value value) {
  return llvm::any_of(groups, [&](AsyncTileProducerGroup group) {
    return group.baseMemDesc == value;
  });
}

static bool isProducerLoadOp(ArrayRef<AsyncTileProducerGroup> groups,
                             Operation *op) {
  return llvm::any_of(groups, [&](AsyncTileProducerGroup group) {
    return group.loadOp && group.loadOp.getOperation() == op;
  });
}

static bool isProducerAllocOp(ArrayRef<AsyncTileProducerGroup> groups,
                              Operation *op) {
  return llvm::any_of(groups, [&](AsyncTileProducerGroup group) {
    return group.allocOp && group.allocOp.getOperation() == op;
  });
}

static bool isProducerAsyncCopyOp(ArrayRef<AsyncTileProducerGroup> groups,
                                  Operation *op) {
  return llvm::any_of(groups, [&](AsyncTileProducerGroup group) {
    return llvm::any_of(group.asyncCopyOps,
                        [&](ttg::AsyncCopyGlobalToLocalOp copy) {
                          return copy.getOperation() == op;
                        });
  });
}

static bool isTleDirectAsyncProducer(ttg::AsyncCopyGlobalToLocalOp asyncCopy) {
  return asyncCopy && asyncCopy->hasAttr(kTleLocalPointerAsyncStoreAttr);
}

static std::optional<int64_t> getLoopStage(Operation *op) {
  if (auto attr = op->getAttrOfType<IntegerAttr>(tt::kLoopStageAttrName))
    return attr.getInt();
  return std::nullopt;
}

static bool isStageZero(Operation *op) {
  std::optional<int64_t> stage = getLoopStage(op);
  return stage && *stage == 0;
}

static ttg::MemDescType ensureMutableMemDescType(ttg::MemDescType type) {
  if (type.getMutableMemory())
    return type;
  return ttg::MemDescType::get(type.getShape(), type.getElementType(),
                               type.getEncoding(), type.getMemorySpace(),
                               /*mutableMemory=*/true, type.getAllocShape());
}

static bool hasAnyLoopStageAttr(scf::ForOp forOp) {
  return llvm::any_of(
      forOp.getBody()->without_terminator(),
      [&](Operation &op) { return op.hasAttr(tt::kLoopStageAttrName); });
}

static ttg::LocalAllocOp getSingleDirectLocalAllocUser(tt::LoadOp loadOp) {
  ttg::LocalAllocOp localAlloc;
  for (Operation *user : loadOp->getUsers()) {
    auto allocOp = dyn_cast<ttg::LocalAllocOp>(user);
    if (!allocOp || allocOp.getSrc() != loadOp.getResult())
      return ttg::LocalAllocOp();
    if (localAlloc)
      return ttg::LocalAllocOp();
    localAlloc = allocOp;
  }
  return localAlloc;
}

static Value stripProducerMemDescViews(Value value) {
  Value current = value;
  while (true) {
    if (auto subslice = current.getDefiningOp<ttg::MemDescSubsliceOp>()) {
      current = subslice.getSrc();
      continue;
    }
    if (auto index = current.getDefiningOp<ttg::MemDescIndexOp>()) {
      current = index.getSrc();
      continue;
    }
    break;
  }
  return current;
}

static bool isTileProducerViewLikeOp(Operation *op) {
  return isa<ttg::MemDescIndexOp, ttg::MemDescSubsliceOp>(op) ||
         op->getName().getStringRef() == "tle.memdesc_wgmma_view";
}

static Operation *cloneWithUpdatedMemDescViewType(OpBuilder &builder,
                                                  Operation *op,
                                                  IRMapping &mapping) {
  auto mapResults = [&](Operation *newOp) -> Operation * {
    for (auto [oldResult, newResult] :
         llvm::zip(op->getResults(), newOp->getResults()))
      mapping.map(oldResult, newResult);
    return newOp;
  };
  if (auto index = dyn_cast<ttg::MemDescIndexOp>(op)) {
    Value src = mapping.lookupOrDefault(index.getSrc());
    auto oldTy = index.getType();
    bool isMutable = cast<ttg::MemDescType>(src.getType()).getMutableMemory();
    auto newTy = ttg::MemDescType::get(oldTy.getShape(), oldTy.getElementType(),
                                       oldTy.getEncoding(),
                                       oldTy.getMemorySpace(), isMutable);
    auto newOp =
        ttg::MemDescIndexOp::create(builder, index.getLoc(), newTy, src,
                                    mapping.lookupOrDefault(index.getIndex()));
    newOp->setAttrs(op->getAttrs());
    return mapResults(newOp);
  }
  if (auto subslice = dyn_cast<ttg::MemDescSubsliceOp>(op)) {
    Value src = mapping.lookupOrDefault(subslice.getSrc());
    auto oldTy = subslice.getType();
    bool isMutable = cast<ttg::MemDescType>(src.getType()).getMutableMemory();
    auto newTy = ttg::MemDescType::get(
        oldTy.getShape(), oldTy.getElementType(), oldTy.getEncoding(),
        oldTy.getMemorySpace(), isMutable, oldTy.getAllocShape());
    auto newOp = ttg::MemDescSubsliceOp::create(
        builder, subslice.getLoc(), newTy, src, subslice.getOffsets());
    newOp->setAttrs(op->getAttrs());
    return mapResults(newOp);
  }
  if (auto trans = dyn_cast<ttg::MemDescTransOp>(op)) {
    Value src = mapping.lookupOrDefault(trans.getSrc());
    auto newOp = ttg::MemDescTransOp::create(builder, trans.getLoc(), src,
                                             trans.getOrder());
    newOp->setAttrs(op->getAttrs());
    return mapResults(newOp);
  }
  if (auto reshape = dyn_cast<ttg::MemDescReshapeOp>(op)) {
    Value src = mapping.lookupOrDefault(reshape.getSrc());
    auto newOp = ttg::MemDescReshapeOp::create(builder, reshape.getLoc(), src,
                                               reshape.getType().getShape());
    newOp->setAttrs(op->getAttrs());
    return mapResults(newOp);
  }
  if (auto view = dyn_cast<triton::tle::MemDescWGMMAViewOp>(op)) {
    Value src = mapping.lookupOrDefault(view.getSrc());
    auto oldTy = view.getType();
    bool isMutable = cast<ttg::MemDescType>(src.getType()).getMutableMemory();
    auto newTy = ttg::MemDescType::get(
        oldTy.getShape(), oldTy.getElementType(), oldTy.getEncoding(),
        oldTy.getMemorySpace(), isMutable, oldTy.getAllocShape());
    auto newOp = triton::tle::MemDescWGMMAViewOp::create(
        builder, view.getLoc(), newTy, src, view.getOrder());
    newOp->setAttrs(op->getAttrs());
    return mapResults(newOp);
  }
  return builder.clone(*op, mapping);
}

static bool isRematerializablePrefixOp(Operation *op) {
  if (isMemoryEffectFree(op))
    return true;
  if (auto loadOp = dyn_cast<tt::LoadOp>(op))
    return !loadOp.getIsVolatile();
  return false;
}

static bool
isRematerializablePrefixValue(Value value, Block *body,
                              llvm::SmallDenseSet<Operation *, 32> &prefixSet,
                              ArrayRef<AsyncTileProducerGroup> producerGroups,
                              DenseMap<Value, bool> &cache) {
  auto it = cache.find(value);
  if (it != cache.end())
    return it->second;

  if (isa<ttg::AsyncTokenType>(value.getType()))
    return cache[value] = false;

  if (isProducerAllocResult(producerGroups, value) ||
      isProducerBaseMemDesc(producerGroups, value))
    return cache[value] = false;

  if (auto barg = dyn_cast<BlockArgument>(value)) {
    if (barg.getOwner() != body)
      return cache[value] = true;
    return cache[value] = barg.getArgNumber() == 0;
  }

  Operation *def = value.getDefiningOp();
  if (!def)
    return cache[value] = false;
  if (isProducerLoadOp(producerGroups, def) ||
      isProducerAllocOp(producerGroups, def) ||
      isProducerAsyncCopyOp(producerGroups, def))
    return cache[value] = false;
  if (!prefixSet.contains(def) && !isRematerializablePrefixOp(def))
    return cache[value] = true;
  if (prefixSet.contains(def) && !isRematerializablePrefixOp(def))
    return cache[value] = false;

  cache[value] = false;
  for (Value operand : def->getOperands()) {
    if (!isRematerializablePrefixValue(operand, body, prefixSet, producerGroups,
                                       cache))
      return false;
  }
  cache[value] = true;
  return true;
}

static LogicalResult materializeRematerializablePrefixValue(
    Value value, Block *body, llvm::SmallDenseSet<Operation *, 32> &prefixSet,
    ExplicitPipelinePlan &plan, OpBuilder &builder, IRMapping &mapping,
    llvm::SmallDenseSet<Operation *, 32> &activeDefs) {
  if (mapping.contains(value))
    return success();

  if (isa<ttg::AsyncTokenType>(value.getType()))
    return failure();

  if (isProducerAllocResult(plan.producerGroups, value) ||
      isProducerBaseMemDesc(plan.producerGroups, value))
    return failure();

  if (auto barg = dyn_cast<BlockArgument>(value)) {
    if (barg.getOwner() != body)
      return success();
    if (barg.getArgNumber() == 0)
      return success();
    return mapping.contains(value) ? success() : failure();
  }

  Operation *def = value.getDefiningOp();
  if (!def)
    return success();
  if (isProducerLoadOp(plan.producerGroups, def) ||
      isProducerAllocOp(plan.producerGroups, def) ||
      isProducerAsyncCopyOp(plan.producerGroups, def))
    return failure();
  if (!prefixSet.contains(def) && !isRematerializablePrefixOp(def))
    return success();
  if (prefixSet.contains(def) && !isRematerializablePrefixOp(def))
    return failure();
  if (!activeDefs.insert(def).second)
    return failure();

  for (Value operand : def->getOperands()) {
    if (failed(materializeRematerializablePrefixValue(
            operand, body, prefixSet, plan, builder, mapping, activeDefs))) {
      activeDefs.erase(def);
      return failure();
    }
  }

  bool allMapped = llvm::all_of(def->getResults(), [&](Value result) {
    return mapping.contains(result);
  });
  if (!allMapped)
    cloneWithUpdatedMemDescViewType(builder, def, mapping);
  activeDefs.erase(def);
  return success();
}

static LogicalResult materializeRematerializableOperands(
    Operation *op, Block *body, llvm::SmallDenseSet<Operation *, 32> &prefixSet,
    ExplicitPipelinePlan &plan, OpBuilder &builder, IRMapping &mapping) {
  llvm::SmallDenseSet<Operation *, 32> activeDefs;
  for (Value operand : op->getOperands()) {
    if (failed(materializeRematerializablePrefixValue(
            operand, body, prefixSet, plan, builder, mapping, activeDefs)))
      return failure();
  }
  return success();
}

static bool hasDotLikeConsumer(Value value, Block *scopeBody,
                               llvm::SmallDenseSet<Operation *, 8> &visited) {
  for (Operation *user : value.getUsers()) {
    if (!visited.insert(user).second)
      continue;
    if (isa<ttng::WarpGroupDotOp>(user) && user->getBlock() == scopeBody)
      return true;
    if (isTileProducerViewLikeOp(user) && !user->getResults().empty() &&
        hasDotLikeConsumer(user->getResult(0), scopeBody, visited)) {
      return true;
    }
  }
  return false;
}

static Operation *
getFirstDotLikeConsumerOp(Value value, Block *scopeBody,
                          llvm::SmallDenseSet<Operation *, 8> &visited) {
  Operation *firstDot = nullptr;
  for (Operation *user : value.getUsers()) {
    if (!visited.insert(user).second)
      continue;
    if (isa<ttng::WarpGroupDotOp>(user) && user->getBlock() == scopeBody) {
      if (!firstDot || user->isBeforeInBlock(firstDot))
        firstDot = user;
      continue;
    }
    if (!isTileProducerViewLikeOp(user) || user->getNumResults() == 0)
      continue;
    if (Operation *nested =
            getFirstDotLikeConsumerOp(user->getResult(0), scopeBody, visited)) {
      if (!firstDot || nested->isBeforeInBlock(firstDot))
        firstDot = nested;
    }
  }
  return firstDot;
}

static TileStyleLoopAnalysis analyzeTileStyleLoop(scf::ForOp forOp) {
  TileStyleLoopAnalysis analysis;
  Block *body = forOp.getBody();
  bool hasLoopStages = hasAnyLoopStageAttr(forOp);
  llvm::MapVector<Value, SmallVector<ttg::AsyncCopyGlobalToLocalOp>>
      directAsyncFamilies;
  for (Operation &op : forOp.getBody()->without_terminator()) {
    auto loadOp = dyn_cast<tt::LoadOp>(op);
    if (loadOp && isStageZero(loadOp)) {
      ttg::LocalAllocOp allocOp = getSingleDirectLocalAllocUser(loadOp);
      if (!allocOp)
        continue;
      llvm::SmallDenseSet<Operation *, 8> visited;
      if (!hasDotLikeConsumer(allocOp.getResult(), body, visited))
        continue;
      analysis.asyncTileProducers.push_back({.baseMemDesc = allocOp.getResult(),
                                             .loadOp = loadOp,
                                             .allocOp = allocOp});
      continue;
    }

    auto asyncCopyOp = dyn_cast<ttg::AsyncCopyGlobalToLocalOp>(op);
    if (!asyncCopyOp)
      continue;
    if (!isTleDirectAsyncProducer(asyncCopyOp))
      continue;
    if (hasLoopStages && !isStageZero(asyncCopyOp))
      continue;
    Value baseMemDesc = stripProducerMemDescViews(asyncCopyOp->getOperand(1));
    if (!baseMemDesc)
      continue;
    llvm::SmallDenseSet<Operation *, 8> visited;
    if (!hasDotLikeConsumer(baseMemDesc, body, visited))
      continue;
    directAsyncFamilies[baseMemDesc].push_back(asyncCopyOp);
  }

  for (auto &it : directAsyncFamilies) {
    analysis.asyncTileProducers.push_back(
        {.baseMemDesc = it.first, .asyncCopyOps = it.second});
  }
  return analysis;
}

static bool isTileStyleCandidate(scf::ForOp forOp) {
  auto numStagesAttr =
      forOp->getAttrOfType<IntegerAttr>(tt::kNumStagesAttrName);
  if (!numStagesAttr || numStagesAttr.getInt() != 2)
    return false;
  if (auto tileStyleAttr =
          forOp->getAttrOfType<IntegerAttr>(kTleTileStylePipelineAttr))
    return tileStyleAttr.getInt() != 0;
  if (hasAnyLoopStageAttr(forOp))
    return false;
  TileStyleLoopAnalysis analysis = analyzeTileStyleLoop(forOp);
  return llvm::any_of(analysis.asyncTileProducers,
                      [&](AsyncTileProducerGroup group) {
                        return group.isDirectAsyncFamily();
                      });
}

static bool hasPositiveConstantStep(scf::ForOp forOp) {
  auto getConstValue = [](Value value) -> std::optional<int64_t> {
    if (auto cst =
            dyn_cast_or_null<arith::ConstantIntOp>(value.getDefiningOp()))
      return cst.value();
    if (auto cst =
            dyn_cast_or_null<arith::ConstantIndexOp>(value.getDefiningOp()))
      return cst.value();
    return std::nullopt;
  };
  std::optional<int64_t> step = getConstValue(forOp.getStep());
  return step && *step > 0;
}

static void stripPipelineAttrs(Operation *op) {
  op->removeAttr(tt::kLoopStageAttrName);
  op->removeAttr(tt::kLoopClusterAttrName);
  op->removeAttr(tt::kScheduledMaxStageAttrName);
  op->walk([&](Operation *nested) {
    nested->removeAttr(tt::kLoopStageAttrName);
    nested->removeAttr(tt::kLoopClusterAttrName);
    nested->removeAttr(tt::kScheduledMaxStageAttrName);
  });
}

static std::optional<ExplicitPipelinePlan>
buildExplicitPipelinePlan(scf::ForOp forOp,
                          ArrayRef<AsyncTileProducerGroup> producerGroups) {
  // Dynamic trip counts are valid: the explicit materializer emits runtime
  // guards for the empty and single-iteration cases. It only needs a known
  // forward step to form the one-step lead/steady-state bounds.
  if (!hasPositiveConstantStep(forOp))
    return std::nullopt;
  if (producerGroups.empty())
    return std::nullopt;

  SmallVector<Operation *> prefixOps;
  SmallVector<Operation *> restOps;
  Operation *prefixEnd = nullptr;
  bool hasStageZeroOps = false;
  for (Operation &op : forOp.getBody()->without_terminator()) {
    if (!isStageZero(&op))
      continue;
    hasStageZeroOps = true;
    if (!prefixEnd || prefixEnd->isBeforeInBlock(&op))
      prefixEnd = &op;
  }
  Operation *firstDot = nullptr;
  Block *scopeBody = forOp.getBody();
  for (AsyncTileProducerGroup group : producerGroups) {
    llvm::SmallDenseSet<Operation *, 8> visited;
    Operation *dot =
        getFirstDotLikeConsumerOp(group.baseMemDesc, scopeBody, visited);
    if (!dot)
      return std::nullopt;
    if (!firstDot || dot->isBeforeInBlock(firstDot))
      firstDot = dot;
  }
  if (!firstDot)
    return std::nullopt;

  if (hasStageZeroOps) {
    bool inPrefix = true;
    for (Operation &op : forOp.getBody()->without_terminator()) {
      if (inPrefix) {
        prefixOps.push_back(&op);
        if (&op == prefixEnd)
          inPrefix = false;
        continue;
      }
      restOps.push_back(&op);
    }
    if (inPrefix || restOps.empty())
      return std::nullopt;
  } else {
    for (Operation &op : forOp.getBody()->without_terminator()) {
      if (op.isBeforeInBlock(firstDot)) {
        prefixOps.push_back(&op);
        continue;
      }
      restOps.push_back(&op);
    }
    if (prefixOps.empty() || restOps.empty())
      return std::nullopt;
  }

  Block &body = forOp.getRegion().front();
  for (Operation *op : prefixOps) {
    for (Value operand : op->getOperands()) {
      auto barg = dyn_cast<BlockArgument>(operand);
      if (!barg || barg.getOwner() != &body)
        continue;
      if (barg.getArgNumber() != 0)
        return std::nullopt;
    }
  }

  llvm::SmallDenseSet<Operation *, 32> restSet(restOps.begin(), restOps.end());
  llvm::SmallDenseSet<Operation *, 32> prefixSet(prefixOps.begin(),
                                                 prefixOps.end());
  SmallVector<Value> carriedValues;
  llvm::SmallDenseSet<Value, 16> seenValues;
  DenseMap<Value, bool> rematCache;
  for (Operation *op : prefixOps) {
    for (Value result : op->getResults()) {
      if (isProducerAllocResult(producerGroups, result))
        continue;
      bool escapesPrefix =
          llvm::any_of(result.getUsers(),
                       [&](Operation *user) { return restSet.contains(user); });
      if (!escapesPrefix || !seenValues.insert(result).second)
        continue;
      if (isRematerializablePrefixValue(result, &body, prefixSet,
                                        producerGroups, rematCache))
        continue;
      carriedValues.push_back(result);
    }
  }
  return ExplicitPipelinePlan{
      .producerGroups = SmallVector<AsyncTileProducerGroup>(
          producerGroups.begin(), producerGroups.end()),
      .prefixOps = std::move(prefixOps),
      .restOps = std::move(restOps),
      .carriedValues = std::move(carriedValues),
      .firstDotOp = firstDot,
  };
}

static std::optional<AsyncTileProducerGroup>
findProducerGroupByLoad(ExplicitPipelinePlan &plan, Operation *op) {
  for (AsyncTileProducerGroup group : plan.producerGroups) {
    if (group.loadOp.getOperation() == op)
      return group;
  }
  return std::nullopt;
}

static std::optional<AsyncTileProducerGroup>
findProducerGroupByAlloc(ExplicitPipelinePlan &plan, Operation *op) {
  for (AsyncTileProducerGroup group : plan.producerGroups) {
    if (group.allocOp.getOperation() == op)
      return group;
  }
  return std::nullopt;
}

static std::optional<unsigned>
findProducerGroupIndexByAsyncCopy(ExplicitPipelinePlan &plan, Operation *op) {
  for (auto [idx, group] : llvm::enumerate(plan.producerGroups)) {
    if (llvm::any_of(group.asyncCopyOps,
                     [&](ttg::AsyncCopyGlobalToLocalOp copy) {
                       return copy.getOperation() == op;
                     }))
      return idx;
  }
  return std::nullopt;
}

static std::optional<unsigned>
findProducerGroupIndexByAlloc(ExplicitPipelinePlan &plan, Operation *op) {
  for (auto [idx, group] : llvm::enumerate(plan.producerGroups)) {
    if (group.allocOp.getOperation() == op)
      return idx;
  }
  return std::nullopt;
}

static SmallVector<Value> clonePrefixOps(ExplicitPipelinePlan &plan,
                                         OpBuilder &builder, IRMapping &mapping,
                                         ArrayRef<Value> targetViews = {}) {
  SmallVector<Value> commitTokens;
  commitTokens.resize(plan.producerGroups.size());
  SmallVector<SmallVector<Value>> groupedAsyncTokens(
      plan.producerGroups.size());

  if (!targetViews.empty()) {
    if (targetViews.size() != plan.producerGroups.size())
      return {};
    for (auto [idx, group] : llvm::enumerate(plan.producerGroups)) {
      mapping.map(group.baseMemDesc, targetViews[idx]);
      if (group.isLegacyLoadAlloc())
        mapping.map(group.allocOp.getResult(), targetViews[idx]);
    }
  }

  llvm::SmallDenseSet<Operation *, 32> prefixSet(plan.prefixOps.begin(),
                                                 plan.prefixOps.end());
  for (Operation *op : plan.prefixOps) {
    if (findProducerGroupByLoad(plan, op))
      continue;
    if (std::optional<unsigned> groupIdx =
            findProducerGroupIndexByAlloc(plan, op)) {
      AsyncTileProducerGroup group = plan.producerGroups[*groupIdx];
      auto loadOp = group.loadOp;
      Value ptr = mapping.lookupOrDefault(loadOp.getPtr());
      Value mask = loadOp.getMask() ? mapping.lookupOrDefault(loadOp.getMask())
                                    : Value();
      Value other = loadOp.getOther()
                        ? mapping.lookupOrDefault(loadOp.getOther())
                        : Value();
      Value asyncView;
      if (!targetViews.empty()) {
        if (targetViews.size() != plan.producerGroups.size())
          return {};
        asyncView = targetViews[*groupIdx];
      } else {
        asyncView = ttg::LocalAllocOp::create(
            builder, loadOp.getLoc(),
            ensureMutableMemDescType(group.allocOp.getType()));
      }
      auto asyncCopy = ttg::AsyncCopyGlobalToLocalOp::create(
          builder, loadOp.getLoc(), ptr, asyncView, mask, other,
          loadOp.getCache(), loadOp.getEvict(), loadOp.getIsVolatile());
      auto asyncCommit = ttg::AsyncCommitGroupOp::create(
          builder, asyncCopy.getLoc(), asyncCopy.getToken().getType(),
          asyncCopy.getToken());
      mapping.map(group.allocOp.getResult(), asyncView);
      commitTokens[*groupIdx] = asyncCommit.getResult();
      continue;
    }
    if (auto groupIdx = findProducerGroupIndexByAsyncCopy(plan, op)) {
      if (failed(materializeRematerializableOperands(
              op, builder.getInsertionBlock(), prefixSet, plan, builder,
              mapping)))
        return {};
      Operation *cloned = cloneWithUpdatedMemDescViewType(builder, op, mapping);
      groupedAsyncTokens[*groupIdx].push_back(cloned->getResult(0));
      continue;
    }
    if (isa<ttg::AsyncCommitGroupOp, ttg::AsyncWaitOp>(op))
      continue;
    if (failed(materializeRematerializableOperands(
            op, builder.getInsertionBlock(), prefixSet, plan, builder,
            mapping)))
      return {};
    cloneWithUpdatedMemDescViewType(builder, op, mapping);
  }

  for (auto [idx, group] : llvm::enumerate(plan.producerGroups)) {
    if (!group.isDirectAsyncFamily())
      continue;
    auto commit = ttg::AsyncCommitGroupOp::create(
        builder, group.asyncCopyOps.front().getLoc(),
        ValueRange(groupedAsyncTokens[idx]));
    commitTokens[idx] = commit.getAsyncToken();
  }
  return commitTokens;
}

static FailureOr<PrefixMaterializationResult>
materializePrefixAt(OpBuilder &builder, scf::ForOp forOp,
                    ExplicitPipelinePlan &plan, ArrayRef<Value> baseAllocs,
                    Value iv, Value slotIdx) {
  IRMapping mapping;
  mapping.map(forOp.getInductionVar(), iv);
  SmallVector<Value> views;
  views.reserve(baseAllocs.size());
  for (Value baseAlloc : baseAllocs)
    views.push_back(
        triton::createSingleBufferView(builder, baseAlloc, slotIdx));
  SmallVector<Value> tokens = clonePrefixOps(plan, builder, mapping, views);
  if (tokens.size() != plan.producerGroups.size())
    return failure();
  SmallVector<Value> carriedValues;
  carriedValues.reserve(plan.carriedValues.size());
  for (Value carried : plan.carriedValues) {
    Value mapped = mapping.lookupOrNull(carried);
    if (!mapped)
      return failure();
    carriedValues.push_back(mapped);
  }
  return PrefixMaterializationResult{std::move(carriedValues),
                                     std::move(tokens)};
}

static FailureOr<SmallVector<Value>>
materializeConsumeAt(OpBuilder &builder, scf::ForOp forOp,
                     ExplicitPipelinePlan &plan, ArrayRef<Value> baseAllocs,
                     ArrayRef<Value> loopStateValues,
                     ArrayRef<Value> carriedValues, ArrayRef<Value> tokens,
                     Value slotIdx, Value iv, int32_t waitNum) {
  IRMapping mapping;
  mapping.map(forOp.getInductionVar(), iv);
  for (auto [oldArg, newVal] :
       llvm::zip(forOp.getRegionIterArgs(), loopStateValues))
    mapping.map(oldArg, newVal);
  for (auto [oldVal, newVal] : llvm::zip(plan.carriedValues, carriedValues))
    mapping.map(oldVal, newVal);

  SmallVector<Value> views;
  views.reserve(baseAllocs.size());
  for (Value baseAlloc : baseAllocs)
    views.push_back(
        triton::createSingleBufferView(builder, baseAlloc, slotIdx));
  for (auto [groupIdx, group] : llvm::enumerate(plan.producerGroups)) {
    mapping.map(group.baseMemDesc, views[groupIdx]);
    if (group.isLegacyLoadAlloc())
      mapping.map(group.allocOp.getResult(), views[groupIdx]);
  }

  llvm::SmallDenseSet<Operation *, 32> prefixSet(plan.prefixOps.begin(),
                                                 plan.prefixOps.end());
  Block *insertBlock = builder.getInsertionBlock();
  for (Operation *op : plan.restOps) {
    if (failed(materializeRematerializableOperands(op, insertBlock, prefixSet,
                                                   plan, builder, mapping)))
      return failure();
    if (op == plan.firstDotOp) {
      auto waitOp = builder.create<ttg::AsyncWaitOp>(
          op->getLoc(), ValueRange(tokens), builder.getI32IntegerAttr(waitNum));
      waitOp.setNum(waitNum);
    }
    cloneWithUpdatedMemDescViewType(builder, op, mapping);
  }

  auto oldYield = cast<scf::YieldOp>(forOp.getBody()->getTerminator());
  SmallVector<Value> results;
  results.reserve(oldYield.getNumOperands());
  for (Value operand : oldYield.getOperands()) {
    Value mapped = mapping.lookupOrNull(operand);
    if (!mapped)
      return failure();
    results.push_back(mapped);
  }
  return results;
}

static Operation *materializeExplicitOneStepLead(OpBuilder &builder,
                                                 scf::ForOp forOp,
                                                 ExplicitPipelinePlan &plan) {
  auto getConstStepValue = [](Value value) -> std::optional<int64_t> {
    if (auto cst =
            dyn_cast_or_null<arith::ConstantIndexOp>(value.getDefiningOp()))
      return cst.value();
    if (auto cst =
            dyn_cast_or_null<arith::ConstantIntOp>(value.getDefiningOp()))
      return cst.value();
    return std::nullopt;
  };
  std::optional<int64_t> stepVal = getConstStepValue(forOp.getStep());
  if (!stepVal || *stepVal <= 0)
    return nullptr;

  builder.setInsertionPoint(forOp);

  Value zero = arith::ConstantIntOp::create(builder, forOp.getLoc(), 0, 32);
  Value one = arith::ConstantIntOp::create(builder, forOp.getLoc(), 1, 32);
  Value hasAny =
      arith::CmpIOp::create(builder, forOp.getLoc(), arith::CmpIPredicate::slt,
                            forOp.getLowerBound(), forOp.getUpperBound());
  auto outerIf =
      scf::IfOp::create(builder, forOp.getLoc(), forOp.getResultTypes(), hasAny,
                        /*withElseRegion=*/true);

  auto eraseAndFail = [&]() -> Operation * {
    outerIf.erase();
    return nullptr;
  };

  {
    OpBuilder elseBuilder = outerIf.getElseBodyBuilder();
    scf::YieldOp::create(elseBuilder, forOp.getLoc(), forOp.getInitArgs());
  }

  {
    OpBuilder thenBuilder = outerIf.getThenBodyBuilder();
    SmallVector<Value> initState(forOp.getInitArgs().begin(),
                                 forOp.getInitArgs().end());
    SmallVector<Value> baseAllocs;
    baseAllocs.reserve(plan.producerGroups.size());
    for (AsyncTileProducerGroup group : plan.producerGroups) {
      auto baseType = cast<ttg::MemDescType>(group.baseMemDesc.getType());
      auto multiBufferType = ensureMutableMemDescType(
          triton::getMultiBufferedType(baseType, /*depth=*/2));
      Location allocLoc = group.isLegacyLoadAlloc()
                              ? group.allocOp.getLoc()
                              : group.asyncCopyOps.front().getLoc();
      Value baseAlloc =
          ttg::LocalAllocOp::create(thenBuilder, allocLoc, multiBufferType);
      baseAllocs.push_back(baseAlloc);
    }

    FailureOr<PrefixMaterializationResult> curInit = materializePrefixAt(
        thenBuilder, forOp, plan, baseAllocs, forOp.getLowerBound(), zero);
    if (failed(curInit))
      return eraseAndFail();

    Value nextInitIv = arith::AddIOp::create(
        thenBuilder, forOp.getLoc(), forOp.getLowerBound(), forOp.getStep());
    Value hasSecond = arith::CmpIOp::create(thenBuilder, forOp.getLoc(),
                                            arith::CmpIPredicate::slt,
                                            nextInitIv, forOp.getUpperBound());
    auto secondIf = scf::IfOp::create(thenBuilder, forOp.getLoc(),
                                      forOp.getResultTypes(), hasSecond,
                                      /*withElseRegion=*/true);

    {
      OpBuilder elseBuilder = secondIf.getElseBodyBuilder();
      FailureOr<SmallVector<Value>> singleResults =
          materializeConsumeAt(elseBuilder, forOp, plan, baseAllocs, initState,
                               (*curInit).carriedValues, (*curInit).tokens,
                               zero, forOp.getLowerBound(),
                               /*waitNum=*/0);
      if (failed(singleResults))
        return eraseAndFail();
      for (Value baseAlloc : baseAllocs)
        ttg::LocalDeallocOp::create(elseBuilder, forOp.getLoc(), baseAlloc);
      scf::YieldOp::create(elseBuilder, forOp.getLoc(), *singleResults);
    }

    {
      OpBuilder then2Builder = secondIf.getThenBodyBuilder();
      FailureOr<PrefixMaterializationResult> nextInit = materializePrefixAt(
          then2Builder, forOp, plan, baseAllocs, nextInitIv, one);
      if (failed(nextInit))
        return eraseAndFail();

      SmallVector<Value> steadyInitArgs(initState.begin(), initState.end());
      steadyInitArgs.append((*curInit).carriedValues);
      steadyInitArgs.append((*curInit).tokens);
      steadyInitArgs.push_back(zero);
      steadyInitArgs.append((*nextInit).carriedValues);
      steadyInitArgs.append((*nextInit).tokens);
      steadyInitArgs.push_back(one);

      Value upperMinusStep = arith::SubIOp::create(
          then2Builder, forOp.getLoc(), forOp.getUpperBound(), forOp.getStep());
      Value steadyUpper = arith::SubIOp::create(
          then2Builder, forOp.getLoc(), upperMinusStep, forOp.getStep());
      auto steadyFor = then2Builder.create<scf::ForOp>(
          forOp.getLoc(), forOp.getLowerBound(), steadyUpper, forOp.getStep(),
          steadyInitArgs);
      steadyFor->setAttr(kTleExplicitTileStylePipelineAttr,
                         then2Builder.getI32IntegerAttr(1));
      steadyFor->setAttr(kTleAsyncTileProducerCountAttr,
                         then2Builder.getI32IntegerAttr(
                             static_cast<int32_t>(plan.producerGroups.size())));
      if (auto tileStyleAttr = forOp->getAttr(kTleTileStylePipelineAttr))
        steadyFor->setAttr(kTleTileStylePipelineAttr, tileStyleAttr);

      Block &steadyBody = steadyFor.getRegion().front();
      OpBuilder bodyBuilder = OpBuilder::atBlockBegin(&steadyBody);
      unsigned numLoopState = forOp.getNumRegionIterArgs();
      unsigned numCarried = plan.carriedValues.size();
      unsigned numGroups = plan.producerGroups.size();
      auto steadyArgs = steadyFor.getRegionIterArgs();
      SmallVector<Value> loopStateArgs(steadyArgs.begin(),
                                       steadyArgs.begin() + numLoopState);
      SmallVector<Value> curCarriedArgs(steadyArgs.begin() + numLoopState,
                                        steadyArgs.begin() + numLoopState +
                                            numCarried);
      SmallVector<Value> curTokenArgs(
          steadyArgs.begin() + numLoopState + numCarried,
          steadyArgs.begin() + numLoopState + numCarried + numGroups);
      Value curSlotArg = steadyArgs[numLoopState + numCarried + numGroups];
      unsigned nextBase = numLoopState + numCarried + numGroups + 1;
      SmallVector<Value> nextCarriedArgs(steadyArgs.begin() + nextBase,
                                         steadyArgs.begin() + nextBase +
                                             numCarried);
      SmallVector<Value> nextTokenArgs(
          steadyArgs.begin() + nextBase + numCarried,
          steadyArgs.begin() + nextBase + numCarried + numGroups);
      Value nextSlotArg = steadyArgs[nextBase + numCarried + numGroups];

      FailureOr<SmallVector<Value>> curResults = materializeConsumeAt(
          bodyBuilder, forOp, plan, baseAllocs, loopStateArgs, curCarriedArgs,
          curTokenArgs, curSlotArg, steadyFor.getInductionVar(), /*waitNum=*/1);
      if (failed(curResults))
        return eraseAndFail();
      Value nextIv =
          arith::AddIOp::create(bodyBuilder, forOp.getLoc(),
                                steadyFor.getInductionVar(), forOp.getStep());
      Value futureIv = arith::AddIOp::create(bodyBuilder, forOp.getLoc(),
                                             nextIv, forOp.getStep());
      FailureOr<PrefixMaterializationResult> futurePrefix = materializePrefixAt(
          bodyBuilder, forOp, plan, baseAllocs, futureIv, curSlotArg);
      if (failed(futurePrefix))
        return eraseAndFail();

      SmallVector<Value> steadyYield;
      steadyYield.reserve(numLoopState + numCarried + numGroups + 1 +
                          numCarried + numGroups + 1);
      steadyYield.append(*curResults);
      steadyYield.append(nextCarriedArgs);
      steadyYield.append(nextTokenArgs);
      steadyYield.push_back(nextSlotArg);
      steadyYield.append((*futurePrefix).carriedValues);
      steadyYield.append((*futurePrefix).tokens);
      steadyYield.push_back(curSlotArg);
      if (!steadyBody.empty() && isa<scf::YieldOp>(steadyBody.back()))
        steadyBody.back().erase();
      bodyBuilder.create<scf::YieldOp>(forOp.getLoc(), steadyYield);

      SmallVector<Value> steadyResults(steadyFor.getResults().begin(),
                                       steadyFor.getResults().end());
      SmallVector<Value> curLoopState(steadyResults.begin(),
                                      steadyResults.begin() + numLoopState);
      SmallVector<Value> curCarriedFinal(steadyResults.begin() + numLoopState,
                                         steadyResults.begin() + numLoopState +
                                             numCarried);
      SmallVector<Value> curTokensFinal(
          steadyResults.begin() + numLoopState + numCarried,
          steadyResults.begin() + numLoopState + numCarried + numGroups);
      Value curSlotFinal = steadyResults[numLoopState + numCarried + numGroups];
      SmallVector<Value> nextCarriedFinal(steadyResults.begin() + nextBase,
                                          steadyResults.begin() + nextBase +
                                              numCarried);
      SmallVector<Value> nextTokensFinal(
          steadyResults.begin() + nextBase + numCarried,
          steadyResults.begin() + nextBase + numCarried + numGroups);
      Value nextSlotFinal = steadyResults[nextBase + numCarried + numGroups];

      FailureOr<SmallVector<Value>> curEpilogueResults = materializeConsumeAt(
          then2Builder, forOp, plan, baseAllocs, curLoopState, curCarriedFinal,
          curTokensFinal, curSlotFinal, steadyUpper, /*waitNum=*/1);
      if (failed(curEpilogueResults))
        return eraseAndFail();
      FailureOr<SmallVector<Value>> nextEpilogueResults = materializeConsumeAt(
          then2Builder, forOp, plan, baseAllocs, *curEpilogueResults,
          nextCarriedFinal, nextTokensFinal, nextSlotFinal, upperMinusStep,
          /*waitNum=*/0);
      if (failed(nextEpilogueResults))
        return eraseAndFail();
      for (Value baseAlloc : baseAllocs)
        ttg::LocalDeallocOp::create(then2Builder, forOp.getLoc(), baseAlloc);
      scf::YieldOp::create(then2Builder, forOp.getLoc(), *nextEpilogueResults);
    }

    scf::YieldOp::create(thenBuilder, forOp.getLoc(), secondIf.getResults());
  }

  return outerIf.getOperation();
}

static Operation *
cloneLoopWithoutPipeliningAttrs(OpBuilder &builder, scf::ForOp forOp,
                                TileStyleLoopAnalysis analysis) {
  auto newForOp =
      cast<scf::ForOp>(builder.cloneWithoutRegions(*forOp.getOperation()));
  IRMapping mapping;
  builder.cloneRegionBefore(forOp.getRegion(), newForOp.getRegion(),
                            newForOp.getRegion().end(), mapping);

  SmallVector<AsyncTileProducerGroup> mappedAsyncTileProducers;
  mappedAsyncTileProducers.reserve(analysis.asyncTileProducers.size());
  for (AsyncTileProducerGroup group : analysis.asyncTileProducers) {
    AsyncTileProducerGroup mappedGroup;
    mappedGroup.baseMemDesc = mapping.lookupOrDefault(group.baseMemDesc);
    if (group.isLegacyLoadAlloc()) {
      auto mappedLoadOp = dyn_cast_or_null<tt::LoadOp>(
          mapping.lookupOrNull(group.loadOp.getOperation()));
      auto mappedAllocOp = dyn_cast_or_null<ttg::LocalAllocOp>(
          mapping.lookupOrNull(group.allocOp.getOperation()));
      if (!mappedLoadOp || !mappedAllocOp)
        continue;
      mappedGroup.loadOp = mappedLoadOp;
      mappedGroup.allocOp = mappedAllocOp;
    } else {
      for (ttg::AsyncCopyGlobalToLocalOp copyOp : group.asyncCopyOps) {
        auto mappedCopyOp = dyn_cast_or_null<ttg::AsyncCopyGlobalToLocalOp>(
            mapping.lookupOrNull(copyOp.getOperation()));
        if (mappedCopyOp)
          mappedGroup.asyncCopyOps.push_back(mappedCopyOp);
      }
      if (mappedGroup.asyncCopyOps.empty())
        continue;
    }
    mappedAsyncTileProducers.push_back(std::move(mappedGroup));
  }

  newForOp->removeAttr(tt::kNumStagesAttrName);
  newForOp->removeAttr(tt::kScheduledMaxStageAttrName);
  newForOp->removeAttr(tt::kLoopStageAttrName);
  newForOp->removeAttr(tt::kLoopClusterAttrName);
  newForOp->setAttr(kTleExplicitTileStylePipelineAttr,
                    builder.getI32IntegerAttr(1));
  newForOp->setAttr(kTleAsyncTileProducerCountAttr,
                    builder.getI32IntegerAttr(static_cast<int32_t>(
                        analysis.asyncTileProducers.size())));

  if (kEnableExperimentalOneStepLeadMaterialization &&
      !mappedAsyncTileProducers.empty()) {
    if (std::optional<ExplicitPipelinePlan> plan =
            buildExplicitPipelinePlan(newForOp, mappedAsyncTileProducers)) {
      OpBuilder explicitBuilder(newForOp);
      explicitBuilder.setInsertionPoint(newForOp);
      if (Operation *explicitOp = materializeExplicitOneStepLead(
              explicitBuilder, newForOp, *plan)) {
        SmallVector<Value> replacementResults;
        replacementResults.reserve(newForOp.getNumResults());
        for (unsigned i = 0; i < newForOp.getNumResults(); ++i)
          replacementResults.push_back(explicitOp->getResult(i));
        newForOp.getResults().replaceAllUsesWith(replacementResults);
        newForOp.erase();
        return explicitOp;
      }
    }
  }

  stripPipelineAttrs(newForOp);

  for (AsyncTileProducerGroup group : mappedAsyncTileProducers) {
    if (!group.isLegacyLoadAlloc())
      continue;
    ImplicitLocOpBuilder groupBuilder(group.loadOp.getLoc(), group.loadOp);
    Value asyncAlloc = ttg::LocalAllocOp::create(
        groupBuilder, ensureMutableMemDescType(group.allocOp.getType()));
    Operation *asyncCopy = ttg::AsyncCopyGlobalToLocalOp::create(
        groupBuilder, group.loadOp.getPtr(), asyncAlloc, group.loadOp.getMask(),
        group.loadOp.getOther(), group.loadOp.getCache(),
        group.loadOp.getEvict(), group.loadOp.getIsVolatile());
    Operation *asyncCommit =
        ttg::AsyncCommitGroupOp::create(groupBuilder, asyncCopy->getResult(0));

    llvm::SmallDenseSet<Operation *, 8> visited;
    if (Operation *firstDot = getFirstDotLikeConsumerOp(
            group.allocOp.getResult(), newForOp.getBody(), visited)) {
      ImplicitLocOpBuilder waitBuilder(group.loadOp.getLoc(), firstDot);
      ttg::AsyncWaitOp::create(waitBuilder,
                               ValueRange{asyncCommit->getResult(0)}, 0);
    }

    group.allocOp.replaceAllUsesWith(asyncAlloc);
    group.allocOp.erase();
    group.loadOp.erase();
  }

  return newForOp.getOperation();
}

class MaterializeTileStylePipelinePass
    : public impl::TritonTleMaterializeTileStylePipelineBase<
          MaterializeTileStylePipelinePass> {
  void runOnOperation() override {
    ModuleOp module = getOperation();

    SmallVector<scf::ForOp> candidates;
    module.walk([&](scf::ForOp forOp) {
      if (isTileStyleCandidate(forOp))
        candidates.push_back(forOp);
    });

    for (scf::ForOp forOp : candidates) {
      OpBuilder builder(forOp);
      TileStyleLoopAnalysis analysis = analyzeTileStyleLoop(forOp);
      Operation *newOp =
          cloneLoopWithoutPipeliningAttrs(builder, forOp, analysis);
      SmallVector<Value> replacementResults;
      replacementResults.reserve(forOp.getNumResults());
      for (unsigned i = 0; i < forOp.getNumResults(); ++i)
        replacementResults.push_back(newOp->getResult(i));
      forOp.getResults().replaceAllUsesWith(replacementResults);
      forOp.erase();
    }
  }
};

} // namespace
} // namespace mlir::triton::tle
