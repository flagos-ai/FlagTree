#include "tle/dialect/include/Transforms/EncodingRematerialization.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/Triton/IR/Types.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/ScopeExit.h"

namespace mlir::triton::gpu {
namespace {

constexpr unsigned kMaxRematerializationDepth = 64;

static RankedTensorType cloneTensorTypeWithEncoding(RankedTensorType type,
                                                    Attribute encoding) {
  return RankedTensorType::get(type.getShape(), type.getElementType(),
                               encoding);
}

static RankedTensorType
cloneTensorTypeWithElementAndEncoding(RankedTensorType shapeType,
                                      Type elementType, Attribute encoding) {
  return RankedTensorType::get(shapeType.getShape(), elementType, encoding);
}

static bool hasEncoding(Type type, Attribute encoding) {
  auto tensorType = dyn_cast<RankedTensorType>(type);
  return tensorType && tensorType.getEncoding() == encoding;
}

static bool hasSameShapeAndElementType(RankedTensorType source,
                                       RankedTensorType target) {
  return source.getShape() == target.getShape() &&
         source.getElementType() == target.getElementType();
}

static bool hasSameShapeAndElementTypeIgnoringEncoding(Type lhs, Type rhs) {
  auto lhsTensor = dyn_cast<RankedTensorType>(lhs);
  auto rhsTensor = dyn_cast<RankedTensorType>(rhs);
  return lhsTensor && rhsTensor &&
         lhsTensor.getShape() == rhsTensor.getShape() &&
         lhsTensor.getElementType() == rhsTensor.getElementType();
}

static Type getScalarOrElementType(Type type) {
  if (auto tensorTy = dyn_cast<RankedTensorType>(type))
    return tensorTy.getElementType();
  return type;
}

static bool isSharedMemDesc(Type type) {
  auto memDescTy = dyn_cast<MemDescType>(type);
  return memDescTy && isa<SharedMemorySpaceAttr>(memDescTy.getMemorySpace());
}

static bool isSharedMemDesc(Value value) {
  return value && isSharedMemDesc(value.getType());
}

static bool isPointerAddressSpace(Value value, int addressSpace) {
  if (!value)
    return false;
  auto ptrTy =
      dyn_cast<triton::PointerType>(getScalarOrElementType(value.getType()));
  return ptrTy && ptrTy.getAddressSpace() == addressSpace;
}

static bool isSharedPointerValue(Value value) {
  return isPointerAddressSpace(value, /*addressSpace=*/3);
}

static bool isGlobalPointerValue(Value value) {
  return isPointerAddressSpace(value, /*addressSpace=*/1);
}

static Value stripConvertLayouts(Value value) {
  Value current = value;
  while (auto convert = current.getDefiningOp<ConvertLayoutOp>())
    current = convert.getSrc();
  return current;
}

static Value getSharedMemDescRoot(Value value) {
  Value current = stripConvertLayouts(value);
  while (current) {
    if (auto index = current.getDefiningOp<MemDescIndexOp>()) {
      current = stripConvertLayouts(index.getSrc());
      continue;
    }
    if (auto subslice = current.getDefiningOp<MemDescSubsliceOp>()) {
      current = stripConvertLayouts(subslice.getSrc());
      continue;
    }
    if (auto reinterpret = current.getDefiningOp<MemDescReinterpretOp>()) {
      current = stripConvertLayouts(reinterpret.getSrc());
      continue;
    }
    if (auto trans = current.getDefiningOp<MemDescTransOp>()) {
      current = stripConvertLayouts(trans.getSrc());
      continue;
    }
    if (auto reshape = current.getDefiningOp<MemDescReshapeOp>()) {
      current = stripConvertLayouts(reshape.getSrc());
      continue;
    }
    break;
  }
  return current;
}

static bool memDescRootsMayAlias(Value lhs, Value rhs) {
  Value lhsRoot = getSharedMemDescRoot(lhs);
  Value rhsRoot = getSharedMemDescRoot(rhs);
  if (!lhsRoot || !rhsRoot)
    return true;
  return lhsRoot == rhsRoot;
}

static bool isWriteLike(MemoryEffects::Effect *effect) {
  return isa<MemoryEffects::Write, MemoryEffects::Allocate,
             MemoryEffects::Free>(effect);
}

static bool isKnownGlobalResource(SideEffects::Resource *resource) {
  return isa<triton::GlobalMemory>(resource);
}

static bool effectValueProvesNonShared(Value value) {
  if (!value)
    return false;
  if (isSharedMemDesc(value) || isSharedPointerValue(value))
    return false;
  if (isGlobalPointerValue(value))
    return true;
  return false;
}

static std::optional<Value> getDirectSharedWriteMemDesc(Operation *op) {
  if (auto localStore = dyn_cast<LocalStoreOp>(op))
    return localStore.getDst();
  if (auto asyncCopy = dyn_cast<AsyncCopyGlobalToLocalOp>(op))
    return asyncCopy.getResult();
  if (auto localAlloc = dyn_cast<LocalAllocOp>(op))
    return localAlloc.getResult();
  if (auto tmaCopy = dyn_cast<TMACopyOp>(op)) {
    if (isSharedMemDesc(tmaCopy.getDst()))
      return tmaCopy.getDst();
    return std::nullopt;
  }
  return std::nullopt;
}

static bool mayWriteSharedMemoryAlias(Operation *op, Value memdesc) {
  if (!op || !memdesc)
    return true;

  if (std::optional<Value> directMemDesc = getDirectSharedWriteMemDesc(op))
    return memDescRootsMayAlias(*directMemDesc, memdesc);

  if (auto store = dyn_cast<triton::StoreOp>(op))
    return isSharedPointerValue(store.getPtr());

  if (isMemoryEffectFree(op))
    return false;

  auto iface = dyn_cast<MemoryEffectOpInterface>(op);
  if (!iface)
    return true;

  SmallVector<MemoryEffects::EffectInstance> effects;
  iface.getEffects(effects);
  if (effects.empty())
    return true;

  for (MemoryEffects::EffectInstance effect : effects) {
    if (!isWriteLike(effect.getEffect()))
      continue;

    Value value = effect.getValue();
    if (value && isSharedMemDesc(value)) {
      if (memDescRootsMayAlias(value, memdesc))
        return true;
      continue;
    }
    if (value && isSharedPointerValue(value))
      return true;
    if (isa<SideEffects::DefaultResource>(effect.getResource()) ||
        isa<SharedMemory>(effect.getResource()))
      return true;
    if (isKnownGlobalResource(effect.getResource()))
      continue;
    if (effectValueProvesNonShared(value))
      continue;
    return true;
  }
  return false;
}

static bool hasInterveningSharedMemoryWriteAliasSameBlock(Operation *from,
                                                          Operation *to,
                                                          Value memdesc) {
  if (!from || !to || from->getBlock() != to->getBlock() ||
      !from->isBeforeInBlock(to))
    return true;

  for (Operation *cur = from->getNextNode(); cur && cur != to;
       cur = cur->getNextNode()) {
    if (mayWriteSharedMemoryAlias(cur, memdesc))
      return true;
  }
  return false;
}

static Value lookupCached(Value value, Attribute targetEncoding,
                          EncodingRematerializer &rematerializer,
                          EncodingRematerializationCache &cache) {
  for (auto &entry : cache) {
    if (entry.source == value && entry.targetEncoding == targetEncoding &&
        rematerializer.isAvailableAt(entry.rematerialized))
      return entry.rematerialized;
  }
  return {};
}

static void cacheValue(Value source, Attribute targetEncoding,
                       Value rematerialized,
                       EncodingRematerializationCache &cache) {
  cache.push_back({source, targetEncoding, rematerialized});
}

static void eraseOpsCreatedAfter(Operation *previous, Operation *insertBefore) {
  if (!insertBefore || !insertBefore->getBlock())
    return;
  Operation *cur =
      previous ? previous->getNextNode() : &insertBefore->getBlock()->front();
  while (cur && cur != insertBefore) {
    Operation *next = cur->getNextNode();
    cur->erase();
    cur = next;
  }
}

static bool areConstantsEquivalent(arith::ConstantOp lhs,
                                   arith::ConstantOp rhs) {
  auto lhsTy = dyn_cast<RankedTensorType>(lhs.getType());
  auto rhsTy = dyn_cast<RankedTensorType>(rhs.getType());
  if (lhsTy || rhsTy) {
    if (!lhsTy || !rhsTy ||
        !hasSameShapeAndElementTypeIgnoringEncoding(lhsTy, rhsTy))
      return false;

    auto lhsSplat = dyn_cast<SplatElementsAttr>(lhs.getValue());
    auto rhsSplat = dyn_cast<SplatElementsAttr>(rhs.getValue());
    return lhsSplat && rhsSplat &&
           lhsSplat.getSplatValue<Attribute>() ==
               rhsSplat.getSplatValue<Attribute>();
  }

  return lhs.getValue() == rhs.getValue();
}

static bool haveSameNonTensorOperands(
    Operation *lhs, Operation *rhs, EncodingRematerializer &rematerializer,
    SmallVectorImpl<std::pair<Value, Value>> &active, unsigned depth) {
  if (lhs->getNumOperands() != rhs->getNumOperands())
    return false;

  for (auto [lhsOperand, rhsOperand] :
       llvm::zip(lhs->getOperands(), rhs->getOperands())) {
    if (isa<RankedTensorType>(lhsOperand.getType()) ||
        isa<RankedTensorType>(rhsOperand.getType())) {
      if (!rematerializer.areEquivalentIgnoringEncoding(lhsOperand, rhsOperand,
                                                        active, depth + 1))
        return false;
      continue;
    }

    if (lhsOperand != rhsOperand)
      return false;
  }
  return true;
}

static bool areLocalLoadsEquivalent(EncodingRematerializer &rematerializer,
                                    LocalLoadOp lhs, LocalLoadOp rhs) {
  if (lhs.getSrc() != rhs.getSrc())
    return false;
  if (lhs.getToken() || rhs.getToken())
    return lhs.getToken() == rhs.getToken();
  const EncodingRematerializationPolicy &policy = rematerializer.getPolicy();
  Operation *insertBefore = rematerializer.getInsertBefore();
  return !policy.hasInterveningSharedMemoryWriteAlias(lhs, insertBefore,
                                                      lhs.getSrc()) &&
         !policy.hasInterveningSharedMemoryWriteAlias(rhs, insertBefore,
                                                      rhs.getSrc());
}

static bool
isEncodingPolymorphicOp(Operation *op,
                        const EncodingRematerializationPolicy &policy) {
  if (!op)
    return false;
  if (op->hasTrait<OpTrait::Elementwise>())
    return true;
  if (isa<triton::MakeRangeOp, triton::SplatOp, triton::ExpandDimsOp,
          triton::BroadcastOp, triton::AddPtrOp>(op))
    return true;
  return policy.isCustomEncodingPolymorphicOp(op);
}

static Value
findAvailableEquivalentValue(Value value, Attribute targetEncoding,
                             EncodingRematerializer &rematerializer) {
  auto sourceTy = dyn_cast<RankedTensorType>(value.getType());
  Operation *insertBefore = rematerializer.getInsertBefore();
  if (!sourceTy || !targetEncoding || !insertBefore)
    return {};

  RankedTensorType targetTy =
      cloneTensorTypeWithEncoding(sourceTy, targetEncoding);
  Block *block = insertBefore->getBlock();
  if (!block)
    return {};

  for (Operation &op : *block) {
    if (&op == insertBefore)
      break;
    for (Value candidate : op.getResults()) {
      if (candidate == value || candidate.getType() != targetTy)
        continue;
      if (!rematerializer.isAvailableAt(candidate))
        continue;
      SmallVector<std::pair<Value, Value>, 16> active;
      if (rematerializer.areEquivalentIgnoringEncoding(value, candidate, active,
                                                       /*depth=*/0))
        return candidate;
    }
  }
  return {};
}

static void collectEquivalentNvidiaMmaEncodingsImpl(
    Value root, Operation *insertBefore, DominanceInfo &dominance,
    const EncodingRematerializationPolicy &policy,
    SmallVectorImpl<Attribute> &encodings, SmallPtrSetImpl<Value> &visited,
    unsigned depth) {
  if (!root || depth > kMaxRematerializationDepth ||
      !visited.insert(root).second)
    return;

  EncodingRematerializationCache cache;
  IRRewriter rewriter(insertBefore->getContext());
  EncodingRematerializer rematerializer(rewriter, insertBefore, cache,
                                        dominance, policy);

  auto rootTy = dyn_cast<RankedTensorType>(root.getType());
  if (rootTy) {
    Block *block = insertBefore->getBlock();
    if (block) {
      for (Operation &op : *block) {
        if (&op == insertBefore)
          break;
        for (Value candidate : op.getResults()) {
          auto candidateTy = dyn_cast<RankedTensorType>(candidate.getType());
          if (!candidateTy ||
              !hasSameShapeAndElementType(rootTy, candidateTy) ||
              !isa<NvidiaMmaEncodingAttr>(candidateTy.getEncoding()))
            continue;
          if (llvm::is_contained(encodings, candidateTy.getEncoding()))
            continue;
          if (!rematerializer.isAvailableAt(candidate))
            continue;
          SmallVector<std::pair<Value, Value>, 16> active;
          if (rematerializer.areEquivalentIgnoringEncoding(root, candidate,
                                                           active, /*depth=*/0))
            encodings.push_back(candidateTy.getEncoding());
        }
      }
    }
  }

  Operation *def = root.getDefiningOp();
  if (!def)
    return;
  if (auto convert = dyn_cast<ConvertLayoutOp>(def)) {
    collectEquivalentNvidiaMmaEncodingsImpl(convert.getSrc(), insertBefore,
                                            dominance, policy, encodings,
                                            visited, depth + 1);
    return;
  }

  if (def->getNumRegions() != 0)
    return;
  for (Value operand : def->getOperands())
    if (isa<RankedTensorType>(operand.getType()))
      collectEquivalentNvidiaMmaEncodingsImpl(operand, insertBefore, dominance,
                                              policy, encodings, visited,
                                              depth + 1);
}

static FailureOr<Value> rematerializeConstant(arith::ConstantOp constant,
                                              RankedTensorType targetTy,
                                              RewriterBase &rewriter) {
  auto sourceTy = dyn_cast<RankedTensorType>(constant.getType());
  if (!sourceTy || !hasSameShapeAndElementType(sourceTy, targetTy))
    return failure();

  auto splat = dyn_cast<SplatElementsAttr>(constant.getValue());
  if (!splat)
    return failure();

  auto attr =
      SplatElementsAttr::get(targetTy, splat.getSplatValue<Attribute>());
  return rewriter.create<arith::ConstantOp>(constant.getLoc(), targetTy, attr)
      .getResult();
}

static FailureOr<Value> rematerializeMakeRange(triton::MakeRangeOp range,
                                               RankedTensorType targetTy,
                                               RewriterBase &rewriter) {
  auto sourceTy = dyn_cast<RankedTensorType>(range.getType());
  if (!sourceTy || !hasSameShapeAndElementType(sourceTy, targetTy))
    return failure();
  if (targetTy.getRank() != 1 || !targetTy.getElementType().isInteger(32))
    return failure();

  return rewriter
      .create<triton::MakeRangeOp>(
          range.getLoc(), targetTy,
          static_cast<uint32_t>(range.getStartAttr().getInt()),
          static_cast<uint32_t>(range.getEndAttr().getInt()))
      .getResult();
}

static FailureOr<Value>
rematerializeExpandDims(EncodingRematerializer &rematerializer,
                        triton::ExpandDimsOp expand, RankedTensorType targetTy,
                        Attribute targetEncoding,
                        llvm::SmallPtrSetImpl<Value> &active, unsigned depth) {
  auto sourceTy = dyn_cast<RankedTensorType>(expand.getType());
  auto inputTy = dyn_cast<RankedTensorType>(expand.getSrc().getType());
  if (!sourceTy || !inputTy || !hasSameShapeAndElementType(sourceTy, targetTy))
    return failure();

  unsigned axis = expand.getAxis();
  if (axis >= static_cast<unsigned>(targetTy.getRank()))
    return failure();
  auto distributedEncoding =
      dyn_cast_or_null<DistributedEncodingTrait>(targetEncoding);
  if (!distributedEncoding)
    return failure();

  Attribute inputEncoding = SliceEncodingAttr::get(
      rematerializer.getRewriter().getContext(), axis, distributedEncoding);
  auto input = rematerializer.rematerialize(expand.getSrc(), inputEncoding,
                                            active, depth + 1);
  if (failed(input))
    return failure();

  return rematerializer.getRewriter()
      .create<triton::ExpandDimsOp>(expand.getLoc(), targetTy, *input,
                                    expand.getAxisAttr())
      .getResult();
}

static FailureOr<Value>
rematerializeBroadcast(EncodingRematerializer &rematerializer,
                       triton::BroadcastOp broadcast, RankedTensorType targetTy,
                       Attribute targetEncoding,
                       llvm::SmallPtrSetImpl<Value> &active, unsigned depth) {
  auto sourceTy = dyn_cast<RankedTensorType>(broadcast.getType());
  auto inputTy = dyn_cast<RankedTensorType>(broadcast.getSrc().getType());
  if (!sourceTy || !inputTy || !hasSameShapeAndElementType(sourceTy, targetTy))
    return failure();

  auto input = rematerializer.rematerialize(broadcast.getSrc(), targetEncoding,
                                            active, depth + 1);
  if (failed(input))
    return failure();

  return rematerializer.getRewriter()
      .create<triton::BroadcastOp>(broadcast.getLoc(), targetTy, *input)
      .getResult();
}

static bool
canCloneSameShapePureOp(Operation *op, RankedTensorType sourceResultTy,
                        RankedTensorType targetTy,
                        const EncodingRematerializationPolicy &policy) {
  if (op->getNumResults() != 1 || op->getNumRegions() != 0 ||
      op->hasTrait<OpTrait::IsTerminator>())
    return false;
  if (!isEncodingPolymorphicOp(op, policy) || !isSpeculatable(op) ||
      !isMemoryEffectFree(op))
    return false;
  if (sourceResultTy.getShape() != targetTy.getShape() ||
      sourceResultTy.getElementType() != targetTy.getElementType())
    return false;

  for (Value operand : op->getOperands()) {
    auto operandTy = dyn_cast<RankedTensorType>(operand.getType());
    if (!operandTy)
      continue;
    if (operandTy.getShape() != sourceResultTy.getShape())
      return false;
  }
  return true;
}

static FailureOr<Value>
cloneSameShapePureOp(EncodingRematerializer &rematerializer, Operation *op,
                     RankedTensorType targetTy, Attribute targetEncoding,
                     llvm::SmallPtrSetImpl<Value> &active, unsigned depth) {
  IRMapping mapping;
  for (Value operand : op->getOperands()) {
    auto operandTy = dyn_cast<RankedTensorType>(operand.getType());
    if (!operandTy) {
      if (!rematerializer.isAvailableAt(operand))
        return failure();
      mapping.map(operand, operand);
      continue;
    }

    RankedTensorType targetOperandTy = cloneTensorTypeWithElementAndEncoding(
        targetTy, operandTy.getElementType(), targetEncoding);
    auto rematerializedOperand = rematerializer.rematerialize(
        operand, targetEncoding, active, depth + 1);
    if (failed(rematerializedOperand) ||
        (*rematerializedOperand).getType() != targetOperandTy)
      return failure();
    mapping.map(operand, *rematerializedOperand);
  }

  Operation *cloned = rematerializer.getRewriter().clone(*op, mapping);
  cloned->getResult(0).setType(targetTy);
  return cloned->getResult(0);
}

} // namespace

bool EncodingRematerializationPolicy::isCustomEncodingPolymorphicOp(
    Operation *op) const {
  return false;
}

bool EncodingRematerializationPolicy::hasInterveningSharedMemoryWriteAlias(
    Operation *from, Operation *to, Value memdesc) const {
  return hasInterveningSharedMemoryWriteAliasSameBlock(from, to, memdesc);
}

bool EncodingRematerializationPolicy::areCustomValuesEquivalent(
    EncodingRematerializer &rematerializer, Value lhs, Value rhs,
    SmallVectorImpl<std::pair<Value, Value>> &active, unsigned depth) const {
  return false;
}

FailureOr<Value> EncodingRematerializationPolicy::rematerializeCustomValue(
    EncodingRematerializer &rematerializer, Value value,
    RankedTensorType targetType, Attribute targetEncoding,
    llvm::SmallPtrSetImpl<Value> &active, unsigned depth) const {
  return failure();
}

EncodingRematerializer::EncodingRematerializer(
    RewriterBase &rewriter, Operation *insertBefore,
    EncodingRematerializationCache &cache, DominanceInfo &dominance,
    const EncodingRematerializationPolicy &policy)
    : rewriter(rewriter), insertBefore(insertBefore), cache(cache),
      dominance(dominance), policy(policy) {}

bool EncodingRematerializer::isAvailableAt(Value value) const {
  if (!insertBefore)
    return false;
  if (auto blockArg = dyn_cast<BlockArgument>(value))
    return dominance.dominates(blockArg, insertBefore);

  Operation *def = value.getDefiningOp();
  if (!def)
    return false;
  if (def->getBlock() == insertBefore->getBlock())
    return def->isBeforeInBlock(insertBefore);
  return dominance.dominates(def, insertBefore);
}

bool EncodingRematerializer::areEquivalentIgnoringEncoding(
    Value lhs, Value rhs, SmallVectorImpl<std::pair<Value, Value>> &active,
    unsigned depth) {
  if (lhs == rhs)
    return true;
  if (depth > kMaxRematerializationDepth)
    return false;

  auto lhsConvert = lhs.getDefiningOp<ConvertLayoutOp>();
  if (lhsConvert)
    return areEquivalentIgnoringEncoding(lhsConvert.getSrc(), rhs, active,
                                         depth + 1);
  auto rhsConvert = rhs.getDefiningOp<ConvertLayoutOp>();
  if (rhsConvert)
    return areEquivalentIgnoringEncoding(lhs, rhsConvert.getSrc(), active,
                                         depth + 1);

  bool lhsTensor = isa<RankedTensorType>(lhs.getType());
  bool rhsTensor = isa<RankedTensorType>(rhs.getType());
  if (lhsTensor || rhsTensor) {
    if (!hasSameShapeAndElementTypeIgnoringEncoding(lhs.getType(),
                                                    rhs.getType()))
      return false;
  } else {
    return lhs == rhs;
  }

  if (!isAvailableAt(lhs) || !isAvailableAt(rhs))
    return false;

  std::pair<Value, Value> pair{lhs, rhs};
  if (llvm::is_contained(active, pair))
    return true;
  active.push_back(pair);
  auto popActive = llvm::make_scope_exit([&]() { active.pop_back(); });

  Operation *lhsDef = lhs.getDefiningOp();
  Operation *rhsDef = rhs.getDefiningOp();
  if (!lhsDef || !rhsDef)
    return false;
  if (lhsDef->getName() != rhsDef->getName())
    return false;
  if (lhsDef->getNumResults() != 1 || rhsDef->getNumResults() != 1 ||
      lhsDef->getNumRegions() != 0 || rhsDef->getNumRegions() != 0)
    return false;

  if (auto lhsCst = dyn_cast<arith::ConstantOp>(lhsDef))
    return areConstantsEquivalent(lhsCst, cast<arith::ConstantOp>(rhsDef));

  if (auto lhsLocalLoad = dyn_cast<LocalLoadOp>(lhsDef))
    return areLocalLoadsEquivalent(*this, lhsLocalLoad,
                                   cast<LocalLoadOp>(rhsDef));

  if (policy.areCustomValuesEquivalent(*this, lhs, rhs, active, depth))
    return true;

  if (!isEncodingPolymorphicOp(lhsDef, policy) || !isSpeculatable(lhsDef) ||
      !isMemoryEffectFree(lhsDef))
    return false;
  if (lhsDef->getAttrs() != rhsDef->getAttrs())
    return false;

  return haveSameNonTensorOperands(lhsDef, rhsDef, *this, active, depth);
}

FailureOr<Value>
EncodingRematerializer::rematerialize(Value value, Attribute targetEncoding,
                                      llvm::SmallPtrSetImpl<Value> &active,
                                      unsigned depth) {
  if (depth > kMaxRematerializationDepth)
    return failure();

  auto sourceTy = dyn_cast<RankedTensorType>(value.getType());
  if (!sourceTy) {
    if (!isAvailableAt(value))
      return failure();
    return value;
  }

  RankedTensorType targetTy =
      cloneTensorTypeWithEncoding(sourceTy, targetEncoding);
  if (value.getType() == targetTy) {
    if (!isAvailableAt(value))
      return failure();
    return value;
  }

  if (Value cached = lookupCached(value, targetEncoding, *this, cache))
    return cached;

  if (Value equivalent =
          findAvailableEquivalentValue(value, targetEncoding, *this)) {
    cacheValue(value, targetEncoding, equivalent, cache);
    return equivalent;
  }

  if (!active.insert(value).second)
    return failure();

  auto eraseActive = llvm::make_scope_exit([&]() { active.erase(value); });

  Operation *def = value.getDefiningOp();
  if (!def || !isAvailableAt(value))
    return failure();

  FailureOr<Value> result = failure();
  rewriter.setInsertionPoint(insertBefore);
  Operation *rollbackPrevious = insertBefore->getPrevNode();
  size_t rollbackCacheSize = cache.size();

  if (auto custom = policy.rematerializeCustomValue(
          *this, value, targetTy, targetEncoding, active, depth);
      succeeded(custom)) {
    result = custom;
  } else if (auto convert = dyn_cast<ConvertLayoutOp>(def)) {
    result = rematerialize(convert.getSrc(), targetEncoding, active, depth + 1);
  } else if (auto constant = dyn_cast<arith::ConstantOp>(def)) {
    result = rematerializeConstant(constant, targetTy, rewriter);
  } else if (auto range = dyn_cast<triton::MakeRangeOp>(def)) {
    result = rematerializeMakeRange(range, targetTy, rewriter);
  } else if (auto expand = dyn_cast<triton::ExpandDimsOp>(def)) {
    result = rematerializeExpandDims(*this, expand, targetTy, targetEncoding,
                                     active, depth);
  } else if (auto broadcast = dyn_cast<triton::BroadcastOp>(def)) {
    result = rematerializeBroadcast(*this, broadcast, targetTy, targetEncoding,
                                    active, depth);
  } else if (auto localLoad = dyn_cast<LocalLoadOp>(def)) {
    auto sourceTy = dyn_cast<RankedTensorType>(localLoad.getType());
    if (sourceTy && hasSameShapeAndElementType(sourceTy, targetTy) &&
        !localLoad.getToken() &&
        !policy.hasInterveningSharedMemoryWriteAlias(localLoad, insertBefore,
                                                     localLoad.getSrc())) {
      auto newLoad = rewriter.create<LocalLoadOp>(localLoad.getLoc(), targetTy,
                                                  localLoad.getSrc());
      newLoad->setAttrs(localLoad->getAttrs());
      result = newLoad.getResult();
    }
  } else if (canCloneSameShapePureOp(def, sourceTy, targetTy, policy)) {
    result = cloneSameShapePureOp(*this, def, targetTy, targetEncoding, active,
                                  depth);
  }

  if (failed(result)) {
    eraseOpsCreatedAfter(rollbackPrevious, insertBefore);
    cache.resize(rollbackCacheSize);
    return failure();
  }
  if ((*result).getType() != targetTy ||
      !hasEncoding((*result).getType(), targetEncoding)) {
    eraseOpsCreatedAfter(rollbackPrevious, insertBefore);
    cache.resize(rollbackCacheSize);
    return failure();
  }
  cacheValue(value, targetEncoding, *result, cache);
  return result;
}

FailureOr<Value> rematerializeWithEncoding(
    RewriterBase &rewriter, Operation *insertBefore, Value value,
    Attribute targetEncoding, EncodingRematerializationCache &cache,
    DominanceInfo &dominance, const EncodingRematerializationPolicy &policy) {
  if (!targetEncoding || !insertBefore)
    return failure();

  llvm::SmallPtrSet<Value, 16> active;
  EncodingRematerializer rematerializer(rewriter, insertBefore, cache,
                                        dominance, policy);
  return rematerializer.rematerialize(value, targetEncoding, active,
                                      /*depth=*/0);
}

void collectAvailableEquivalentNvidiaMmaEncodings(
    Value root, Operation *insertBefore, DominanceInfo &dominance,
    const EncodingRematerializationPolicy &policy,
    SmallVectorImpl<Attribute> &encodings) {
  if (!root || !insertBefore)
    return;

  llvm::SmallPtrSet<Value, 32> visited;
  collectEquivalentNvidiaMmaEncodingsImpl(root, insertBefore, dominance, policy,
                                          encodings, visited, /*depth=*/0);
}

} // namespace mlir::triton::gpu
