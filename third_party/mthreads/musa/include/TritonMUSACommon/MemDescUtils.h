#ifndef TRITONMUSA_COMMON_MEMDESC_UTILS_H
#define TRITONMUSA_COMMON_MEMDESC_UTILS_H

#include "TritonMUSACommon/SqmmaAttrUtils.h"
#include "TritonMUSACommon/TMEUtils.h"
#include "mlir/IR/PatternMatch.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/Transforms/Utility.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/StringRef.h"

#include <optional>

namespace mlir::triton::musa {
namespace tt = mlir::triton;
namespace ttg = mlir::triton::gpu;

inline SmallVector<int64_t> getNormalizedAllocShape(ttg::MemDescType type) {
  SmallVector<int64_t> allocShape(type.getAllocShape().begin(),
                                  type.getAllocShape().end());
  if (allocShape.empty())
    allocShape.assign(type.getShape().begin(), type.getShape().end());
  return allocShape;
}

inline bool areMemDescTypesCompatible(ttg::MemDescType lhs,
                                      ttg::MemDescType rhs) {
  return lhs.getShape() == rhs.getShape() &&
         lhs.getElementType() == rhs.getElementType() &&
         lhs.getEncoding() == rhs.getEncoding() &&
         lhs.getMemorySpace() == rhs.getMemorySpace() &&
         getNormalizedAllocShape(lhs) == getNormalizedAllocShape(rhs);
}

inline bool
areMemDescTypesEquivalentForDescriptorContract(ttg::MemDescType lhs,
                                               ttg::MemDescType rhs) {
  return lhs.getShape() == rhs.getShape() &&
         lhs.getElementType() == rhs.getElementType() &&
         lhs.getEncoding() == rhs.getEncoding() &&
         lhs.getMemorySpace() == rhs.getMemorySpace();
}

inline bool areMemDescTypesLayoutEquivalent(ttg::MemDescType lhs,
                                            ttg::MemDescType rhs) {
  return lhs.getShape() == rhs.getShape() &&
         lhs.getElementType() == rhs.getElementType() &&
         lhs.getMemorySpace() == rhs.getMemorySpace() &&
         getNormalizedAllocShape(lhs) == getNormalizedAllocShape(rhs) &&
         getMUSASharedLinearLayoutOrGeneric(lhs) ==
             getMUSASharedLinearLayoutOrGeneric(rhs);
}

inline bool isCanonicalLandingMemDescViewOp(Operation *op) {
  return isa<ttg::MemDescIndexOp, ttg::MemDescSubsliceOp,
             ttg::MemDescReinterpretOp, ttg::MemDescTransOp,
             ttg::MemDescReshapeOp>(op);
}

inline SmallVector<ttg::LocalAllocOp>
collectCanonicalLandingLocalAllocUsers(tt::DescriptorLoadOp loadOp) {
  SmallVector<ttg::LocalAllocOp> localAllocs;
  llvm::SmallPtrSet<Operation *, 4> seen;
  for (Operation *user : loadOp.getResult().getUsers()) {
    auto localAlloc = dyn_cast<ttg::LocalAllocOp>(user);
    if (!localAlloc || localAlloc.getSrc() != loadOp.getResult())
      continue;
    if (seen.insert(localAlloc.getOperation()).second)
      localAllocs.push_back(localAlloc);
  }
  return localAllocs;
}

inline SmallVector<Value>
collectCanonicalLandingMemDescRoots(tt::DescriptorLoadOp loadOp) {
  SmallVector<Value> roots;
  llvm::SmallPtrSet<void *, 4> seen;
  for (Operation *user : loadOp.getResult().getUsers()) {
    if (auto localAlloc = dyn_cast<ttg::LocalAllocOp>(user)) {
      if (localAlloc.getSrc() != loadOp.getResult())
        continue;
      if (seen.insert(localAlloc.getResult().getAsOpaquePointer()).second)
        roots.push_back(localAlloc.getResult());
      continue;
    }
    auto localStore = dyn_cast<ttg::LocalStoreOp>(user);
    if (!localStore || localStore.getSrc() != loadOp.getResult())
      continue;
    if (seen.insert(localStore.getDst().getAsOpaquePointer()).second)
      roots.push_back(localStore.getDst());
  }
  return roots;
}

inline SmallVector<Value>
collectCanonicalLandingTerminalMemDescValues(tt::DescriptorLoadOp loadOp) {
  SmallVector<Value> terminals;
  SmallVector<Value> worklist = collectCanonicalLandingMemDescRoots(loadOp);
  llvm::SmallPtrSet<void *, 8> visited;
  llvm::SmallPtrSet<void *, 8> seenTerminals;
  while (!worklist.empty()) {
    Value current = worklist.pop_back_val();
    if (!visited.insert(current.getAsOpaquePointer()).second)
      continue;
    if (!isa<ttg::MemDescType>(current.getType()))
      continue;

    bool hasViewUser = false;
    bool hasNonViewUser = false;
    for (Operation *user : current.getUsers()) {
      if (isCanonicalLandingMemDescViewOp(user)) {
        hasViewUser = true;
        for (Value result : user->getResults())
          worklist.push_back(result);
      } else {
        hasNonViewUser = true;
      }
    }

    if ((!hasViewUser || hasNonViewUser) &&
        seenTerminals.insert(current.getAsOpaquePointer()).second) {
      terminals.push_back(current);
    }
  }
  return terminals;
}

inline std::optional<ttg::MemDescType>
getUniqueCanonicalLandingRootMemDescType(tt::DescriptorLoadOp loadOp) {
  std::optional<ttg::MemDescType> uniqueTy;
  for (Value root : collectCanonicalLandingMemDescRoots(loadOp)) {
    auto rootTy = dyn_cast<ttg::MemDescType>(root.getType());
    if (!rootTy)
      continue;
    if (!uniqueTy) {
      uniqueTy = rootTy;
      continue;
    }
    if (!areMemDescTypesCompatible(*uniqueTy, rootTy))
      return std::nullopt;
  }
  return uniqueTy;
}

inline std::optional<ttg::MemDescType>
getUniqueCanonicalLandingTerminalMemDescType(tt::DescriptorLoadOp loadOp) {
  std::optional<ttg::MemDescType> uniqueTy;
  for (Value terminal : collectCanonicalLandingTerminalMemDescValues(loadOp)) {
    auto terminalTy = dyn_cast<ttg::MemDescType>(terminal.getType());
    if (!terminalTy)
      continue;
    if (!uniqueTy) {
      uniqueTy = terminalTy;
      continue;
    }
    if (!areMemDescTypesCompatible(*uniqueTy, terminalTy))
      return std::nullopt;
  }
  return uniqueTy;
}

inline std::optional<ttg::MemDescType>
getUniqueCanonicalLandingMemDescType(tt::DescriptorLoadOp loadOp) {
  return getUniqueCanonicalLandingTerminalMemDescType(loadOp);
}

inline bool hasCanonicalSharedLanding(tt::DescriptorLoadOp loadOp) {
  auto landingTy = getUniqueCanonicalLandingRootMemDescType(loadOp);
  return landingTy &&
         isa_and_nonnull<ttg::SharedEncodingTrait>(landingTy->getEncoding());
}

inline SmallVector<Value> collectConnectedCanonicalMemDescChain(Value seed) {
  SmallVector<Value> chain;
  SmallVector<Value> worklist{seed};
  llvm::SmallPtrSet<void *, 8> visited;
  while (!worklist.empty()) {
    Value current = worklist.pop_back_val();
    if (!visited.insert(current.getAsOpaquePointer()).second)
      continue;
    if (!isa<ttg::MemDescType>(current.getType()))
      continue;
    chain.push_back(current);
    if (Operation *defOp = current.getDefiningOp()) {
      if (isCanonicalLandingMemDescViewOp(defOp)) {
        if (auto srcOperand =
                dyn_cast<TypedValue<ttg::MemDescType>>(defOp->getOperand(0))) {
          worklist.push_back(srcOperand);
        }
      }
    }
    for (Operation *user : current.getUsers()) {
      if (!isCanonicalLandingMemDescViewOp(user))
        continue;
      for (Value result : user->getResults())
        worklist.push_back(result);
    }
  }
  return chain;
}

inline std::optional<ttg::MemDescType>
getUniqueCanonicalLandingSqmmaMemDescType(tt::DescriptorLoadOp loadOp) {
  std::optional<ttg::MemDescType> uniqueTy;
  auto expectedShape = loadOp.getType().getShape();
  Type expectedElemTy = loadOp.getType().getElementType();

  for (Value root : collectCanonicalLandingMemDescRoots(loadOp)) {
    for (Value memDescValue : collectConnectedCanonicalMemDescChain(root)) {
      auto memDescTy = dyn_cast<ttg::MemDescType>(memDescValue.getType());
      Operation *defOp = memDescValue.getDefiningOp();
      if (!memDescTy || !defOp || !hasSqmmaOpIdxAttr(defOp))
        continue;
      if (memDescTy.getShape() != expectedShape ||
          memDescTy.getElementType() != expectedElemTy)
        continue;

      if (!uniqueTy) {
        uniqueTy = memDescTy;
        continue;
      }
      if (!areMemDescTypesCompatible(*uniqueTy, memDescTy))
        return std::nullopt;
    }
  }
  return uniqueTy;
}

inline Operation *
getUniqueCanonicalLandingSqmmaAttrSource(tt::DescriptorLoadOp loadOp) {
  Operation *source = nullptr;
  auto mergeSource = [&](Operation *candidate) -> bool {
    if (!hasSqmmaOpIdxAttr(candidate))
      return true;
    if (!source) {
      source = candidate;
      return true;
    }
    for (auto name : kSqmmaAttrNames) {
      if (source->getAttr(name) != candidate->getAttr(name))
        return false;
    }
    return true;
  };

  for (Value root : collectCanonicalLandingMemDescRoots(loadOp)) {
    if (auto rootOp = root.getDefiningOp()) {
      if (!mergeSource(rootOp))
        return nullptr;
    }
    for (Value memDescValue : collectConnectedCanonicalMemDescChain(root)) {
      for (Operation *user : memDescValue.getUsers()) {
        if (!mergeSource(user))
          return nullptr;
      }
    }
  }
  return source;
}

inline bool copyCanonicalLandingSqmmaAttrs(tt::DescriptorLoadOp loadOp,
                                           Operation *dst) {
  Operation *source = getUniqueCanonicalLandingSqmmaAttrSource(loadOp);
  if (!source && hasSqmmaOpIdxAttr(loadOp.getOperation()))
    source = loadOp.getOperation();
  if (!source)
    return false;
  copySqmmaAttrs(source, dst);
  return true;
}

inline unsigned getDescriptorContractNumCTAs(Operation *op) {
  if (auto func = op ? op->getParentOfType<tt::FuncOp>() : tt::FuncOp())
    return std::max(1u, static_cast<unsigned>(ttg::lookupNumCTAs(func)));
  return 1;
}

inline FailureOr<ttg::MemDescType>
buildDescriptorLandingMemDescType(Operation *op, tt::TensorDescType descTy,
                                  RankedTensorType tensorTy,
                                  bool mutableMemory) {
  auto descBlockTy = descTy.getSignlessBlockType();
  auto swizzled = dyn_cast_or_null<ttg::SwizzledSharedEncodingAttr>(
      descBlockTy.getEncoding());
  if (!swizzled) {
    if (op)
      op->emitError("expected descriptor block type to be normalized to "
                    "canonical swizzled shared encoding");
    return failure();
  }
  auto normalizedEncoding =
      tryMapTMECompatibleSharedEncodingToCanonicalSwizzled(
          op, tensorTy, swizzled, tensorTy.getShape(),
          getDescriptorContractNumCTAs(op));
  if (!normalizedEncoding) {
    if (op)
      op->emitError("unable to project descriptor shared encoding onto "
                    "descriptor load/store tensor rank");
    return failure();
  }
  return ttg::MemDescType::get(
      tensorTy.getShape(), tensorTy.getElementType(), *normalizedEncoding,
      ttg::SharedMemorySpaceAttr::get(op->getContext()), mutableMemory);
}

inline std::optional<ttg::MemDescType> tryNormalizeCanonicalLandingMemDescType(
    Operation *op, ttg::MemDescType memDescTy, bool mutableMemory) {
  auto tensorTy =
      RankedTensorType::get(memDescTy.getShape(), memDescTy.getElementType());
  auto normalizedEncoding =
      tryMapTMECompatibleSharedEncodingToCanonicalSwizzled(
          op, tensorTy, memDescTy.getEncoding(), memDescTy.getShape(),
          getDescriptorContractNumCTAs(op));
  if (!normalizedEncoding)
    return std::nullopt;
  return ttg::MemDescType::get(memDescTy.getShape(), memDescTy.getElementType(),
                               *normalizedEncoding, memDescTy.getMemorySpace(),
                               mutableMemory, memDescTy.getAllocShape());
}

inline FailureOr<ttg::MemDescType>
resolveDescriptorLoadLandingMemDescType(tt::DescriptorLoadOp loadOp) {
  auto authoritative = buildDescriptorLandingMemDescType(
      loadOp.getOperation(), loadOp.getDesc().getType(), loadOp.getType(),
      /*mutableMemory=*/true);
  if (failed(authoritative))
    return failure();
  auto canonicalTy = getUniqueCanonicalLandingMemDescType(loadOp);
  if (!canonicalTy)
    return authoritative;
  auto normalizedCanonical = tryNormalizeCanonicalLandingMemDescType(
      loadOp.getOperation(), *canonicalTy,
      /*mutableMemory=*/true);
  if (!normalizedCanonical)
    return authoritative;
  if (!areMemDescTypesEquivalentForDescriptorContract(*normalizedCanonical,
                                                      *authoritative))
    return authoritative;
  return *normalizedCanonical;
}

inline FailureOr<ttg::MemDescType> resolveDescriptorStoreLandingMemDescType(
    tt::DescriptorStoreLikeOpInterface storeOp, bool mutableMemory) {
  return buildDescriptorLandingMemDescType(
      storeOp.getOperation(), storeOp.getDesc().getType(),
      storeOp.getSrc().getType(), mutableMemory);
}

inline Value adaptMemDescValue(RewriterBase &rewriter, Location loc,
                               Value value, ttg::MemDescType targetTy,
                               Operation *sqmmaAttrSource = nullptr) {
  auto srcTy = dyn_cast<ttg::MemDescType>(value.getType());
  if (!srcTy)
    return {};
  if (srcTy == targetTy)
    return value;
  if (!areMemDescTypesCompatible(srcTy, targetTy) &&
      !areMemDescTypesLayoutEquivalent(srcTy, targetTy))
    return {};
  auto reinterpret =
      ttg::MemDescReinterpretOp::create(rewriter, loc, targetTy, value);
  if (sqmmaAttrSource)
    copySqmmaAttrs(sqmmaAttrSource, reinterpret.getOperation());
  return reinterpret;
}

inline Value findReusableLocalAllocForSource(Value source,
                                             ttg::MemDescType targetTy) {
  for (Operation *user : source.getUsers()) {
    auto localAlloc = dyn_cast<ttg::LocalAllocOp>(user);
    if (!localAlloc || localAlloc.getSrc() != source)
      continue;
    auto allocTy = dyn_cast<ttg::MemDescType>(localAlloc.getResult().getType());
    if (allocTy != targetTy)
      continue;
    return localAlloc.getResult();
  }
  return {};
}

inline SmallVector<int64_t> invertTrailingPermutation(ArrayRef<int64_t> shape,
                                                      ArrayRef<int32_t> order) {
  SmallVector<int64_t> result;
  auto rank = static_cast<size_t>(order.size());
  if (shape.size() < rank)
    return result;
  result.assign(shape.begin(), shape.end() - rank);

  SmallVector<int64_t> tail(shape.end() - rank, shape.end());
  SmallVector<int64_t> inverse(rank);
  for (auto [idx, permutedIdx] : llvm::enumerate(order))
    inverse[permutedIdx] = tail[idx];
  result.append(inverse.begin(), inverse.end());
  return result;
}

inline ttg::MemDescType
inferTransformedMemDescSourceType(tt::TransOp transOp,
                                  ttg::MemDescType targetTy) {
  auto srcTy = dyn_cast<RankedTensorType>(transOp.getSrc().getType());
  if (!srcTy || !targetTy.getEncoding())
    return {};

  Dialect &dialect = targetTy.getEncoding().getDialect();
  auto inferLayoutInterface =
      dyn_cast<tt::DialectInferLayoutInterface>(&dialect);
  if (!inferLayoutInterface)
    return {};

  SmallVector<int32_t> inverseOrder(transOp.getOrder().size());
  for (auto [idx, permutedIdx] : llvm::enumerate(transOp.getOrder()))
    inverseOrder[permutedIdx] = idx;

  Attribute sourceEncoding;
  if (failed(inferLayoutInterface->inferTransOpEncoding(
          targetTy.getEncoding(), targetTy.getShape(), inverseOrder,
          sourceEncoding, transOp.getLoc()))) {
    return {};
  }

  SmallVector<int64_t> sourceAllocShape;
  if (!targetTy.getAllocShape().empty()) {
    sourceAllocShape =
        invertTrailingPermutation(targetTy.getAllocShape(), transOp.getOrder());
    if (sourceAllocShape.empty())
      return {};
  }

  return ttg::MemDescType::get(srcTy.getShape(), srcTy.getElementType(),
                               sourceEncoding, targetTy.getMemorySpace(),
                               targetTy.getMutableMemory(), sourceAllocShape);
}

inline ttg::MemDescType
inferReshapedMemDescSourceType(tt::ReshapeOp reshapeOp,
                               ttg::MemDescType targetTy) {
  auto srcTy = dyn_cast<RankedTensorType>(reshapeOp.getSrc().getType());
  if (!srcTy)
    return {};

  ttg::MemDescType sourceTy;
  if (failed(ttg::MemDescReshapeOp::inferReturnTypes(
          reshapeOp.getContext(), reshapeOp.getLoc(), targetTy,
          srcTy.getShape(), sourceTy))) {
    return {};
  }
  return sourceTy;
}

inline Value
materializeTransformedMemDesc(RewriterBase &rewriter, tt::TransOp transOp,
                              Value sourceMemDesc,
                              Operation *sqmmaAttrSource = nullptr) {
  SmallVector<int32_t> transposeOrder(transOp.getOrder().begin(),
                                      transOp.getOrder().end());
  auto transformedOp = ttg::MemDescTransOp::create(
      rewriter, transOp.getLoc(), sourceMemDesc, transposeOrder);
  if (sqmmaAttrSource)
    copySqmmaAttrs(sqmmaAttrSource, transformedOp.getOperation());
  return transformedOp.getResult();
}

inline Value materializeTransformedMemDescForTarget(
    RewriterBase &rewriter, tt::TransOp transOp, Value sourceMemDesc,
    ttg::MemDescType targetTy, Operation *sqmmaAttrSource = nullptr) {
  Value transformed = materializeTransformedMemDesc(
      rewriter, transOp, sourceMemDesc, sqmmaAttrSource);
  if (transformed.getType() == targetTy)
    return transformed;
  Value adapted = adaptMemDescValue(rewriter, transOp.getLoc(), transformed,
                                    targetTy, sqmmaAttrSource);
  if (adapted)
    return adapted;
  transformed.getDefiningOp()->erase();
  return {};
}

inline Value materializeReshapedMemDesc(RewriterBase &rewriter,
                                        tt::ReshapeOp reshapeOp,
                                        Value sourceMemDesc,
                                        Operation *sqmmaAttrSource = nullptr) {
  auto resultTy = cast<RankedTensorType>(reshapeOp.getResult().getType());
  auto transformedOp = ttg::MemDescReshapeOp::create(
      rewriter, reshapeOp.getLoc(), sourceMemDesc, resultTy.getShape());
  if (sqmmaAttrSource)
    copySqmmaAttrs(sqmmaAttrSource, transformedOp.getOperation());
  return transformedOp.getResult();
}

inline Value materializeReshapedMemDescForTarget(
    RewriterBase &rewriter, tt::ReshapeOp reshapeOp, Value sourceMemDesc,
    ttg::MemDescType targetTy, Operation *sqmmaAttrSource = nullptr) {
  Value transformed = materializeReshapedMemDesc(
      rewriter, reshapeOp, sourceMemDesc, sqmmaAttrSource);
  if (transformed.getType() == targetTy)
    return transformed;
  Value adapted = adaptMemDescValue(rewriter, reshapeOp.getLoc(), transformed,
                                    targetTy, sqmmaAttrSource);
  if (adapted)
    return adapted;
  transformed.getDefiningOp()->erase();
  return {};
}

inline bool replaceTensorLocalAllocWithMemDesc(RewriterBase &rewriter,
                                               Operation *user,
                                               Value sourceMemDesc) {
  auto localAlloc = dyn_cast<ttg::LocalAllocOp>(user);
  if (!localAlloc)
    return false;
  auto targetTy = dyn_cast<ttg::MemDescType>(localAlloc.getResult().getType());
  if (!targetTy)
    return false;
  OpBuilder::InsertionGuard guard(rewriter);
  rewriter.setInsertionPoint(localAlloc);
  Value replacement =
      adaptMemDescValue(rewriter, localAlloc.getLoc(), sourceMemDesc, targetTy,
                        localAlloc.getOperation());
  if (!replacement)
    return false;
  rewriter.replaceOp(localAlloc, replacement);
  return true;
}

inline bool isSqmmaMemDescLocalAlloc(ttg::LocalAllocOp localAlloc) {
  return localAlloc && localAlloc.getSrc() &&
         hasSqmmaOpIdxAttr(localAlloc.getOperation()) &&
         isa<ttg::MemDescType>(localAlloc.getType());
}

inline void replaceSqmmaLocalAllocWithMemDesc(IRRewriter &rewriter,
                                              ttg::LocalAllocOp localAlloc,
                                              Value replacement) {
  assert(replacement && "expected a materialized SQMMA operand memdesc");
  replaceUsesAndPropagateType(rewriter, localAlloc, replacement);
  localAlloc.erase();
}

inline bool replaceSqmmaTensorUsersWithMemDesc(IRRewriter &rewriter,
                                               Value tensorValue,
                                               Value sourceMemDesc) {
  bool changed = false;
  SmallVector<Operation *> users(tensorValue.getUsers().begin(),
                                 tensorValue.getUsers().end());
  for (Operation *user : users) {
    if (auto localAlloc = dyn_cast<ttg::LocalAllocOp>(user)) {
      if (!isSqmmaMemDescLocalAlloc(localAlloc))
        continue;
      auto targetTy = cast<ttg::MemDescType>(localAlloc.getType());
      rewriter.setInsertionPoint(localAlloc);
      Value replacement =
          adaptMemDescValue(rewriter, localAlloc.getLoc(), sourceMemDesc,
                            targetTy, localAlloc.getOperation());
      if (!replacement)
        continue;
      replaceSqmmaLocalAllocWithMemDesc(rewriter, localAlloc, replacement);
      changed = true;
      continue;
    }

    if (auto transOp = dyn_cast<tt::TransOp>(user)) {
      for (Operation *viewUser : llvm::make_early_inc_range(user->getUsers())) {
        auto localAlloc = dyn_cast<ttg::LocalAllocOp>(viewUser);
        if (!isSqmmaMemDescLocalAlloc(localAlloc))
          continue;
        rewriter.setInsertionPoint(localAlloc);
        Value replacement = materializeTransformedMemDesc(
            rewriter, transOp, sourceMemDesc, localAlloc.getOperation());
        replaceSqmmaLocalAllocWithMemDesc(rewriter, localAlloc, replacement);
        changed = true;
      }
      if (user->use_empty())
        rewriter.eraseOp(user);
      continue;
    }

    if (auto reshapeOp = dyn_cast<tt::ReshapeOp>(user)) {
      for (Operation *viewUser : llvm::make_early_inc_range(user->getUsers())) {
        auto localAlloc = dyn_cast<ttg::LocalAllocOp>(viewUser);
        if (!isSqmmaMemDescLocalAlloc(localAlloc))
          continue;
        auto targetTy = cast<ttg::MemDescType>(localAlloc.getType());
        rewriter.setInsertionPoint(localAlloc);
        Value replacement = materializeReshapedMemDescForTarget(
            rewriter, reshapeOp, sourceMemDesc, targetTy,
            localAlloc.getOperation());
        if (!replacement)
          continue;
        replaceSqmmaLocalAllocWithMemDesc(rewriter, localAlloc, replacement);
        changed = true;
      }
      if (user->use_empty())
        rewriter.eraseOp(user);
      continue;
    }
  }
  return changed;
}

inline bool tryReplaceTensorUserWithMemDesc(RewriterBase &rewriter,
                                            Value tensorValue,
                                            Value sourceMemDesc,
                                            Operation *user) {
  if (replaceTensorLocalAllocWithMemDesc(rewriter, user, sourceMemDesc))
    return true;

  if (auto transOp = dyn_cast<tt::TransOp>(user)) {
    bool changed = false;
    for (Operation *transUser :
         llvm::make_early_inc_range(transOp->getUsers())) {
      auto localAlloc = dyn_cast<ttg::LocalAllocOp>(transUser);
      if (!localAlloc)
        continue;
      auto targetTy = dyn_cast<ttg::MemDescType>(localAlloc.getType());
      if (!targetTy)
        continue;
      OpBuilder::InsertionGuard guard(rewriter);
      rewriter.setInsertionPoint(localAlloc);
      Value replacement = materializeTransformedMemDescForTarget(
          rewriter, transOp, sourceMemDesc, targetTy,
          localAlloc.getOperation());
      if (!replacement)
        continue;
      rewriter.replaceOp(localAlloc, replacement);
      changed = true;
    }
    return changed;
  }

  if (auto reshapeOp = dyn_cast<tt::ReshapeOp>(user)) {
    bool changed = false;
    for (Operation *reshapeUser :
         llvm::make_early_inc_range(reshapeOp->getUsers())) {
      auto localAlloc = dyn_cast<ttg::LocalAllocOp>(reshapeUser);
      if (!localAlloc)
        continue;
      auto targetTy = dyn_cast<ttg::MemDescType>(localAlloc.getType());
      if (!targetTy)
        continue;
      OpBuilder::InsertionGuard guard(rewriter);
      rewriter.setInsertionPoint(localAlloc);
      Value replacement = materializeReshapedMemDescForTarget(
          rewriter, reshapeOp, sourceMemDesc, targetTy,
          localAlloc.getOperation());
      if (!replacement)
        continue;
      rewriter.replaceOp(localAlloc, replacement);
      changed = true;
    }
    return changed;
  }

  (void)tensorValue;
  return false;
}

} // namespace mlir::triton::musa

#endif // TRITONMUSA_COMMON_MEMDESC_UTILS_H
