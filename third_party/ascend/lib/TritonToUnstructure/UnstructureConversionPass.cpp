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

#include "TritonToUnstructure/UnstructureConversionPass.h"
#include "TritonToLinalg/MaskAnalysis.h"
#include "TritonToStructured/CannonicalizerConverter.h"
#include "TritonToUnstructure/IndirectAtomicUtils.h"
#include "Utils/Utils.h"
#include "triton/Dialect/Triton/IR/Dialect.h"

#include "bishengir/Dialect/Annotation/IR/Annotation.h"
#include "bishengir/Dialect/HIVM/IR/HIVM.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "mlir/Transforms/Passes.h"

#include "llvm/ADT/STLExtras.h"

#include <optional>

#define DEBUG_TYPE "triton-unstructure-converter"

using namespace mlir;
using namespace triton;

#include "llvm/Support/Debug.h"

static triton::ascend::CompileMode unstructureCompileMode =
    triton::ascend::CompileMode::Simd;

namespace {

static bool isTensorOfPointers(Type type) {
  auto tensorType = dyn_cast<RankedTensorType>(type);
  return tensorType && isa<triton::PointerType>(tensorType.getElementType());
}

constexpr int64_t kBitsPerByte = 8;

static triton::PointerType getScalarPointerType(Type type) {
  if (auto tensorType = dyn_cast<RankedTensorType>(type))
    type = tensorType.getElementType();
  return dyn_cast<triton::PointerType>(type);
}

// Keep this deliberately narrow. These are the byte-buffer views used by the
// paged-attention and KV-cache kernels covered by this canonicalization.
static std::optional<int64_t>
getSupportedPointerBitcastScale(triton::BitcastOp op) {
  auto srcType = getScalarPointerType(op.getSrc().getType());
  auto dstType = getScalarPointerType(op.getType());
  if (!srcType || !dstType ||
      srcType.getAddressSpace() != dstType.getAddressSpace() ||
      !srcType.getPointeeType().isInteger(8))
    return std::nullopt;

  Type dstPointeeType = dstType.getPointeeType();
  if (dstPointeeType.isF16() || dstPointeeType.isBF16())
    return 2;
  if (dstPointeeType.isF32() || dstPointeeType.isInteger(32))
    return 4;
  return std::nullopt;
}

static FailureOr<Value> scaleTensorPointerOffset(Value offset, int64_t divisor,
                                                 IRRewriter &rewriter) {
  auto offsetType = dyn_cast<RankedTensorType>(offset.getType());
  if (!offsetType || !offsetType.hasStaticShape())
    return failure();
  auto elementType = dyn_cast<IntegerType>(offsetType.getElementType());
  if (!elementType ||
      (elementType.getWidth() != 32 && elementType.getWidth() != 64))
    return failure();
  auto divisorAttr = rewriter.getIntegerAttr(elementType, divisor);
  Value divisorValue = rewriter.create<arith::ConstantOp>(
      offset.getLoc(), DenseElementsAttr::get(offsetType, divisorAttr));
  return rewriter.create<arith::DivSIOp>(offset.getLoc(), offset, divisorValue)
      .getResult();
}

static bool isPublicKernelArgument(Value value) {
  auto blockArgument = dyn_cast<BlockArgument>(value);
  if (!blockArgument)
    return false;
  auto funcOp =
      dyn_cast<triton::FuncOp>(blockArgument.getOwner()->getParentOp());
  return funcOp && funcOp.getVisibility() == SymbolTable::Visibility::Public;
}

static FailureOr<Value> materializeScalarByteAddress(Value pointer,
                                                     Location loc,
                                                     IRRewriter &rewriter) {
  SmallVector<Value> byteOffsets;
  Value root = pointer;
  while (auto addPtrOp = root.getDefiningOp<triton::AddPtrOp>()) {
    auto offsetType = dyn_cast<IntegerType>(addPtrOp.getOffset().getType());
    if (isa<RankedTensorType>(addPtrOp.getType()) || !offsetType ||
        (offsetType.getWidth() != 32 && offsetType.getWidth() != 64))
      return failure();
    byteOffsets.push_back(addPtrOp.getOffset());
    root = addPtrOp.getPtr();
  }
  if (!isPublicKernelArgument(root))
    return failure();

  Value address =
      rewriter.create<triton::PtrToIntOp>(loc, rewriter.getI64Type(), root);
  for (Value offset : llvm::reverse(byteOffsets)) {
    auto offsetType = cast<IntegerType>(offset.getType());
    if (offsetType.getWidth() < 64)
      offset =
          rewriter.create<arith::ExtSIOp>(loc, rewriter.getI64Type(), offset);
    address = rewriter.create<arith::AddIOp>(loc, address, offset);
  }
  return address;
}

static bool hasOnlySupportedPointerUses(Value pointer) {
  if (pointer.use_empty())
    return false;
  for (Operation *user : pointer.getUsers()) {
    if (auto addPtrOp = dyn_cast<triton::AddPtrOp>(user)) {
      if (addPtrOp.getPtr() != pointer ||
          !hasOnlySupportedPointerUses(addPtrOp.getResult()))
        return false;
      continue;
    }
    if (auto splatOp = dyn_cast<triton::SplatOp>(user)) {
      if (splatOp.getSrc() != pointer ||
          !hasOnlySupportedPointerUses(splatOp.getResult()))
        return false;
      continue;
    }
    if (auto expandDimsOp = dyn_cast<triton::ExpandDimsOp>(user)) {
      if (expandDimsOp.getSrc() != pointer ||
          !hasOnlySupportedPointerUses(expandDimsOp.getResult()))
        return false;
      continue;
    }
    if (auto broadcastOp = dyn_cast<triton::BroadcastOp>(user)) {
      if (broadcastOp.getSrc() != pointer ||
          !hasOnlySupportedPointerUses(broadcastOp.getResult()))
        return false;
      continue;
    }
    if (auto loadOp = dyn_cast<triton::LoadOp>(user)) {
      if (loadOp.getPtr() != pointer)
        return false;
      continue;
    }
    if (auto storeOp = dyn_cast<triton::StoreOp>(user)) {
      if (storeOp.getPtr() != pointer)
        return false;
      continue;
    }
    return false;
  }
  return true;
}

static bool canRewriteSupportedPointerBitcast(triton::BitcastOp op) {
  if (!getSupportedPointerBitcastScale(op) ||
      !hasOnlySupportedPointerUses(op.getResult()))
    return false;

  Value current = op.getSrc();
  while (Operation *producer = current.getDefiningOp()) {
    if (auto addPtrOp = dyn_cast<triton::AddPtrOp>(producer)) {
      Type offsetType = addPtrOp.getOffset().getType();
      if (isa<RankedTensorType>(current.getType())) {
        auto pointerType = cast<RankedTensorType>(current.getType());
        auto tensorType = dyn_cast<RankedTensorType>(offsetType);
        if (!pointerType.hasStaticShape() || !tensorType ||
            !tensorType.hasStaticShape() ||
            pointerType.getShape() != tensorType.getShape())
          return false;
        auto elementType = dyn_cast<IntegerType>(tensorType.getElementType());
        if (!elementType ||
            (elementType.getWidth() != 32 && elementType.getWidth() != 64))
          return false;
      } else if (auto intType = dyn_cast<IntegerType>(offsetType);
                 !intType ||
                 (intType.getWidth() != 32 && intType.getWidth() != 64)) {
        return false;
      }
      current = addPtrOp.getPtr();
      continue;
    }
    if (auto splatOp = dyn_cast<triton::SplatOp>(producer)) {
      if (isa<RankedTensorType>(splatOp.getSrc().getType()))
        return false;
      current = splatOp.getSrc();
      continue;
    }
    return false;
  }
  return !isa<RankedTensorType>(current.getType()) &&
         isPublicKernelArgument(current);
}

// Rewrite only the supported byte-buffer views into ordinary target-element
// addptr operations. All later pointer and control-flow handling remains on the
// existing main-dev paths.
static LogicalResult
rewriteSupportedPointerBitcast(triton::BitcastOp op, IRRewriter &rewriter,
                               triton::BitcastOp &nextBitcast) {
  auto divisor = getSupportedPointerBitcastScale(op);
  if (!divisor)
    return failure();

  Value src = op.getSrc();
  rewriter.setInsertionPoint(op);
  if (!isa<RankedTensorType>(src.getType())) {
    auto address = materializeScalarByteAddress(src, op.getLoc(), rewriter);
    if (failed(address))
      return failure();
    auto targetPtr = rewriter.create<triton::IntToPtrOp>(
        op.getLoc(), op.getType(), *address);
    rewriter.replaceOp(op, targetPtr.getResult());
    nextBitcast = nullptr;
    return success();
  }

  if (src.getDefiningOp<triton::AddPtrOp>()) {
    Value base = src;
    Value byteOffset;
    while (auto currentAddPtr = base.getDefiningOp<triton::AddPtrOp>()) {
      Value currentOffset = currentAddPtr.getOffset();
      auto offsetType = cast<RankedTensorType>(currentOffset.getType());
      auto elementType = cast<IntegerType>(offsetType.getElementType());
      if (elementType.getWidth() < 64) {
        auto i64OffsetType =
            RankedTensorType::get(offsetType.getShape(), rewriter.getI64Type());
        currentOffset = rewriter.create<arith::ExtSIOp>(
            op.getLoc(), i64OffsetType, currentOffset);
      }
      if (byteOffset)
        byteOffset = rewriter.create<arith::AddIOp>(op.getLoc(), byteOffset,
                                                    currentOffset);
      else
        byteOffset = currentOffset;
      base = currentAddPtr.getPtr();
    }
    auto scaledOffset =
        scaleTensorPointerOffset(byteOffset, *divisor, rewriter);
    if (failed(scaledOffset))
      return failure();
    nextBitcast =
        rewriter.create<triton::BitcastOp>(op.getLoc(), op.getType(), base);
    auto newAddPtr = rewriter.create<triton::AddPtrOp>(
        op.getLoc(), op.getType(), nextBitcast, *scaledOffset);
    rewriter.replaceOp(op, newAddPtr.getResult());
    return success();
  }

  if (auto splatOp = src.getDefiningOp<triton::SplatOp>()) {
    auto resultType = dyn_cast<RankedTensorType>(op.getType());
    if (!resultType)
      return failure();
    nextBitcast = rewriter.create<triton::BitcastOp>(
        op.getLoc(), resultType.getElementType(), splatOp.getSrc());
    auto newSplat = rewriter.create<triton::SplatOp>(op.getLoc(), resultType,
                                                     nextBitcast.getResult());
    rewriter.replaceOp(op, newSplat.getResult());
    return success();
  }

  return failure();
}

static void canonicalizeSupportedPointerBitcasts(ModuleOp moduleOp) {
  SmallVector<triton::BitcastOp> bitcasts;
  moduleOp.walk([&](triton::BitcastOp op) {
    if (canRewriteSupportedPointerBitcast(op))
      bitcasts.push_back(op);
  });

  IRRewriter rewriter(moduleOp.getContext());
  for (triton::BitcastOp bitcast : bitcasts) {
    triton::BitcastOp current = bitcast;
    while (current) {
      triton::BitcastOp next;
      if (failed(rewriteSupportedPointerBitcast(current, rewriter, next)))
        break;
      current = next;
    }
  }
}

static RankedTensorType resolvePtrTensorType(Value ptr) {
  auto ptrType = dyn_cast<RankedTensorType>(ptr.getType());
  if (auto ptrPtrType = dyn_cast<triton::PointerType>(ptr.getType())) {
    if (auto ptrTensorType =
            dyn_cast_or_null<RankedTensorType>(ptrPtrType.getPointeeType())) {
      ptrType = ptrTensorType;
    }
  }
  return ptrType;
}

static Type getResultElementType(RankedTensorType ptrType) {
  auto resultElementType = ptrType.getElementType();
  if (auto pointerType =
          dyn_cast<triton::PointerType>(ptrType.getElementType())) {
    resultElementType = pointerType.getPointeeType();
  }
  return resultElementType;
}

static int64_t getTypeSizeInByte(Type type) {
  if (auto intType = dyn_cast<IntegerType>(type)) {
    return intType.getWidth() / kBitsPerByte;
  }
  if (auto floatType = dyn_cast<FloatType>(type)) {
    return floatType.getWidth() / kBitsPerByte;
  }
  llvm_unreachable("Unhandled element type of tensor");
}

template <typename MemAccOpTy>
void normalizeDiscreteMaskAccessForFallback(MemAccOpTy &op,
                                            PtrOffsetInfo &ptrOffsetInfo,
                                            PatternRewriter &rewriter) {
  if (!op->hasAttr(ConverterUtils::discreteMaskAttrName)) {
    return;
  }

  if constexpr (std::is_same_v<MemAccOpTy, triton::StoreOp>) {
    auto selectOp = op.getValue().template getDefiningOp<arith::SelectOp>();
    op = rewriter.replaceOpWithNewOp<triton::StoreOp>(
        op, op.getPtr(), selectOp.getTrueValue(), selectOp.getCondition(),
        op.getCache(), op.getEvict());
  } else if constexpr (std::is_same_v<MemAccOpTy, triton::AtomicRMWOp>) {
    if (auto selectOp = op.getVal().template getDefiningOp<arith::SelectOp>()) {
      op = rewriter.replaceOpWithNewOp<triton::AtomicRMWOp>(
          op, op.getType(), op.getAtomicRmwOp(), op.getPtr(),
          selectOp.getTrueValue(), selectOp.getCondition(), op.getSem(),
          op.getScope());
    }
  }

  rewriter.setInsertionPoint(op);
  ptrOffsetInfo.setUnstructured(ptrOffsetInfo.getRank());
}

// ======================== 950 template-SIMT indirect fast-path lowering
// ========================
// Load/store uses the established indirect-template ABI when its rank and
// pointer requirements are met. Atomic operations use the same gate with their
// indirect custom-op lowering. Other cases fall back to scalar-loop lowering.
// ======================================================================================
static bool canUseIndirectFastPath(Value srcPtr, Value ptrOffset) {
  if (!srcPtr || !ptrOffset)
    return false;
  auto ptrTy = dyn_cast<triton::PointerType>(srcPtr.getType());
  if (!ptrTy || isa<ShapedType>(ptrTy.getPointeeType()))
    return false;
  return isa<RankedTensorType>(ptrOffset.getType());
}

template <typename MemAccOpTy>
LogicalResult tryRewriteIndirectFastPath(MemAccOpTy op, Location loc,
                                         Value srcPtr, Value ptrOffset,
                                         ArrayRef<int64_t> resultShape,
                                         PatternRewriter &rewriter) {
  // Indirect backend operations accept one scalar base plus lane offsets. An
  // opaque tensor base can contain a different pointer in every lane, so it
  // must be handled by the scalar-loop fallback below.
  if (!isa<triton::PointerType>(srcPtr.getType()))
    return failure();

  bool rankWithinIndirectLoadStoreFastPathLimit = resultShape.size() <= 5;

  if (!canUseIndirectFastPath(srcPtr, ptrOffset)) {
    LLVM_DEBUG({
      llvm::dbgs()
          << "Skip SIMT indirect fast path: src must be scalar elem ptr and "
             "offset must be an int tensor (reject block_ptr)\n";
    });
    return failure();
  }

  if constexpr (std::is_same_v<MemAccOpTy, triton::LoadOp>) {
    if (!rankWithinIndirectLoadStoreFastPathLimit) {
      return failure();
    }

    assert(isa<triton::PointerType>(srcPtr.getType()) &&
           "src must be ptr type");
    Value mask = op.getMask();
    Value other = op.getOther();
    auto resultType = op.getType();
    auto indirect = rewriter.create<triton::ascend::IndirectLoadOp>(
        loc, resultType, srcPtr, ptrOffset, mask, other,
        ConverterUtils::requiresVolatileIndirectLoad(op.getPtr(), op));
    rewriter.replaceOp(op, indirect.getResult());
    LLVM_DEBUG({
      auto &os = llvm::dbgs();
      os << "Rewriting tt.load to tt.indirect_load\n";
      os << indirect << "\n";
    });
    return success();
  } else if constexpr (std::is_same_v<MemAccOpTy, triton::StoreOp>) {
    if (!rankWithinIndirectLoadStoreFastPathLimit) {
      return failure();
    }

    assert(isa<triton::PointerType>(srcPtr.getType()) &&
           "src must be ptr type");
    Value value = op.getValue();
    Value mask = op.getMask();

    // For bool store, unwrap ptr<i1> -> ptr<i8> bitcast before creating
    // indirect_store. Keep ptr<i1> so TypeConverter can map it to memref<?xi8>.
    if (auto bitcastOp = srcPtr.getDefiningOp<triton::BitcastOp>()) {
      auto srcPtrTy =
          dyn_cast<triton::PointerType>(bitcastOp.getSrc().getType());
      auto dstPtrTy = dyn_cast<triton::PointerType>(bitcastOp.getType());

      if (srcPtrTy && dstPtrTy && srcPtrTy.getPointeeType().isInteger(1) &&
          dstPtrTy.getPointeeType().isInteger(8)) {
        srcPtr = bitcastOp.getSrc();
      }
    }
    auto indirect = rewriter.create<triton::ascend::IndirectStoreOp>(
        loc, srcPtr, ptrOffset, value, mask);
    rewriter.eraseOp(op);
    LLVM_DEBUG({
      auto &os = llvm::dbgs();
      os << "Rewriting tt.store to tt.indirect_store\n";
      os << indirect << "\n";
    });
    return success();
  } else if constexpr (std::is_same_v<MemAccOpTy, triton::AtomicRMWOp>) {
    assert(isa<triton::PointerType>(srcPtr.getType()) &&
           "src must be ptr type");
    if (!IndirectAtomicUtils::canUseIndirectAtomicFastPath(op, ptrOffset)) {
      return failure();
    }

    auto customResult =
        IndirectAtomicUtils::tryConvertAtomicRmwToIndirectCustom(
            op, srcPtr, ptrOffset, rewriter);
    if (failed(customResult)) {
      return failure();
    }

    rewriter.replaceOp(op, *customResult);
    LLVM_DEBUG({
      auto &os = llvm::dbgs();
      os << "Rewriting tt.atomic_rmw to hivm.hir.custom indirect atomic\n";
    });
    return success();
  } else if constexpr (std::is_same_v<MemAccOpTy, triton::AtomicCASOp>) {
    assert(isa<triton::PointerType>(srcPtr.getType()) &&
           "src must be ptr type");
    if (!IndirectAtomicUtils::canUseIndirectAtomicFastPath(op, ptrOffset)) {
      return failure();
    }

    auto customResult =
        IndirectAtomicUtils::tryConvertAtomicCasToIndirectCustom(
            op, srcPtr, ptrOffset, rewriter);
    if (failed(customResult)) {
      return failure();
    }

    rewriter.replaceOp(op, *customResult);
    LLVM_DEBUG({
      auto &os = llvm::dbgs();
      os << "Rewriting tt.atomic_cas to hivm.hir.custom indirect atomic\n";
    });
    return success();
  }
}

} // namespace

template <typename MemAccOpTy>
bool UnstructuredMemAccessConverter<MemAccOpTy>::checkUnstructureAnnotated(
    MemAccOpTy op, PatternRewriter &rewriter) const {
  return llvm::any_of(op->getUsers(), [&rewriter](Operation *user) {
    auto annotationOp = dyn_cast<annotation::MarkOp>(user);
    if (annotationOp && annotationOp->hasAttr("mayDiscretememaccess")) {
      rewriter.eraseOp(annotationOp);
      return true;
    }
    return false;
  });
}

template <>
bool UnstructuredMemAccessConverter<triton::StoreOp>::checkUnstructureAnnotated(
    triton::StoreOp op, PatternRewriter &rewriter) const {
  return llvm::any_of(op.getValue().getUsers(), [&rewriter](Operation *user) {
    auto annotationOp = dyn_cast<annotation::MarkOp>(user);
    if (annotationOp && annotationOp->hasAttr("mayDiscretememaccess")) {
      rewriter.eraseOp(annotationOp);
      return true;
    }
    return false;
  });
}

// A null result denotes a predicate already enforced by the generated loops.
// Keep this limited to conjunctions and broadcast upper bounds. In particular,
// do not remove scalar gates or predicates with nonzero/unknown loop starts.
static Value getResidualLoadMask(Value mask, ArrayRef<bool> boundedLoopAxes,
                                 PatternRewriter &rewriter) {
  if (auto andOp = mask.getDefiningOp<arith::AndIOp>()) {
    Value lhs = getResidualLoadMask(andOp.getLhs(), boundedLoopAxes, rewriter);
    Value rhs = getResidualLoadMask(andOp.getRhs(), boundedLoopAxes, rewriter);
    if (!lhs)
      return rhs;
    if (!rhs)
      return lhs;
    if (lhs == andOp.getLhs() && rhs == andOp.getRhs())
      return mask;
    return rewriter.create<arith::AndIOp>(andOp.getLoc(), lhs, rhs);
  }

  auto broadcast = mask.getDefiningOp<triton::BroadcastOp>();
  if (!broadcast)
    return mask;
  auto sourceType = cast<RankedTensorType>(broadcast.getSrc().getType());
  if (sourceType.getShape().size() != boundedLoopAxes.size())
    return mask;
  bool hasLoopAxis = false;
  for (auto [axis, size] : llvm::enumerate(sourceType.getShape())) {
    if (size == 1)
      continue;
    if (!boundedLoopAxes[axis])
      return mask;
    hasLoopAxis = true;
  }
  if (!hasLoopAxis)
    return mask;

  Value predicate = broadcast.getSrc();
  while (auto expand = predicate.getDefiningOp<triton::ExpandDimsOp>())
    predicate = expand.getSrc();
  auto cmp = predicate.getDefiningOp<arith::CmpIOp>();
  if (!cmp || cmp.getPredicate() != arith::CmpIPredicate::slt)
    return mask;
  return {};
}

template <typename MemAccOpTy>
Value UnstructuredMemAccessConverter<MemAccOpTy>::createExtractOp(
    Location loc, Value value, PatternRewriter &rewriter,
    ArrayRef<OpFoldResult> iterIdx) const {
  if (!value)
    return value;
  SmallVector<Value> indices;
  for (auto idxOfr : iterIdx) {
    auto idx = getValueOrCreateConstantIndexOp(rewriter, loc, idxOfr);
    indices.push_back(idx);
  }
  auto extractedOp = rewriter.create<tensor::ExtractOp>(loc, value, indices);
  extractedOp->setAttr(ConverterUtils::discreteAttrName,
                       UnitAttr::get(rewriter.getContext()));
  return extractedOp;
}

template <typename MemAccOpTy>
Value UnstructuredMemAccessConverter<MemAccOpTy>::createExtractOp(
    Location loc, Value value, PatternRewriter &rewriter,
    ArrayRef<OpFoldResult> offsets, ArrayRef<OpFoldResult> sizes,
    ArrayRef<OpFoldResult> strides) const {
  if (!value)
    return value;
  LLVM_DEBUG({
    auto &os = llvm::dbgs();
    os << "Extracting\n";
    os << value << "\n";
  });
  auto extractedOp = rewriter.create<tensor::ExtractSliceOp>(
      loc, value, offsets, sizes, strides);
  extractedOp->setAttr(ConverterUtils::discreteAttrName,
                       UnitAttr::get(rewriter.getContext()));
  return extractedOp;
}

template <>
template <typename... Args>
triton::LoadOp UnstructuredMemAccessConverter<triton::LoadOp>::createMemAccOp(
    triton::LoadOp op, Value ptrToAccess, Location loc,
    PatternRewriter &rewriter, Args &&...args) const {
  return rewriter.create<triton::LoadOp>(loc, ptrToAccess, op.getCache(),
                                         op.getEvict(), op.getIsVolatile());
}

template <>
template <typename... Args>
triton::AtomicRMWOp
UnstructuredMemAccessConverter<triton::AtomicRMWOp>::createMemAccOp(
    triton::AtomicRMWOp op, Value ptrToAccess, Location loc,
    PatternRewriter &rewriter, Args &&...args) const {
  auto extractedValue =
      createExtractOp(loc, op.getVal(), rewriter, std::forward<Args>(args)...);
  auto extractedMask =
      createExtractOp(loc, op.getMask(), rewriter, std::forward<Args>(args)...);
  Type targetType = ptrToAccess.getType();
  if (auto tensorType = dyn_cast<RankedTensorType>(targetType)) {
    auto ptrType = cast<triton::PointerType>(tensorType.getElementType());
    targetType =
        RankedTensorType::get(tensorType.getShape(), ptrType.getPointeeType());
  } else {
    auto resultType = cast<RankedTensorType>(op.getResult().getType());
    SmallVector<int64_t> scalarLikeShape(resultType.getRank(), 1);
    targetType =
        RankedTensorType::get(scalarLikeShape, resultType.getElementType());
    ptrToAccess = rewriter.create<triton::SplatOp>(
        loc, RankedTensorType::get(scalarLikeShape, ptrToAccess.getType()),
        ptrToAccess);
    extractedValue = rewriter.create<triton::SplatOp>(
        loc, RankedTensorType::get(scalarLikeShape, extractedValue.getType()),
        extractedValue);
    if (extractedMask) {
      extractedMask = rewriter.create<triton::SplatOp>(
          loc, RankedTensorType::get(scalarLikeShape, extractedMask.getType()),
          extractedMask);
    }
  }
  return rewriter.create<triton::AtomicRMWOp>(
      loc, targetType, op.getAtomicRmwOpAttr(), ptrToAccess, extractedValue,
      extractedMask, op.getSemAttr(), op.getScopeAttr());
}

template <>
template <typename... Args>
triton::AtomicCASOp
UnstructuredMemAccessConverter<triton::AtomicCASOp>::createMemAccOp(
    triton::AtomicCASOp op, Value ptrToAccess, Location loc,
    PatternRewriter &rewriter, Args &&...args) const {
  auto extractedCmp =
      createExtractOp(loc, op.getCmp(), rewriter, std::forward<Args>(args)...);
  auto extractedValue =
      createExtractOp(loc, op.getVal(), rewriter, std::forward<Args>(args)...);
  Type targetType = ptrToAccess.getType();
  if (auto tensorType = dyn_cast<RankedTensorType>(targetType)) {
    auto ptrType = cast<triton::PointerType>(tensorType.getElementType());
    targetType =
        RankedTensorType::get(tensorType.getShape(), ptrType.getPointeeType());
  } else {
    auto resultType = cast<RankedTensorType>(op.getResult().getType());
    SmallVector<int64_t> scalarLikeShape(resultType.getRank(), 1);
    targetType =
        RankedTensorType::get(scalarLikeShape, resultType.getElementType());
    ptrToAccess = rewriter.create<triton::SplatOp>(
        loc, RankedTensorType::get(scalarLikeShape, ptrToAccess.getType()),
        ptrToAccess);
    extractedCmp = rewriter.create<triton::SplatOp>(
        loc, RankedTensorType::get(scalarLikeShape, extractedCmp.getType()),
        extractedCmp);
    extractedValue = rewriter.create<triton::SplatOp>(
        loc, RankedTensorType::get(scalarLikeShape, extractedValue.getType()),
        extractedValue);
  }
  return rewriter.create<triton::AtomicCASOp>(
      loc, targetType, ptrToAccess, extractedCmp, extractedValue,
      op.getSemAttr(), op.getScopeAttr());
}

template <>
template <typename... Args>
triton::StoreOp UnstructuredMemAccessConverter<triton::StoreOp>::createMemAccOp(
    triton::StoreOp op, Value ptrToAccess, Location loc,
    PatternRewriter &rewriter, Args &&...args) const {
  auto extractedValue = createExtractOp(loc, op.getValue(), rewriter,
                                        std::forward<Args>(args)...);
  auto extractedMask =
      createExtractOp(loc, op.getMask(), rewriter, std::forward<Args>(args)...);
  return rewriter.create<triton::StoreOp>(loc, ptrToAccess, extractedValue,
                                          extractedMask);
}

template <>
template <>
LogicalResult
UnstructuredMemAccessConverter<triton::LoadOp>::splatAndLoadScenario<
    triton::LoadOp>(triton::LoadOp op, const PtrOffsetInfo &ptrOffsetInfo,
                    PatternRewriter &rewriter) const {
  auto loc = op.getLoc();
  if (!ptrOffsetInfo.isPointerDescriptorOwned()) {
    // Preserve the legacy scalar-like path for unmarked pointer tensors. The
    // analyzed-base reconstruction below is a descriptor handoff contract.
    // Provenance is carried by PtrOffsetInfo so supported pointer-preserving
    // operations, such as an ordinary addptr after the rebuild, do not lose
    // ownership; unrelated pointer tensors still keep this legacy path.
    SmallVector<OpFoldResult> indices(ptrOffsetInfo.getRank(),
                                      rewriter.getIndexAttr(0));
    Value extractedPtr = createExtractOp(loc, op.getPtr(), rewriter, indices);
    Value mask = op.getMask();
    Value other = op.getOther();
    Value loadedValue = rewriter.create<triton::LoadOp>(
        loc, extractedPtr, /*mask=*/nullptr, /*other=*/nullptr,
        /*boundaryCheck=*/ArrayRef<int32_t>(),
        /*PaddingOptionAttr=*/nullptr);
    loadedValue = rewriter.create<triton::SplatOp>(
        loc, op.getResult().getType(), loadedValue);
    if (mask)
      rewriter.replaceOpWithNewOp<arith::SelectOp>(op, mask, loadedValue,
                                                   other);
    else
      rewriter.replaceOp(op, loadedValue);
    return success();
  }

  Value scalarBase = ptrOffsetInfo.getPtr();
  Value scalarOffset = ptrOffsetInfo.getOffset();
  if (!scalarBase || !isa<triton::PointerType>(scalarBase.getType()) ||
      !scalarOffset)
    return failure();

  if (auto offsetType = dyn_cast<RankedTensorType>(scalarOffset.getType())) {
    if (offsetType.getRank() != ptrOffsetInfo.getRank() ||
        !isa<IntegerType>(offsetType.getElementType()))
      return failure();
    SmallVector<OpFoldResult> indices(ptrOffsetInfo.getRank(),
                                      rewriter.getIndexAttr(0));
    scalarOffset = createExtractOp(loc, scalarOffset, rewriter, indices);
  } else if (!scalarOffset.getType().isIndex() &&
             !isa<IntegerType>(scalarOffset.getType())) {
    return failure();
  }

  // PtrOffsetInfo already represents the analyzed address as scalarBase plus
  // scalarOffset. Rebuild that scalar pointer directly instead of extracting
  // a pointer lane from the original tensor-of-pointers value. This keeps the
  // pointer producer in the T2L-supported addptr family; only the numerical
  // offset is extracted from a tensor when necessary.
  Value ptrToLoad = rewriter.create<triton::AddPtrOp>(loc, scalarBase.getType(),
                                                      scalarBase, scalarOffset);
  Value mask = op.getMask();
  Value other = op.getOther();
  Value loadedValue = rewriter.create<triton::LoadOp>(
      loc, ptrToLoad, /*mask=*/nullptr, /*other=*/nullptr,
      /*boundaryCheck=*/ArrayRef<int32_t>(),
      /*PaddingOptionAttr=*/nullptr);
  loadedValue = rewriter.create<triton::SplatOp>(loc, op.getResult().getType(),
                                                 loadedValue);
  if (mask)
    rewriter.replaceOpWithNewOp<arith::SelectOp>(op, mask, loadedValue, other);
  else
    rewriter.replaceOp(op, loadedValue);
  return success();
}

template <typename MemAccOpTy>
UnstructuredMemAccessConverter<MemAccOpTy>::UnstructuredMemAccessConverter(
    MLIRContext *context, bool forceScalarizeMode,
    const llvm::DenseMap<Value, PtrOffsetInfo> &offsetMap,
    const llvm::SmallDenseMap<Value, bool> &fromTensorArg)
    : OpRewritePattern<MemAccOpTy>(context),
      forceScalarizeMode(forceScalarizeMode), offsetMap(offsetMap),
      fromTensorArg(fromTensorArg) {}

template <typename MemAccOpTy>
LogicalResult UnstructuredMemAccessConverter<MemAccOpTy>::matchAndRewrite(
    MemAccOpTy op, PatternRewriter &rewriter) const {
  auto loc = op.getLoc();

  auto ptr = op.getPtr();
  auto ptrType = resolvePtrTensorType(ptr);
  auto mixCompileDiscreteMask =
      op->hasAttr(ConverterUtils::mixCompileDiscreteMaskAttrName);
  bool runtimeLoopMask = op->hasAttr(ConverterUtils::runtimeLoopMaskAttrName);

  if (!ptrType || op->hasAttr(ConverterUtils::discreteAttrName))
    return failure();
  if (!offsetMap.contains(ptr))
    return op.emitError() << "PtrOffsetInfo should be computed\n" << ptr;

  auto ptrOffsetInfo = offsetMap.at(ptr);

  // Handle these accesses before any speculative fallback IR is created.
  // A failed match must leave the original operation intact for the final
  // unsupported-mask diagnostic, without repeatedly inserting dead helpers.
  if (runtimeLoopMask) {
    if constexpr (std::is_same_v<MemAccOpTy, triton::LoadOp> ||
                  std::is_same_v<MemAccOpTy, triton::StoreOp>) {
      // A one-element tensor needs only a scalar predicate, including in SIMD
      // mode. Use explicit control flow: the legacy scalar masked-load
      // converter otherwise loads unconditionally and selects afterwards.
      if (ptrType.getNumElements() == 1 &&
          getResultElementType(ptrType).isIntOrFloat() &&
          canUseIndirectFastPath(ptrOffsetInfo.getPtr(),
                                 ptrOffsetInfo.getOffset())) {
        SmallVector<OpFoldResult> indices(ptrType.getRank(),
                                          rewriter.getIndexAttr(0));
        Value offset =
            createExtractOp(loc, ptrOffsetInfo.getOffset(), rewriter, indices);
        Value base = ptrOffsetInfo.getPtr();
        Value scalarPtr = rewriter.create<triton::AddPtrOp>(loc, base.getType(),
                                                            base, offset);
        Value mask = createExtractOp(loc, op.getMask(), rewriter, indices);
        if constexpr (std::is_same_v<MemAccOpTy, triton::LoadOp>) {
          Value other = createExtractOp(loc, op.getOther(), rewriter, indices);
          if (!other)
            other = rewriter.create<arith::ConstantOp>(
                loc, rewriter.getZeroAttr(getResultElementType(ptrType)));
          auto guardedLoad = rewriter.create<scf::IfOp>(
              loc, mask,
              [&](OpBuilder &b, Location loc) {
                auto load =
                    b.create<triton::LoadOp>(loc, scalarPtr, op.getCache(),
                                             op.getEvict(), op.getIsVolatile());
                b.create<scf::YieldOp>(loc, load.getResult());
              },
              [&](OpBuilder &b, Location loc) {
                b.create<scf::YieldOp>(loc, other);
              });
          rewriter.replaceOpWithNewOp<triton::SplatOp>(
              op, op.getType(), guardedLoad.getResult(0));
        } else {
          Value value = createExtractOp(loc, op.getValue(), rewriter, indices);
          rewriter.create<scf::IfOp>(
              loc, mask, [&](OpBuilder &b, Location loc) {
                b.create<triton::StoreOp>(loc, scalarPtr, value, op.getCache(),
                                          op.getEvict());
                b.create<scf::YieldOp>(loc);
              });
          rewriter.eraseOp(op);
        }
        return success();
      }
    }
    if (compileOn91095Flag &&
        triton::ascend::isSimtTemplateMode(unstructureCompileMode))
      return tryRewriteIndirectFastPath(op, loc, ptrOffsetInfo.getPtr(),
                                        ptrOffsetInfo.getOffset(),
                                        ptrType.getShape(), rewriter);
    return rewriter.notifyMatchFailure(
        op, "loop mask requires a supported masked indirect access");
  }

  if (checkUnstructureAnnotated(op, rewriter))
    ptrOffsetInfo.setUnstructured(ptrOffsetInfo.getRank());

  if (ptrOffsetInfo.isStructured() && !mixCompileDiscreteMask &&
      (!ptrOffsetInfo.isScalarLike() ||
       llvm::all_of(ptrType.getShape(), [](int64_t dim) { return dim == 1; })))
    return failure();

  LLVM_DEBUG({
    auto &os = llvm::dbgs();
    os << "Converting " << op->getName() << "\n";
    os << op << "\n";
    for (auto structured : ptrOffsetInfo.getStructuredRef())
      os << static_cast<int>(structured);
    os << "\n";
    os << ptrOffsetInfo.isScalarLike() << "\n";
  });

  if constexpr (std::is_same_v<MemAccOpTy, triton::LoadOp>) {
    if (ptrOffsetInfo.isScalarLike() &&
        succeeded(splatAndLoadScenario(op, ptrOffsetInfo, rewriter))) {
      return success();
    }
  }

  std::optional<MaskState> mstate = runMaskAnalysis(op, rewriter);

  normalizeDiscreteMaskAccessForFallback(op, ptrOffsetInfo, rewriter);

  if (forceScalarizeMode || ptrOffsetInfo.isScalarLike() ||
      fromTensorArg.at(ptr)) {
    ptrOffsetInfo.setUnstructured(ptrOffsetInfo.getRank());
  }

  auto srcPtr = ptrOffsetInfo.getPtr();
  auto ptrOffset = ptrOffsetInfo.getOffset();
  if (!isa<triton::PointerType>(srcPtr.getType()) &&
      !isTensorOfPointers(srcPtr.getType()))
    return rewriter.notifyMatchFailure(
        op, "expected a scalar pointer or a tensor of scalar pointers");

  // LoadLike is operation with result
  bool isLoadLike = !op->use_empty();

  Value zeroIdx =
      rewriter.create<arith::ConstantOp>(loc, rewriter.getIndexAttr(0));
  Value oneIdx =
      rewriter.create<arith::ConstantOp>(loc, rewriter.getIndexAttr(1));
  auto resultShape = ptrType.getShape();
  auto resultElementType = getResultElementType(ptrType);
  int64_t sizeInByte = getTypeSizeInByte(resultElementType);

  for (int i = ptrOffsetInfo.getRank() - 1; i >= 0; i--) {
    if (!ptrOffsetInfo.isStructured(i))
      break;
    sizeInByte *= resultShape[i];
  }

  // Force scalarize if memory is not aligned
  if (sizeInByte % 32 != 0) {
    ptrOffsetInfo.setUnstructured(ptrOffsetInfo.getRank());
  }

  LLVM_DEBUG({
    auto &os = llvm::dbgs();
    os << "UnStructured Flag check:\n";
    os << "ptrOffsetInfo.isStructured: " << ptrOffsetInfo.isStructured()
       << "\n";
    os << "compileOn91095Flag: " << compileOn91095Flag << "\n";
    os << "compileMode: " << static_cast<int>(unstructureCompileMode) << "\n";
  });

  bool templateIndirectFastPathEnabled =
      compileOn91095Flag &&
      triton::ascend::isSimtTemplateMode(unstructureCompileMode) &&
      ((!ptrOffsetInfo.isStructured() && sizeInByte < 64) ||
       mixCompileDiscreteMask);
  bool rankWithinIndirectLoadStoreFastPathLimit = resultShape.size() <= 5;
  if (templateIndirectFastPathEnabled &&
      succeeded(tryRewriteIndirectFastPath(op, loc, srcPtr, ptrOffset,
                                           resultShape, rewriter))) {
    return success();
  }

  LLVM_DEBUG({
    if (triton::ascend::isSimtTemplateMode(unstructureCompileMode) &&
        sizeInByte >= 64) {
      auto &os = llvm::dbgs();
      os << "Skip template-SIMT indirect fast path because continuous shape "
            "product is "
         << sizeInByte << " (>=64)\n";
    }
    if constexpr (std::is_same_v<MemAccOpTy, triton::LoadOp> ||
                  std::is_same_v<MemAccOpTy, triton::StoreOp>) {
      if (templateIndirectFastPathEnabled &&
          !rankWithinIndirectLoadStoreFastPathLimit) {
        auto &os = llvm::dbgs();
        os << "Skip tt.indirect_load/store fast path because rank is "
           << resultShape.size() << " (>5), falling back to scalar loop path\n";
      }
    }
  });

  Value iterArg = nullptr;

  // Only load case
  if (isLoadLike) {
    iterArg =
        rewriter.create<tensor::EmptyOp>(loc, resultShape, resultElementType);
  }
  Value newOpResult = nullptr;

  auto insertPoint = rewriter.saveInsertionPoint();

  SmallVector<OpFoldResult> offsets;
  SmallVector<OpFoldResult> sizes;
  SmallVector<OpFoldResult> strides;
  SmallVector<int64_t> extractedShape;
  SmallVector<bool> boundedLoopAxes(resultShape.size(), false);

  for (size_t i = 0; i < resultShape.size(); i++) {
    auto size = resultShape[i];
    auto structured = ptrOffsetInfo.getStructuredRef()[i] ==
                      PtrOffsetInfo::AxisInfo::structured;
    // handle indirect dimension
    strides.push_back(rewriter.getIndexAttr(1));
    Value sizeVal =
        rewriter.create<arith::ConstantOp>(loc, rewriter.getIndexAttr(size));
    if (structured) {
      offsets.push_back(rewriter.getIndexAttr(0));
      sizes.push_back(rewriter.getIndexAttr(size));
      extractedShape.push_back(size);
    } else {
      scf::ForOp forOp;
      if (auto mtptOp =
              srcPtr.template getDefiningOp<triton::MakeTensorPtrOp>()) {
        auto tptShape = mtptOp.getShape()[i];
        if (tptShape.getType() != rewriter.getIndexType()) {
          tptShape = rewriter.create<arith::IndexCastOp>(
              loc, rewriter.getIndexType(), tptShape);
        }
        sizeVal = rewriter.create<arith::MinSIOp>(loc, sizeVal, tptShape);
      }

      Value loopLower = zeroIdx;
      Value loopUpper = sizeVal;
      if (mstate && i < mstate->dims.size() && i < mstate->offsets.size()) {
        Value maskOffset =
            getValueOrCreateConstantIndexOp(rewriter, loc, mstate->offsets[i]);
        maskOffset = rewriter.create<arith::MaxSIOp>(loc, maskOffset, zeroIdx);
        maskOffset = rewriter.create<arith::MinSIOp>(loc, maskOffset, sizeVal);
        loopLower = maskOffset;

        Value maskDim =
            getValueOrCreateConstantIndexOp(rewriter, loc, mstate->dims[i]);
        maskDim = rewriter.create<arith::AddIOp>(loc, maskOffset, maskDim);
        maskDim = rewriter.create<arith::MinSIOp>(loc, maskDim, sizeVal);
        loopUpper = maskDim;
        // With a zero mask offset, [0, loopUpper) is contained in the
        // analyzed interval. Do not infer this for a clipped negative start:
        // clamping the start can otherwise shift a nonempty interval.
        boundedLoopAxes[i] = mstate->isMask() &&
                             mstate->dims.size() == resultShape.size() &&
                             isConstantIntValue(mstate->offsets[i], 0);
      }

      if (isLoadLike) {
        forOp = rewriter.create<scf::ForOp>(loc, loopLower, loopUpper, oneIdx,
                                            ValueRange({iterArg}));
        if (!newOpResult) {
          newOpResult = forOp->getResult(0);
        } else {
          rewriter.create<scf::YieldOp>(loc, forOp->getResult(0));
        }
        iterArg = forOp.getRegionIterArg(0);
      } else {
        forOp = rewriter.create<scf::ForOp>(loc, loopLower, loopUpper, oneIdx);
      }
      sizes.push_back(rewriter.getIndexAttr(1));
      offsets.push_back(forOp.getInductionVar());
      extractedShape.push_back(1);
      forOp->setAttr("ExtractedLoadOrStore",
                     UnitAttr::get(rewriter.getContext()));
      rewriter.setInsertionPointToStart(forOp.getBody());
    }
  }

  bool fullyUnstructured = ptrOffsetInfo.isUnstructuredOrScalarlike();
  auto extractedType = RankedTensorType::get(extractedShape, resultElementType);

  Value extractedOffset;
  if (fullyUnstructured) {
    if (auto mtptOp =
            srcPtr.template getDefiningOp<triton::MakeTensorPtrOp>()) {
      auto I64Type = rewriter.getIntegerType(64);
      srcPtr = mtptOp.getBase();
      extractedOffset = rewriter.create<arith::ConstantIntOp>(loc, 0, 64);
      for (auto [indVar, offset, stride] : llvm::zip_equal(
               offsets, ptrOffsetInfo.getOffsets(), mtptOp.getStrides())) {
        Value inductionVar = rewriter.create<arith::IndexCastOp>(
            loc, I64Type, cast<Value>(indVar));
        Value tptOffset = rewriter.create<arith::ExtSIOp>(loc, I64Type, offset);
        Value tptStride = rewriter.create<arith::ExtSIOp>(loc, I64Type, stride);
        tptOffset = rewriter.create<arith::MulIOp>(loc, tptStride, tptOffset);
        tptStride =
            rewriter.create<arith::MulIOp>(loc, tptStride, inductionVar);
        extractedOffset =
            rewriter.create<arith::AddIOp>(loc, extractedOffset, tptOffset);
        extractedOffset =
            rewriter.create<arith::AddIOp>(loc, extractedOffset, tptStride);
      }
    } else {
      extractedOffset = createExtractOp(loc, ptrOffset, rewriter, offsets);
    }
  } else {
    extractedOffset =
        createExtractOp(loc, ptrOffset, rewriter, offsets, sizes, strides);
  }

  LLVM_DEBUG({
    auto &os = llvm::dbgs();
    os << "Extracted offset\n";
    os << extractedOffset << "\n";
  });

  // A tensor-of-pointers base is opaque: each lane may select a different
  // allocation. Extract the base with the same scalar or slice coordinates as
  // the offset, then form the access pointer lane by lane.
  if (isTensorOfPointers(srcPtr.getType())) {
    if (fullyUnstructured)
      srcPtr = createExtractOp(loc, srcPtr, rewriter, offsets);
    else
      srcPtr = createExtractOp(loc, srcPtr, rewriter, offsets, sizes, strides);
  }

  assert((isa<triton::PointerType>(srcPtr.getType()) ||
          isTensorOfPointers(srcPtr.getType())) &&
         "src must be a scalar pointer or tensor of pointers");
  if (!fullyUnstructured && isa<triton::PointerType>(srcPtr.getType())) {
    srcPtr = rewriter.create<triton::SplatOp>(
        loc, RankedTensorType::get(extractedShape, srcPtr.getType()), srcPtr);
  }
  Value ptrToAccess = rewriter.create<triton::AddPtrOp>(
      loc, srcPtr.getType(), srcPtr, extractedOffset);

  MemAccOpTy accessedOp;
  if (fullyUnstructured) {
    accessedOp = createMemAccOp(op, ptrToAccess, loc, rewriter, offsets);
  } else {
    accessedOp =
        createMemAccOp(op, ptrToAccess, loc, rewriter, offsets, sizes, strides);
  }
  if constexpr (std::is_same_v<MemAccOpTy, triton::LoadOp>) {
    // Preserve the mask on retained dimensions, just as stores and atomics do.
    // BubbleUpOperation exposes the sliced mask for structured lowering.
    if (mstate && !fullyUnstructured) {
      OpBuilder::InsertionGuard guard(rewriter);
      rewriter.setInsertionPoint(accessedOp);
      Value residualMask =
          getResidualLoadMask(op.getMask(), boundedLoopAxes, rewriter);
      if (residualMask)
        accessedOp.getMaskMutable().assign(createExtractOp(
            loc, residualMask, rewriter, offsets, sizes, strides));
    }
  }

  accessedOp->setAttr(ConverterUtils::discreteAttrName,
                      UnitAttr::get(rewriter.getContext()));

  if (isLoadLike) {
    assert(iterArg && "Load case must have iterArg in for loop");

    Value value = accessedOp->getResult(0);
    Value result;
    if (!isa<RankedTensorType>(value.getType()) &&
        (std::is_same_v<MemAccOpTy, triton::AtomicRMWOp> ||
         std::is_same_v<MemAccOpTy, triton::AtomicCASOp>)) {
      value = rewriter.create<triton::SplatOp>(loc, extractedType, value);
    }
    if (!isa<RankedTensorType>(value.getType())) {
      SmallVector<Value> indices;
      for (auto idxOfr : offsets) {
        auto idx = getValueOrCreateConstantIndexOp(rewriter, loc, idxOfr);
        indices.push_back(idx);
      }
      result = rewriter.create<tensor::InsertOp>(loc, value, iterArg, indices);
    } else {
      result = rewriter.create<tensor::InsertSliceOp>(loc, value, iterArg,
                                                      offsets, sizes, strides);
    }
    rewriter.create<scf::YieldOp>(loc, result)
        ->setAttr(ConverterUtils::discreteAttrName,
                  UnitAttr::get(rewriter.getContext()));
    rewriter.restoreInsertionPoint(insertPoint);
    if constexpr (std::is_same_v<MemAccOpTy, triton::LoadOp>) {
      if (op.getMask() && op.getOther()) {
        rewriter
            .replaceOpWithNewOp<arith::SelectOp>(op, op.getMask(), newOpResult,
                                                 op.getOther())
            ->setAttr(ConverterUtils::discreteAttrName,
                      UnitAttr::get(rewriter.getContext()));
      } else {
        rewriter.replaceOp(op, newOpResult);
      }
    } else {
      rewriter.replaceOp(op, newOpResult);
    }
  } else {
    if constexpr (std::is_same_v<MemAccOpTy, triton::AtomicRMWOp>) {
      if (fullyUnstructured && accessedOp.getMask()) {
        auto mask = createExtractOp(
            loc, accessedOp.getMask(), rewriter,
            SmallVector<OpFoldResult>(ptrOffsetInfo.getRank(),
                                      rewriter.getIndexAttr(0)));
        rewriter.create<scf::IfOp>(loc, mask, [&](OpBuilder &b, Location loc) {
          b.create<triton::AtomicRMWOp>(
               loc, accessedOp.getType(), accessedOp.getAtomicRmwOp(),
               accessedOp.getPtr(), accessedOp.getVal(), nullptr,
               accessedOp.getSem(), accessedOp.getScope())
              ->setAttr(ConverterUtils::discreteAttrName,
                        UnitAttr::get(rewriter.getContext()));
          b.create<scf::YieldOp>(loc);
        });
        rewriter.eraseOp(accessedOp);
      }
    }
    rewriter.eraseOp(op);
  }
  LLVM_DEBUG({
    auto &os = llvm::dbgs();
    os << "After conversion\n"
       << ptrToAccess.getDefiningOp()
              ->template getParentOfType<triton::FuncOp>()
       << "\n";
  });
  return success();
}

void TritonToUnstructurePass::runPreparse(LoopLikeOpInterface op) {
  IRRewriter rewriter(&getContext());
  auto loc = op.getLoc();

  LLVM_DEBUG({
    auto &os = llvm::dbgs();
    os << "Pre-parsing " << op->getName() << "\n" << op << "\n";
  });

  Block::BlockArgListType args;
  ValueRange yields;
  if (auto whileOp = dyn_cast<scf::WhileOp>(op.getOperation())) {
    args = whileOp.getBeforeArguments();
    yields = whileOp.getYieldOp().getOperands();
  } else {
    args = op.getRegionIterArgs();
    yields = op.getYieldedValues();
  }

  for (auto [arg, yield] : llvm::zip_equal(args, yields)) {
    if (auto tensorType = dyn_cast<RankedTensorType>(yield.getType())) {
      parse(yield, loc, rewriter, offsetMapForLoopArgs);
      offsetMap[arg] = offsetMapForLoopArgs.at(yield);
      LLVM_DEBUG({
        auto &os = llvm::dbgs();
        os << "Pre-parsing result of\n" << arg << "\nis ";
        for (auto structured : offsetMap[arg].getStructuredRef())
          os << static_cast<int>(structured);
        os << '\n';
      });
    }
  }
}

static bool isFromTensorArg(Value v,
                            llvm::SmallDenseMap<Value, bool> &fromTensorArg) {
  if (fromTensorArg.contains(v))
    return fromTensorArg.at(v);
  auto *defOp = v.getDefiningOp();
  if (!defOp) {
    fromTensorArg[v] = isa<RankedTensorType>(v.getType());
    return isa<RankedTensorType>(v.getType());
  }
  for (auto opr : defOp->getOperands()) {
    if (isFromTensorArg(opr, fromTensorArg)) {
      fromTensorArg[v] = true;
      return true;
    }
  }
  fromTensorArg[v] = false;
  return false;
}

template <typename MemAccOpTy, typename>
void TritonToUnstructurePass::runParse(MemAccOpTy op) {
  IRRewriter rewriter(&getContext());
  LLVM_DEBUG({
    auto &os = llvm::dbgs();
    os << "Parsing " << op->getName() << "\n" << op << "\n";
  });
  parse(op.getPtr(), op.getLoc(), rewriter, offsetMap);
  isFromTensorArg(op.getPtr(), fromTensorArg);
}

LogicalResult
TritonToUnstructurePass::processIfYieldAddHoistOperations(ModuleOp moduleOp) {
  mlir::RewritePatternSet patterns(&getContext());
  patterns.add<CannonicalizerConverter::IfYieldAddHoistConverter>(
      patterns.getContext());
  if (failed(applyPatternsGreedily(moduleOp, std::move(patterns)))) {
    moduleOp.emitWarning("IfYieldAddHoist processing failed");
    return failure();
  }
  return success();
}

TritonToUnstructurePass::TritonToUnstructurePass(
    const TritonToUnstructureOptions &options)
    : TritonToUnstructureBase(options) {}

void TritonToUnstructurePass::runOnOperation() {
  compileOn91095Flag = this->compileOn91095;
  auto compileMode = triton::ascend::parseCompileMode(this->compileMode);
  if (!compileMode) {
    getOperation().emitError()
        << "triton-to-unstructure compile-mode is invalid: "
        << this->compileMode;
    signalPassFailure();
    return;
  }
  unstructureCompileMode = *compileMode;

  LLVM_DEBUG({
    auto &os = llvm::dbgs();
    os << "TritonToUnstructurePass started with options:\n";
    os << "  compileOn91095: " << compileOn91095Flag << "\n";
    os << "  compileMode: " << this->compileMode << "\n";
  });

  ModuleOp moduleOp = getOperation();
  MLIRContext *ctx = &getContext();

  canonicalizeSupportedPointerBitcasts(moduleOp);

  moduleOp->walk([this](triton::FuncOp funcOp) {
    replacePtrArguments(funcOp, offsetMapForLoopArgs);
  });
  offsetMapForLoopArgs.clear();

  if (failed(processIfYieldAddHoistOperations(moduleOp))) {
    moduleOp.emitWarning("Failed to process IfYieldAddHoist operations");
  }

  moduleOp->walk([this](LoopLikeOpInterface op) { runPreparse(op); });
  moduleOp->walk([this](Operation *op) {
    if (auto loadOp = dyn_cast<triton::LoadOp>(op)) {
      runParse(loadOp);
    } else if (auto storeOp = dyn_cast<triton::StoreOp>(op)) {
      runParse(storeOp);
    } else if (auto atomicRMWOp = dyn_cast<triton::AtomicRMWOp>(op)) {
      runParse(atomicRMWOp);
    } else if (auto atomicCASOp = dyn_cast<triton::AtomicCASOp>(op)) {
      runParse(atomicCASOp);
    }
  });

  RewritePatternSet patterns(ctx);

  patterns.add<UnstructuredMemAccessConverter<triton::LoadOp>,
               UnstructuredMemAccessConverter<triton::StoreOp>,
               UnstructuredMemAccessConverter<triton::AtomicRMWOp>,
               UnstructuredMemAccessConverter<triton::AtomicCASOp>>(
      ctx, forceScalarizeMode, offsetMap, fromTensorArg);

  LLVM_DEBUG({
    auto &os = llvm::dbgs();
    os << "Parsing done\n";
  });

  if (failed(applyPatternsGreedily(moduleOp, std::move(patterns)))) {
    moduleOp->emitError("failed to apply Patterns");
    signalPassFailure();
    return;
  }

  bool unsupportedLoopMask = false;
  moduleOp.walk([&](Operation *op) {
    if (op->hasAttr(ConverterUtils::runtimeLoopMaskAttrName)) {
      op->emitError("cannot lower this loop-carried mask with a masked "
                    "indirect access for the selected target and compile mode");
      unsupportedLoopMask = true;
    }
  });
  if (unsupportedLoopMask) {
    signalPassFailure();
    return;
  }

  // The offset-boundary marker is only an intra-pass analysis handoff.  Do
  // not expose it to T2L or let it inhibit later canonicalization.
  moduleOp->walk(
      [](Operation *op) { op->removeAttr(kScalarPointerOffsetBoundaryAttr); });

  PassManager pm(&getContext(), moduleOp.getOperationName());
  pm.addPass(createCSEPass());
  pm.addPass(createCanonicalizerPass());
  if (failed(runPipeline(pm, getOperation()))) {
    signalPassFailure();
  }
}

void TritonToUnstructurePass::getDependentDialects(
    DialectRegistry &registry) const {
  registry.insert<func::FuncDialect, arith::ArithDialect, linalg::LinalgDialect,
                  affine::AffineDialect, scf::SCFDialect, tensor::TensorDialect,
                  bufferization::BufferizationDialect, memref::MemRefDialect,
                  triton::TritonDialect, triton::ascend::TritonAscendDialect,
                  hivm::HIVMDialect>();
}

std::unique_ptr<OperationPass<ModuleOp>> triton::createTritonToUnstructurePass(
    const TritonToUnstructureOptions &options) {
  return std::make_unique<TritonToUnstructurePass>(options);
}
