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

#include "TritonToStructured/MemOpConverter.h"

#include <cassert>
#include <numeric>
#include <type_traits>

#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Arith/Utils/Utils.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/Passes.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Utils/ReshapeOpsUtils.h"
#include "mlir/Dialect/Utils/StaticValueUtils.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypeInterfaces.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Location.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/Value.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/Triton/IR/Types.h"

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/SmallVectorExtras.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/FormatVariadic.h"
#include "llvm/Support/MathExtras.h"

#include "llvm/Support/Debug.h"

#include "TritonToStructured/CannonicalizerConverter.h"
#include "TritonToStructured/MaskAnalysis.h"
#include "TritonToStructured/PtrAnalysis.h"
#include "TritonToStructured/TritonToStructuredPass.h"
#include "Utils/InterleaveOptimization.h"
#include "Utils/Utils.h"
#include "bishengir/Dialect/Annotation/IR/Annotation.h"
#include "bishengir/Dialect/HIVM/Utils/Utils.h"

#define DEBUG_TYPE "triton-mem-op-converter"

namespace MemOpConverter {
using namespace mlir;
using namespace triton;
using namespace TritonToStructured;

namespace {

struct CatPointerShape {
  int64_t lhsSize;
  int64_t rhsSize;

  int64_t totalSize() const { return lhsSize + rhsSize; }
};

struct SplitValues {
  Value lhs;
  Value rhs;
};

FailureOr<CatPointerShape> getCatPointerShape(triton::CatOp catOp) {
  auto lhsType = dyn_cast<RankedTensorType>(catOp.getLhs().getType());
  auto rhsType = dyn_cast<RankedTensorType>(catOp.getRhs().getType());
  auto resultType = dyn_cast<RankedTensorType>(catOp.getType());
  if (!lhsType || !rhsType || !resultType || lhsType.getRank() != 1 ||
      rhsType.getRank() != 1 || resultType.getRank() != 1 ||
      !lhsType.hasStaticShape() || !rhsType.hasStaticShape() ||
      !resultType.hasStaticShape() ||
      !isa<triton::PointerType>(lhsType.getElementType()) ||
      lhsType.getElementType() != rhsType.getElementType() ||
      lhsType.getElementType() != resultType.getElementType()) {
    return failure();
  }

  CatPointerShape shape{lhsType.getDimSize(0), rhsType.getDimSize(0)};
  if (shape.lhsSize != shape.rhsSize ||
      resultType.getDimSize(0) != shape.totalSize())
    return failure();
  return shape;
}

bool isCatAlignedValue(Value value, int64_t totalSize) {
  if (!value)
    return true;
  auto type = dyn_cast<RankedTensorType>(value.getType());
  return type && type.getRank() == 1 && type.hasStaticShape() &&
         type.getDimSize(0) == totalSize;
}

SplitValues splitCatAlignedValue(Value value, const CatPointerShape &shape,
                                 Location loc, PatternRewriter &rewriter) {
  if (!value)
    return {};

  auto valueType = cast<RankedTensorType>(value.getType());
  Type elementType = valueType.getElementType();
  // Ascend's downstream tensor layout requires byte-aligned mask slices.
  // Convert back to i1 only after selecting the contiguous halves.
  if (elementType.isInteger(1)) {
    auto byteType = valueType.clone(rewriter.getI8Type());
    Value byteValue = rewriter.create<arith::ExtUIOp>(loc, byteType, value);
    auto byteHalfType =
        RankedTensorType::get({shape.lhsSize}, rewriter.getI8Type());
    SmallVector<OpFoldResult> offsets{rewriter.getIndexAttr(0)};
    SmallVector<OpFoldResult> sizes{rewriter.getIndexAttr(shape.lhsSize)};
    SmallVector<OpFoldResult> strides{rewriter.getIndexAttr(1)};
    Value lhsByte = rewriter.create<tensor::ExtractSliceOp>(
        loc, byteHalfType, byteValue, offsets, sizes, strides);
    offsets[0] = rewriter.getIndexAttr(shape.lhsSize);
    Value rhsByte = rewriter.create<tensor::ExtractSliceOp>(
        loc, byteHalfType, byteValue, offsets, sizes, strides);
    auto maskHalfType =
        RankedTensorType::get({shape.lhsSize}, rewriter.getI1Type());
    Value lhs = rewriter.create<arith::TruncIOp>(loc, maskHalfType, lhsByte);
    Value rhs = rewriter.create<arith::TruncIOp>(loc, maskHalfType, rhsByte);
    return {lhs, rhs};
  }
  auto reshapedType = RankedTensorType::get({2, shape.lhsSize}, elementType);
  auto transposedType = RankedTensorType::get({shape.lhsSize, 2}, elementType);
  auto halfType = RankedTensorType::get({shape.lhsSize}, elementType);

  Value reshaped = rewriter.create<triton::ReshapeOp>(loc, reshapedType, value);
  SmallVector<int32_t> order{1, 0};
  Value transposed =
      rewriter.create<triton::TransOp>(loc, transposedType, reshaped, order);
  auto split =
      rewriter.create<triton::SplitOp>(loc, halfType, halfType, transposed);
  return {split.getOutLHS(), split.getOutRHS()};
}

void copyMissingAttrs(Operation *source, Operation *target) {
  for (NamedAttribute attr : source->getAttrs()) {
    if (!target->hasAttr(attr.getName()))
      target->setAttr(attr.getName(), attr.getValue());
  }
}

LogicalResult rewriteCatPointerLoad(triton::LoadOp op, triton::CatOp catOp,
                                    PatternRewriter &rewriter) {
  FailureOr<CatPointerShape> shape = getCatPointerShape(catOp);
  if (failed(shape) || !isCatAlignedValue(op.getMask(), shape->totalSize()) ||
      !isCatAlignedValue(op.getOther(), shape->totalSize())) {
    return failure();
  }

  Location loc = op.getLoc();
  SplitValues masks = splitCatAlignedValue(op.getMask(), *shape, loc, rewriter);
  SplitValues others =
      splitCatAlignedValue(op.getOther(), *shape, loc, rewriter);
  auto lhsLoad = rewriter.create<triton::LoadOp>(
      loc, catOp.getLhs(), masks.lhs, others.lhs, op.getBoundaryCheck(),
      op.getPadding(), op.getCache(), op.getEvict(), op.getIsVolatile());
  auto rhsLoad = rewriter.create<triton::LoadOp>(
      loc, catOp.getRhs(), masks.rhs, others.rhs, op.getBoundaryCheck(),
      op.getPadding(), op.getCache(), op.getEvict(), op.getIsVolatile());
  copyMissingAttrs(op.getOperation(), lhsLoad.getOperation());
  copyMissingAttrs(op.getOperation(), rhsLoad.getOperation());

  auto result = rewriter.create<triton::CatOp>(
      loc, op.getType(), ValueRange{lhsLoad.getResult(), rhsLoad.getResult()});
  copyMissingAttrs(catOp.getOperation(), result.getOperation());
  rewriter.replaceOp(op, result.getResult());
  if (catOp.getResult().use_empty())
    rewriter.eraseOp(catOp);
  return success();
}

LogicalResult rewriteCatPointerStore(triton::StoreOp op, triton::CatOp catOp,
                                     PatternRewriter &rewriter) {
  FailureOr<CatPointerShape> shape = getCatPointerShape(catOp);
  if (failed(shape) || !isCatAlignedValue(op.getValue(), shape->totalSize()) ||
      !isCatAlignedValue(op.getMask(), shape->totalSize())) {
    return failure();
  }

  Location loc = op.getLoc();
  SplitValues values =
      splitCatAlignedValue(op.getValue(), *shape, loc, rewriter);
  SplitValues masks = splitCatAlignedValue(op.getMask(), *shape, loc, rewriter);
  auto lhsStore = rewriter.create<triton::StoreOp>(
      loc, catOp.getLhs(), values.lhs, masks.lhs, op.getBoundaryCheck(),
      op.getCache(), op.getEvict());
  auto rhsStore = rewriter.create<triton::StoreOp>(
      loc, catOp.getRhs(), values.rhs, masks.rhs, op.getBoundaryCheck(),
      op.getCache(), op.getEvict());
  copyMissingAttrs(op.getOperation(), lhsStore.getOperation());
  copyMissingAttrs(op.getOperation(), rhsStore.getOperation());

  rewriter.eraseOp(op);
  if (catOp.getResult().use_empty())
    rewriter.eraseOp(catOp);
  return success();
}

// TTIR requires a store value to have the same shape as its pointer. A DSL
// scalar is consequently represented either by tt.splat of a scalar or by a
// folded dense splat constant.
bool isUniformScalarStoreValue(Value value) {
  if (auto splat = value.getDefiningOp<triton::SplatOp>())
    return !isa<ShapedType>(splat.getSrc().getType());

  if (auto constant = value.getDefiningOp<arith::ConstantOp>()) {
    if (auto dense = dyn_cast<DenseElementsAttr>(constant.getValue()))
      return dense.isSplat();
  }
  return false;
}

// Return true only for the store form that must be left to the later pointer
// broadcast lowering. A non-singleton zero-stride pointer dimension aliases
// lanes, so deferring is sound only when the DSL supplied one scalar value.
bool shouldDeferScalarPointerBroadcastStore(Value ptr, Value value) {
  if (!isUniformScalarStoreValue(value))
    return false;

  auto broadcast = ptr.getDefiningOp<triton::BroadcastOp>();
  if (!broadcast)
    return false;

  auto resultType = dyn_cast<RankedTensorType>(broadcast.getType());
  auto sourceType = dyn_cast<RankedTensorType>(broadcast.getSrc().getType());
  if (!resultType || !sourceType ||
      !isa<triton::PointerType>(resultType.getElementType()) ||
      resultType.getRank() != sourceType.getRank()) {
    return false;
  }

  int broadcastedDims = 0;
  for (size_t i = 0; i < resultType.getShape().size(); ++i) {
    const int64_t srcDim = sourceType.getShape()[i];
    const int64_t resultDim = resultType.getShape()[i];
    if (srcDim == 1 && resultDim != 1) {
      ++broadcastedDims;
      continue;
    }
    if (srcDim != resultDim)
      return false;
  }
  if (broadcastedDims != 1)
    return false;

  Value source = broadcast.getSrc();
  while (auto addPtr = source.getDefiningOp<triton::AddPtrOp>())
    source = addPtr.getPtr();
  auto splat = source.getDefiningOp<triton::SplatOp>();
  return splat && isa<triton::PointerType>(splat.getSrc().getType());
}

} // namespace

LogicalResult LoadConverter::matchAndRewrite(triton::LoadOp op,
                                             PatternRewriter &rewriter) const {
  auto loc = op.getLoc();
  auto oldPtr = op.getPtr();
  auto oldMask = op.getMask();
  auto oldOther = op.getOther();

  if (auto catOp = oldPtr.getDefiningOp<triton::CatOp>()) {
    if (succeeded(rewriteCatPointerLoad(op, catOp, rewriter)))
      return success();
  }

  MemOpTransformer tf(MemOpTransformer::MemType::load, optimizeDynamicOffset);

  auto newPtr = tf.createNewPtr(oldPtr, loc, rewriter);
  auto newMask = tf.createNewMask(oldMask, loc, rewriter);
  auto newOther = tf.createNewOther(oldOther, loc, rewriter);

  if (!tf.ptrState.shouldLinearize) {
    // no need to rewrite
    return failure();
  }

  if (!newPtr) {
    LLVM_DEBUG({
      InFlightDiagnostic diag =
          emitWarning(loc) << "PtrAnalysis: failed to analyze load pointer.";
    });
    return failure();
  }

  if (!enableMaskFallbackConversion && oldMask && !newMask) {
    LLVM_DEBUG({
      InFlightDiagnostic diag = emitWarning(loc)
                                << "MaskAnalysis: failed to analyze load mask.";
    });
    return failure();
  }

  auto loadOp = rewriter.create<triton::LoadOp>(loc, newPtr, newMask, newOther,
                                                op.getCache(), op.getEvict(),
                                                op.getIsVolatile());

  // insert implicit ops
  auto broadCastResult =
      tf.materializeImplicitBroadcast(loadOp.getResult(), loc, rewriter);
  auto permuteResult =
      tf.materializeImplicitPermute(broadCastResult, loc, rewriter);
  auto reshapeResult =
      tf.materializeImplicitReshape(permuteResult, loc, rewriter);
  auto selectResult = tf.materializeImplicitSelect(reshapeResult, oldMask,
                                                   oldOther, loc, rewriter);

  rewriter.replaceOp(op, selectResult);
  return success();
}

LogicalResult StoreConverter::matchAndRewrite(triton::StoreOp op,
                                              PatternRewriter &rewriter) const {
  auto loc = op.getLoc();
  auto oldPtr = op.getPtr();
  auto oldMask = op.getMask();
  auto oldValue = op.getValue();

  if (auto catOp = oldPtr.getDefiningOp<triton::CatOp>()) {
    if (succeeded(rewriteCatPointerStore(op, catOp, rewriter)))
      return success();
  }

  // AddPtrSplatConverter can preserve a non-singleton zero-stride dimension
  // as a pointer broadcast. Rebuilding the scalar store here drops that
  // dimension from the pointer and mask while the store value stays full-rank.
  if (shouldDeferScalarPointerBroadcastStore(oldPtr, oldValue)) {
    return rewriter.notifyMatchFailure(
        op, "defer scalar pointer broadcast store to TritonToLinalg");
  }

  MemOpTransformer tf(MemOpTransformer::MemType::store, optimizeDynamicOffset);

  auto newPtr = tf.createNewPtr(oldPtr, loc, rewriter);
  auto newMask = tf.createNewMask(oldMask, loc, rewriter);

  if (!tf.ptrState.shouldLinearize) {
    // no need to rewrite
    return failure();
  }

  if (!newPtr) {
    LLVM_DEBUG({
      InFlightDiagnostic diag =
          emitWarning(loc) << "PtrAnalysis: failed to analyze store pointer.";
    });
    return failure();
  }

  if (!enableMaskFallbackConversion && oldMask && !newMask) {
    LLVM_DEBUG({
      InFlightDiagnostic diag =
          emitWarning(loc) << "MaskAnalysis: failed to analyze store mask.";
    });
    return failure();
  }

  // insert sync_block_lock
  auto lockVar = createSyncBlockLockVar(rewriter, loc);
  if (oldMask && !newMask) {
    rewriter.create<hivm::SyncBlockLockOp>(loc, lockVar);
  }

  auto selectResult =
      tf.materializeImplicitSelect(oldValue, oldMask, oldPtr, loc, rewriter);
  auto reshapeResult =
      tf.materializeImplicitReshape(selectResult, loc, rewriter);
  auto permuteResult =
      tf.materializeImplicitPermute(reshapeResult, loc, rewriter);

  auto storeOp = rewriter.create<triton::StoreOp>(
      loc, newPtr, permuteResult, newMask, op.getBoundaryCheck(), op.getCache(),
      op.getEvict());

  // insert sync_block_unlock
  if (oldMask && !newMask) {
    rewriter.create<hivm::SyncBlockUnlockOp>(loc, lockVar);
  }
  rewriter.eraseOp(op);
  return success();
}

Value MemOpTransformer::materializeImplicitBroadcast(
    Value srcTensor, const Location loc, PatternRewriter &rewriter) {
  SmallVector<int64_t> broadCastIndex;
  SmallVector<int64_t> broadCastShape;
  for (auto [i, info] : llvm::enumerate(ptrState.stateInfo)) {
    if (isZero(info.stride)) {
      broadCastIndex.emplace_back(i);
    }
    auto staticShape = getIntAttr(info.shape);
    if (!staticShape.has_value()) {
      LLVM_DEBUG({
        InFlightDiagnostic diag =
            emitWarning(loc)
            << "PtrAnalysis: dynamic shape is not supported in broadcast\n";
      });
      return srcTensor;
    }
    broadCastShape.emplace_back(staticShape.value());
  }

  if (broadCastIndex.empty())
    return srcTensor;

  // when load is a scalar, we need to use splat to broadcast
  auto srcType = srcTensor.getType();
  if (srcType.isIntOrFloat()) {
    auto broadCastType = RankedTensorType::get(broadCastShape, srcType);
    auto splatOp =
        rewriter.create<triton::SplatOp>(loc, broadCastType, srcTensor);
    return splatOp.getResult();
  }

  auto init = rewriter.create<tensor::EmptyOp>(
      loc, broadCastShape,
      cast<ShapedType>(srcTensor.getType()).getElementType());

  auto broadCastOp = rewriter.create<linalg::BroadcastOp>(loc, srcTensor, init,
                                                          broadCastIndex);

  return broadCastOp->getResult(0);
}

Value MemOpTransformer::materializeImplicitReshape(Value srcTensor,
                                                   const Location loc,
                                                   PatternRewriter &rewriter) {
  if (ptrState.sizes.size() == ptrState.stateInfo.size())
    return srcTensor;
  SmallVector<int64_t> targetShape;
  if (currentType == MemType::load) {
    for (auto size : ptrState.sizes) {
      auto staticShape = getIntAttr(size);
      if (!staticShape.has_value()) {
        LLVM_DEBUG({
          InFlightDiagnostic diag =
              emitWarning(loc)
              << "PtrAnalysis: dynamic shape is not supported in reshape\n";
        });
        return srcTensor;
      }
      targetShape.emplace_back(staticShape.value());
    }
  } else {
    for (auto info : ptrState.stateInfo) {
      auto staticShape = getIntAttr(info.shape);
      if (!staticShape.has_value()) {
        LLVM_DEBUG({
          InFlightDiagnostic diag =
              emitWarning(loc)
              << "PtrAnalysis: dynamic shape is not supported in reshape\n";
        });
        return srcTensor;
      }
      targetShape.emplace_back(staticShape.value());
    }
  }

  auto targetShapeAttr = DenseIntElementsAttr::get(
      RankedTensorType::get({static_cast<int64_t>(targetShape.size())},
                            rewriter.getI64Type()),
      targetShape);
  auto targetShapeType = RankedTensorType::get(
      targetShape, cast<ShapedType>(srcTensor.getType()).getElementType());
  auto targetShapeValue =
      rewriter.create<arith::ConstantOp>(loc, targetShapeAttr);
  auto reshapeOp = rewriter.create<tensor::ReshapeOp>(
      loc, targetShapeType, srcTensor, targetShapeValue);
  return reshapeOp.getResult();
}

Value MemOpTransformer::materializeImplicitSelect(Value srcTensor, Value mask,
                                                  Value other,
                                                  const Location loc,
                                                  PatternRewriter &rewriter) {
  if (!mask || maskState.newMask)
    return srcTensor;
  auto TensorType = cast<RankedTensorType>(srcTensor.getType());
  if (cast<ShapedType>(mask.getType()).getShape() != TensorType.getShape()) {
    LLVM_DEBUG({
      InFlightDiagnostic diag =
          emitWarning(loc) << "MaskAnalysis: mask shape is not same as Value";
    });
    return srcTensor;
  }

  if (currentType == MemType::store) {
    auto loadOp = rewriter.create<triton::LoadOp>(loc, other, nullptr, nullptr,
                                                  ArrayRef<int32_t>(), nullptr);
    other = loadOp.getResult();
  }

  if (!other) {
    auto elementType = TensorType.getElementType();
    auto emptyOp = rewriter.create<tensor::EmptyOp>(loc, TensorType.getShape(),
                                                    elementType);
    other = emptyOp.getResult();
  }
  auto selectOp = rewriter.create<arith::SelectOp>(loc, mask, srcTensor, other);
  return selectOp->getResult(0);
}

Value MemOpTransformer::materializeImplicitPermute(Value srcTensor,
                                                   const Location loc,
                                                   PatternRewriter &rewriter) {
  auto inTy = dyn_cast<RankedTensorType>(srcTensor.getType());
  if (!inTy || !ptrState.isPermuted)
    return srcTensor;

  auto inShape = inTy.getShape();
  SmallVector<int32_t> order(ptrState.permuteIds.size());
  for (size_t i = 0; i < ptrState.permuteIds.size(); ++i) {
    if (currentType == MemType::load) {
      order[ptrState.permuteIds[i]] = i;
    } else {
      order[i] = ptrState.permuteIds[i];
    }
  }
  SmallVector<int64_t> outShape(order.size());
  if (inShape.size() != outShape.size()) {
    LLVM_DEBUG({
      InFlightDiagnostic diag =
          emitWarning(loc) << "PtrAnalysis: incompatible shape for permute";
    });
    return srcTensor;
  }

  for (size_t i = 0; i < outShape.size(); ++i) {
    outShape[i] = inShape[order[i]];
  }

  auto outTy = RankedTensorType::get(outShape, inTy.getElementType());
  auto transOp = rewriter.create<triton::TransOp>(loc, outTy, srcTensor, order);
  return transOp.getResult();
}

Value MemOpTransformer::createNewPtr(Value oldPtr, const Location loc,
                                     PatternRewriter &rewriter) {
  TritonToStructured::PtrAnalysis ptrAnalysis(optimizeDynamicOffset);

  LLVM_DEBUG({
    llvm::dbgs() << "----------------------------------------------\n";
    llvm::dbgs() << "PtrAnalysis: analyzing load/store's ptr.\n";
  });

  if (ptrAnalysis.visitOperand(oldPtr, ptrState, loc, rewriter).failed()) {
    ptrState.shouldLinearize = false;
    LLVM_DEBUG({
      InFlightDiagnostic diag =
          emitWarning(loc) << "PtranAlysis: failed to analyze load/store ptr.";
    });
    return oldPtr;
  }

  // compute missing strides
  // if stateinfo.shape is 1 and sizes[dimIndex] is 1,
  // then the stride is the accumulated size of all dimensions on the right side
  // ie. for shape [1, 128], sizes [1, 128], originally stride is [0, 1],
  // after normalization, stride is [128, 1]
  OpFoldResult maxStride = rewriter.getIndexAttr(1);
  for (auto it = ptrState.stateInfo.rbegin(); it != ptrState.stateInfo.rend();
       ++it) {
    if (TritonToStructured::isOne(it->shape) && isZero(it->stride)) {
      it->stride = maxStride;
    }
    maxStride = maxOpFoldResult(maxStride, it->stride, loc, rewriter);
  }

  for (auto it = ptrState.stateInfo.rbegin(); it != ptrState.stateInfo.rend();
       ++it) {
    if (isZero(it->stride)) {
      ptrState.shouldLinearize = true;
    }
  }

  ptrState.generateOriginPermuteIds();

  return ptrState.createAddPtrOp(rewriter, loc);
}

Value MemOpTransformer::createNewMask(Value oldMask, const Location loc,
                                      PatternRewriter &rewriter) {
  if (!oldMask)
    return nullptr;

  LLVM_DEBUG({
    llvm::dbgs() << "----------------------------------------------\n";
    llvm::dbgs() << "MaskAnalysis: analyzing load/store mask.\n";
  });

  if (!oldMask || maskState.analysisMask(oldMask).failed()) {
    LLVM_DEBUG({
      llvm::dbgs() << "----------------------------------------------\n";
      llvm::dbgs() << "MaskAnalysis: no mask or failed to analyze mask.\n";
      llvm::dbgs() << "oldMask:" << oldMask << "\n";
      maskState.dump();
      llvm::dbgs() << "----------------------------------------------\n";
    });
    LLVM_DEBUG(
        {
          InFlightDiagnostic diag =
              emitWarning(loc)
              << "MaskAnalysis: failed to analyze load/store mask.";
        });
    return nullptr;
  }

  SmallVector<TritonToStructured::dimInfo> newMaskInfo;
  auto itPtr = ptrState.stateInfo.begin();
  auto itMask = maskState.stateInfo.begin();

  // match and create new mask info
  while (itPtr != ptrState.stateInfo.end() &&
         itMask != maskState.stateInfo.end()) {
    // ptr'shape must be multiple of mask'shape or vice versa
    if (!isMultiple(itMask->shape, itPtr->shape)) {
      LLVM_DEBUG({
        InFlightDiagnostic diag =
            emitWarning(loc)
            << "MaskAnalysis: incompatible shapes between ptr and mask.";
        llvm::dbgs() << "----------------------------------------------\n";
        ptrState.dump();
        llvm::dbgs() << "oldMask:" << oldMask << "\n";
        maskState.dump();
        llvm::dbgs() << "----------------------------------------------\n";
      });
      return nullptr;
    }

    auto newShape = minOpFoldResult(itMask->shape, itPtr->shape, loc, rewriter);
    if (isLess(newShape, itMask->shape) && !itMask->hasBroadCast) {
      LLVM_DEBUG({
        InFlightDiagnostic diag =
            emitWarning(loc)
            << "MaskAnalysis: the mask shape is incompatible with ptr shape.";
      });
      return nullptr;
    }

    TritonToStructured::dimInfo newInfo(itMask->offset, newShape,
                                        itMask->dimIndex, itMask->hasBroadCast,
                                        itMask->currentType, itMask->rhs);

    if (!isZero(itPtr->stride)) {
      newMaskInfo.emplace_back(newInfo);
    }

    ++itPtr;
    if (isEqual(itMask->shape, newShape)) {
      ++itMask;
    }
  }

  if (itPtr != ptrState.stateInfo.end() ||
      itMask != maskState.stateInfo.end()) {
    LLVM_DEBUG({
      llvm::dbgs() << "----------------------------------------------\n";
      llvm::dbgs() << "MaskAnalysis: failed to apply permute on mask.\n";
      ptrState.dump();
      llvm::dbgs() << "oldMask:" << oldMask << "\n";
      maskState.dump();
      llvm::dbgs() << "----------------------------------------------\n";
    });
    LLVM_DEBUG({
      InFlightDiagnostic diag = emitWarning(loc)
                                << "MaskAnalysis: incompatible number of "
                                   "dimensions between ptr and mask.";
    });
    return nullptr;
  }

  maskState.stateInfo = newMaskInfo;

  if (ptrState.isPermuted && !applyPermuteOnMask()) {
    LLVM_DEBUG({
      llvm::dbgs() << "----------------------------------------------\n";
      llvm::dbgs() << "MaskAnalysis: failed to apply permute on mask.\n";
      ptrState.dump();
      llvm::dbgs() << "oldMask:" << oldMask << "\n";
      maskState.dump();
      llvm::dbgs() << "----------------------------------------------\n";
      InFlightDiagnostic diag =
          emitWarning(loc) << "MaskAnalysis: failed to apply permute on mask.";
    });
    return nullptr;
  }

  LLVM_DEBUG({
    llvm::dbgs() << "After matching MaskState: \n";
    for (auto info : newMaskInfo) {
      info.dump();
    }
    llvm::dbgs() << "----------------------------------------------\n";
  });

  auto newMask = maskState.createNewMask(loc, rewriter);
  return newMask;
}

Value MemOpTransformer::createNewOther(Value oldOther, const Location loc,
                                       PatternRewriter &rewriter) {
  if (!oldOther || !maskState.newMask)
    return nullptr;

  auto ptrType = dyn_cast<triton::PointerType>(ptrState.source.getType());
  if (!ptrType) {
    LLVM_DEBUG(
        {
          InFlightDiagnostic diag =
              emitWarning(loc)
              << "PtrAnalysis: source of ptrState is not a pointer type.";
        });
    return nullptr;
  }
  Type elementType = ptrType.getPointeeType();

  SmallVector<int64_t> targetShape;
  for (auto info : maskState.stateInfo) {
    auto staticShape = getIntAttr(info.shape);
    if (!staticShape.has_value()) {
      LLVM_DEBUG({
        InFlightDiagnostic diag =
            emitWarning(loc)
            << "MaskAnalysis: dynamic shape is not supported in reshape\n";
      });
      return oldOther;
    }
    targetShape.emplace_back(staticShape.value());
  }
  auto targetShapeAttr = DenseIntElementsAttr::get(
      RankedTensorType::get({static_cast<int64_t>(targetShape.size())},
                            rewriter.getI64Type()),
      targetShape);
  auto targetShapeType = RankedTensorType::get(targetShape, elementType);
  auto targetShapeValue =
      rewriter.create<arith::ConstantOp>(loc, targetShapeAttr);

  auto reshapeOp = rewriter.create<tensor::ReshapeOp>(
      loc, targetShapeType, oldOther, targetShapeValue);

  return reshapeOp.getResult();
}

bool MemOpTransformer::applyPermuteOnMask() {
  if (!ptrState.isPermuted || maskState.isEmpty()) {
    return true;
  }
  if (ptrState.permuteIds.size() != maskState.stateInfo.size()) {
    return false;
  }
  SmallVector<TritonToStructured::dimInfo> newMaskInfo;
  for (auto id : ptrState.permuteIds) {
    newMaskInfo.push_back(maskState.stateInfo[id]);
  }
  maskState.stateInfo = newMaskInfo;
  return true;
}

hivm::CreateSyncBlockLockOp createSyncBlockLockVar(OpBuilder &builder,
                                                   Location loc) {
  SmallVector<int64_t> shape = {1};
  auto elementType = builder.getI64Type();
  Type memrefType = MemRefType::get(shape, elementType);

  auto createSyncBlockLockOp =
      builder.create<hivm::CreateSyncBlockLockOp>(loc, memrefType, Value());
  return createSyncBlockLockOp;
}
} // namespace MemOpConverter
