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

#include "TritonToLinalg/ImplicitPermute.h"
#include "TritonToLinalg/TritonToLinalgPass.h"
#include "TritonToStructured/CannonicalizerConverter.h"
#include "TritonToStructured/MaskAnalysis.h"
#include "TritonToStructured/PtrAnalysis.h"

#include "Utils/InterleaveOptimization.h"
#include "Utils/Utils.h"
#include "bishengir/Dialect/Annotation/IR/Annotation.h"
#include "bishengir/Dialect/HIVM/Utils/Utils.h"

#define DEBUG_TYPE "triton-to-linalg-implicit-permute"

namespace ImplicitPermute {
using namespace mlir;
using namespace triton;
using namespace TritonToStructured;

LogicalResult LoadConverter::matchAndRewrite(triton::LoadOp op,
                                             PatternRewriter &rewriter) const {
  // no need to analyze and rewrite
  if (compileOn91095Flag && !existDotFlag) {
    LLVM_DEBUG({
      llvm::dbgs() << "----------------------------------------------\n";
      llvm::dbgs() << "compileOn91095Flag :" << compileOn91095Flag << "\n";
      llvm::dbgs() << "!existDotFlag :" << !existDotFlag << "\n";
      llvm::dbgs() << "no need to analyze and rewrite Load" << "\n";
      llvm::dbgs() << "----------------------------------------------\n";
    });
    return success();
  }

  auto loc = op.getLoc();
  auto oldPtr = op.getPtr();
  auto oldMask = op.getMask();
  auto oldOther = op.getOther();

  MemOpTransformer tf(MemOpTransformer::MemType::load);

  Value newPtr = nullptr;
  if (oldPtr.getDefiningOp<triton::MakeTensorPtrOp>()) {
    newPtr = tf.createNewTensorPtr(oldPtr, loc, rewriter);
  } else if (oldPtr.getDefiningOp<triton::AdvanceOp>()) {
    newPtr = tf.createNewAdvancePtr(oldPtr, loc, rewriter);
  } else if (oldPtr.getDefiningOp<triton::AddPtrOp>()) {
    newPtr = tf.createNewAddPtr(oldPtr, loc, rewriter);
  } else {
    InFlightDiagnostic diag = emitWarning(loc)
                              << "PtrAnalysis: only MakeTensorPtrOp, "
                                 "AdvanceOp, and AddPtrOp are supported.";
    return success();
  }
  if (!tf.ptrState.isPermuted) {
    // no need to rewrite
    return success();
  }

  auto newMask = tf.createNewMask(oldMask, loc, rewriter);
  auto newOther = tf.createNewOther(oldOther, loc, rewriter);

  if (!newPtr) {
    InFlightDiagnostic diag = emitWarning(loc)
                              << "PtrAnalysis: failed to analyze load pointer.";
    return failure();
  }

  if (oldMask && !newMask) {
    InFlightDiagnostic diag = emitWarning(loc)
                              << "MaskAnalysis: failed to analyze load mask.";
    return failure();
  }

  auto newBoundaryCheck = tf.getBoundaryCheck(op.getBoundaryCheck());

  auto loadOp = rewriter.create<triton::LoadOp>(
      loc, newPtr, newMask, newOther, newBoundaryCheck, op.getPadding(),
      op.getCache(), op.getEvict(), op.getIsVolatile());
  loadOp->setAttr(ImplicitPermuteHandledTAG,
                  UnitAttr::get(rewriter.getContext()));

  auto permuteResult =
      tf.materializeImplicitPermute(loadOp.getResult(), loc, rewriter);

  rewriter.replaceOp(op, permuteResult);
  return success();
}

LogicalResult StoreConverter::matchAndRewrite(triton::StoreOp op,
                                              PatternRewriter &rewriter) const {
  auto loc = op.getLoc();
  auto oldPtr = op.getPtr();
  auto oldMask = op.getMask();
  auto oldValue = op.getValue();

  MemOpTransformer tf(MemOpTransformer::MemType::store);
  Value newPtr = nullptr;
  if (oldPtr.getDefiningOp<triton::MakeTensorPtrOp>()) {
    newPtr = tf.createNewTensorPtr(oldPtr, loc, rewriter);
  } else if (oldPtr.getDefiningOp<triton::AdvanceOp>()) {
    newPtr = tf.createNewAdvancePtr(oldPtr, loc, rewriter);
  } else if (oldPtr.getDefiningOp<triton::AddPtrOp>()) {
    newPtr = tf.createNewAddPtr(oldPtr, loc, rewriter);
  } else {
    InFlightDiagnostic diag = emitWarning(loc)
                              << "PtrAnalysis: only MakeTensorPtrOp, "
                                 "AdvanceOp, and AddPtrOp are supported.";
    return success();
  }
  if (!tf.ptrState.isPermuted) {
    // no need to rewrite
    return success();
  }
  auto newMask = tf.createNewMask(oldMask, loc, rewriter);

  if (!newPtr) {
    InFlightDiagnostic diag =
        emitWarning(loc) << "PtrAnalysis: failed to analyze store pointer.";
    return failure();
  }

  if (oldMask && !newMask) {
    InFlightDiagnostic diag = emitWarning(loc)
                              << "MaskAnalysis: failed to analyze store mask.";
    return failure();
  }

  auto permuteResult = tf.materializeImplicitPermute(oldValue, loc, rewriter);

  auto newBoundaryCheck = tf.getBoundaryCheck(op.getBoundaryCheck());

  auto storeOp = rewriter.create<triton::StoreOp>(loc, newPtr, permuteResult,
                                                  newMask, newBoundaryCheck,
                                                  op.getCache(), op.getEvict());
  storeOp->setAttr(ImplicitPermuteHandledTAG,
                   UnitAttr::get(rewriter.getContext()));

  rewriter.eraseOp(op);
  return success();
}

LogicalResult
AtomicRMWConverter::matchAndRewrite(triton::AtomicRMWOp op,
                                    PatternRewriter &rewriter) const {
  auto loc = op.getLoc();
  auto oldPtr = op.getPtr();
  auto oldMask = op.getMask();
  auto oldVal = op.getVal();

  MemOpTransformer tf(MemOpTransformer::MemType::store);

  Value newPtr = nullptr;
  if (oldPtr.getDefiningOp<triton::AddPtrOp>()) {
    newPtr = tf.createNewAddPtr(oldPtr, loc, rewriter);
  } else {
    emitWarning(loc) << "PtrAnalysis: AtomicRMW only support AddPtrOp.";
    return success();
  }

  Value newMask = tf.createNewMask(oldMask, loc, rewriter);

  if (!tf.ptrState.isPermuted) {
    return success();
  }
  if (!newPtr) {
    emitWarning(loc) << "PtrAnalysis: failed to analyze atomic_rmw pointer.";
    return failure();
  }
  if (oldMask && !newMask) {
    emitWarning(loc) << "MaskAnalysis: failed to analyze atomic_rmw mask.";
    return failure();
  }

  Value newVal = tf.materializeImplicitPermute(oldVal, loc, rewriter);

  Type newAtomicResTy = newVal.getType();

  auto newAtomic = rewriter.create<triton::AtomicRMWOp>(
      loc, newAtomicResTy, op.getAtomicRmwOp(), newPtr, newVal, newMask,
      op.getSem(), op.getScope());
  newAtomic->setAttr(ImplicitPermuteHandledTAG,
                     UnitAttr::get(rewriter.getContext()));

  // The returned old value should be in OLD layout for users => permute back
  // (load-side).
  MemOpTransformer tfLoad(MemOpTransformer::MemType::load);
  tfLoad.ptrState = tf.ptrState;
  Value permutedRes =
      tfLoad.materializeImplicitPermute(newAtomic.getResult(), loc, rewriter);

  rewriter.replaceOp(op, permutedRes);
  return success();
}

LogicalResult
AtomicCASConverter::matchAndRewrite(triton::AtomicCASOp op,
                                    PatternRewriter &rewriter) const {
  auto loc = op.getLoc();
  auto oldPtr = op.getPtr();
  auto oldCmp = op.getCmp();
  auto oldVal = op.getVal();

  MemOpTransformer tf(MemOpTransformer::MemType::store);

  Value newPtr = nullptr;
  if (oldPtr.getDefiningOp<triton::AddPtrOp>()) {
    newPtr = tf.createNewAddPtr(oldPtr, loc, rewriter);
  } else {
    emitWarning(loc) << "PtrAnalysis: AtomicRMW only support AddPtrOp.";
    return success();
  }

  if (!tf.ptrState.isPermuted) {
    return success();
  }
  if (!newPtr) {
    emitWarning(loc) << "PtrAnalysis: failed to analyze atomic_cas pointer.";
    return failure();
  }

  Value newCmp = tf.materializeImplicitPermute(oldCmp, loc, rewriter);
  Value newVal = tf.materializeImplicitPermute(oldVal, loc, rewriter);

  // CAS result (old value) must have same shape as cmp/val operands.
  Type newAtomicResTy = newVal.getType();

  auto newAtomic = rewriter.create<triton::AtomicCASOp>(
      loc, newAtomicResTy, newPtr, newCmp, newVal, op.getSem(), op.getScope());
  newAtomic->setAttr(ImplicitPermuteHandledTAG,
                     UnitAttr::get(rewriter.getContext()));

  MemOpTransformer tfLoad(MemOpTransformer::MemType::load);
  tfLoad.ptrState = tf.ptrState;
  Value permutedRes =
      tfLoad.materializeImplicitPermute(newAtomic.getResult(), loc, rewriter);

  rewriter.replaceOp(op, permutedRes);
  return success();
}

Value MemOpTransformer::materializeImplicitPermute(Value srcTensor,
                                                   const Location loc,
                                                   PatternRewriter &rewriter) {
  auto inTy = dyn_cast<RankedTensorType>(srcTensor.getType());
  if (!inTy || !ptrState.isPermuted)
    return srcTensor;

  auto inShape = inTy.getShape();
  auto orderSize = ptrState.sizes.size();
  SmallVector<int32_t> permuteOrder(orderSize);
  for (size_t i = 0; i < orderSize; ++i) {
    if (currentType == MemType::load) {
      if (ptrState.isBlockPtr()) {
        permuteOrder[i] = orderSize - 1 - ptrState.order[i];
      } else {
        permuteOrder[ptrState.permuteIds[i]] = i;
      }
    } else {
      if (ptrState.isBlockPtr()) {
        permuteOrder[i] = ptrState.order[orderSize - 1 - i];
      } else {
        permuteOrder[i] = ptrState.permuteIds[i];
      }
    }
  }
  SmallVector<int64_t> outShape(permuteOrder.size());
  if (inShape.size() != outShape.size()) {
    InFlightDiagnostic diag = emitWarning(loc)
                              << "PtrAnalysis: incompatible shape for permute";
    return srcTensor;
  }

  for (size_t i = 0; i < outShape.size(); ++i) {
    outShape[i] = inShape[permuteOrder[i]];
  }

  auto outTy = RankedTensorType::get(outShape, inTy.getElementType());
  auto transOp =
      rewriter.create<triton::TransOp>(loc, outTy, srcTensor, permuteOrder);
  return transOp.getResult();
}

Value MemOpTransformer::createNewAddPtr(Value oldPtr, const Location loc,
                                        PatternRewriter &rewriter) {
  TritonToStructured::PtrAnalysis ptrAnalysis;

  LLVM_DEBUG({
    llvm::dbgs() << "----------------------------------------------\n";
    llvm::dbgs() << "PtrAnalysis: analyzing load/store's ptr.\n";
  });

  if (ptrAnalysis.visitOperand(oldPtr, ptrState, loc, rewriter).failed()) {
    ptrState.shouldLinearize = false;
    InFlightDiagnostic diag =
        emitWarning(loc) << "PtranAlysis: failed to analyze load/store ptr.";
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

  ptrState.analyzePermute();
  return ptrState.createAddPtrOp(rewriter, loc);
}

Value MemOpTransformer::createNewTensorPtr(Value oldPtr, const Location loc,
                                           PatternRewriter &rewriter) {
  TritonToStructured::PtrAnalysis ptrAnalysis;
  auto makeTPtrOp = oldPtr.getDefiningOp<triton::MakeTensorPtrOp>();
  if (!makeTPtrOp) {
    InFlightDiagnostic diag = emitWarning(loc)
                              << "PtrAnalysis: load pointer must originate "
                                 "from 'make_tensor_ptr' operation";
    return oldPtr;
  }
  if (ptrAnalysis.visitOperandMakeTensorPtr(makeTPtrOp, ptrState, loc, rewriter)
          .failed()) {
    ptrState.isPermuted = false;
    return oldPtr;
  }
  ptrState.analyzePermute();
  LLVM_DEBUG({
    llvm::dbgs() << "----------------------------------------------\n";
    llvm::dbgs() << "After ptrState.analyzePermute:\n";
    llvm::dbgs() << "compileOn91095Flag: " << compileOn91095Flag << "\n";
    ptrState.dump();
    llvm::dbgs() << "----------------------------------------------\n";
  });
  return ptrState.createMakeTensorPtrOp(rewriter, loc);
}

Value MemOpTransformer::createNewAdvancePtr(Value oldPtr, const Location loc,
                                            PatternRewriter &rewriter) {
  TritonToStructured::PtrAnalysis ptrAnalysis;
  auto advOp = oldPtr.getDefiningOp<triton::AdvanceOp>();
  if (!advOp) {
    emitWarning(loc)
        << "PtrAnalysis: pointer must originate from 'advance' operation";
    return oldPtr;
  }

  Value basePtr = advOp.getPtr();
  if (!basePtr || !basePtr.getDefiningOp<triton::MakeTensorPtrOp>()) {
    emitWarning(loc) << "PtrAnalysis: advance base ptr must originate from "
                        "'make_tensor_ptr' operation";
    return oldPtr;
  }
  auto newBasePtr = createNewTensorPtr(basePtr, loc, rewriter);
  if (!newBasePtr)
    return oldPtr;

  if (!ptrState.isPermuted)
    return oldPtr;
  // 2) Rewrite advance offsets according to the new make_tensor_ptr layout.
  auto oldOffsets = advOp.getOffsets();
  SmallVector<Value> newOffsets;
  size_t rank = ptrState.order.size();
  // iterate reversed safely: i = rank-1, ..., 0
  for (size_t i = rank; i-- > 0;) {
    newOffsets.push_back(oldOffsets[ptrState.order[i]]);
  }
  return rewriter.create<triton::AdvanceOp>(loc, newBasePtr.getType(),
                                            newBasePtr, newOffsets);
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
    InFlightDiagnostic diag =
        emitWarning(loc) << "MaskAnalysis: failed to analyze load/store mask.";
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
      InFlightDiagnostic diag =
          emitWarning(loc)
          << "MaskAnalysis: incompatible shapes between ptr and mask.";
      LLVM_DEBUG({
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
      InFlightDiagnostic diag =
          emitWarning(loc)
          << "MaskAnalysis: the mask shape is incompatible with ptr shape.";
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
    InFlightDiagnostic diag = emitWarning(loc)
                              << "MaskAnalysis: incompatible number of "
                                 "dimensions between ptr and mask.";
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
    });
    InFlightDiagnostic diag =
        emitWarning(loc) << "MaskAnalysis: failed to apply permute on mask.";
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
    InFlightDiagnostic diag =
        emitWarning(loc)
        << "PtrAnalysis: source of ptrState is not a pointer type.";
    return nullptr;
  }
  Type elementType = ptrType.getPointeeType();

  SmallVector<int64_t> targetShape;
  for (auto info : maskState.stateInfo) {
    auto staticShape = getIntAttr(info.shape);
    if (!staticShape.has_value()) {
      InFlightDiagnostic diag =
          emitWarning(loc)
          << "MaskAnalysis: dynamic shape is not supported in reshape\n";
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

// Remap boundary_check for block ptr implicit permute.
// Formula
// - newAxis = rank - 1 - position(oldAxis in ptrState.order)
// Rules implemented
// - ptrState.order records original axes in memory-priority order.
// - createMakeTensorPtrOp rebuilds the new block ptr from
// reverse(ptrState.order),
//   so the rebuilt ptr is canonicalized to descending order [rank-1, ..., 0].
// - So each old boundary_check axis must be translated into the axis index
//   of the rewritten block ptr before creating the new load/store.
// Examples
// - order=[1,0], boundary_check=[0] => [0]
// - order=[0,1], boundary_check=[0,1] => [1,0]
// - order=[2,0,1], boundary_check=[0,2] => [1,2]
SmallVector<int32_t>
MemOpTransformer::getBoundaryCheck(ArrayRef<int32_t> oldBoundaryCheck) const {
  SmallVector<int32_t> newBoundaryCheck(oldBoundaryCheck.begin(),
                                        oldBoundaryCheck.end());
  if (!ptrState.isPermuted || !ptrState.isBlockPtr() ||
      newBoundaryCheck.empty()) {
    return newBoundaryCheck;
  }

  int32_t rank = static_cast<int32_t>(ptrState.order.size());
  for (auto &boundaryAxis : newBoundaryCheck) {
    auto pos = llvm::find(ptrState.order, static_cast<size_t>(boundaryAxis));
    if (pos == ptrState.order.end()) {
      continue;
    }
    boundaryAxis =
        rank - 1 -
        static_cast<int32_t>(std::distance(ptrState.order.begin(), pos));
  }
  return newBoundaryCheck;
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

} // namespace ImplicitPermute
