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

#include "TritonToStructured/CannonicalizerConverter.h"

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
#include "mlir/Support/LLVM.h"
#include "triton/Dialect/Triton/IR/Dialect.h"

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

#include "TritonToStructured/PtrAnalysis.h"
#include "TritonToStructured/TritonToStructuredPass.h"
#include "Utils/InterleaveOptimization.h"
#include "Utils/Utils.h"
#include "bishengir/Dialect/Annotation/IR/Annotation.h"

#define DEBUG_TYPE "triton-cannonicalizer-converter"

namespace CannonicalizerConverter {
using namespace mlir;
using namespace triton;
constexpr int INT_TYPE_BIT_WIDTH = 32;

// Match and rewrite pattern for optimizing cmp.ne (select(cond, 1, 0), 0) ->
// cond This pattern transforms:
//   %select = arith.select %cond, %true_val, %false_val
//   %cmp = arith.cmpi ne, %select, %zero
// Where:
//   - %true_val is a constant splat tensor of 1s
//   - %false_val is a constant splat tensor of 0s
//   - %zero is a constant splat tensor of 0s
// Into:
//   %cond  (directly replace the cmp with the select condition)
//
// This optimization is valid because:
//   select(cond, 1, 0) != 0
//   is equivalent to: cond != 0
//   Since the result of select is either 1 or 0, the only way it's not equal to
//   0 is when it's 1, which happens exactly when cond is true.
//
// Example:
//   Input IR:
//     %39 = arith.cmpi slt, %15, %cst_14 : tensor<128xi32>
//     %40 = arith.select %39, %cst_13, %cst_12 : tensor<128xi1>,
//     tensor<128xi32> %41 = arith.cmpi ne, %40, %cst_12 : tensor<128xi32>
//   Where cst_13 is constant dense<1> and cst_12 is constant dense<0>
//   Output IR:
//     %39 = arith.cmpi slt, %15, %cst_14 : tensor<128xi32>
LogicalResult CmpConverter::matchAndRewrite(arith::CmpIOp cmpOp,
                                            PatternRewriter &rewriter) const {
  // Only handle "not equal" comparison
  auto cmpType = cmpOp.getPredicate();
  if (cmpType != arith::CmpIPredicate::ne) {
    return failure();
  }

  Value rhs = cmpOp.getRhs();
  Value lhs = cmpOp.getLhs();

  // 1. Check if RHS is a constant zero
  APInt rhsValue;
  if (!matchPattern(rhs, m_ConstantInt(&rhsValue))) {
    return failure(); // RHS is not a constant
  }

  if (!rhsValue.isZero()) {
    return failure(); // RHS is not zero
  }

  // 2. Check if LHS is defined by a select operation
  auto selectOp = lhs.getDefiningOp<arith::SelectOp>();
  if (!selectOp) {
    return failure();
  }

  // 3. Check if select's true and false values are constants
  DenseElementsAttr trueAttr;
  DenseElementsAttr falseAttr;
  if (!matchPattern(selectOp.getTrueValue(), m_Constant(&trueAttr)) ||
      !matchPattern(selectOp.getFalseValue(), m_Constant(&falseAttr))) {
    return failure(); // Either true or false value is not constant
  }

  // 4. Check if true value is all 1s and false value is all 0s
  if (!trueAttr.isSplat() || !trueAttr.getSplatValue<APInt>().isOne() ||
      !falseAttr.isSplat() || !falseAttr.getSplatValue<APInt>().isZero()) {
    return failure();
  }

  // 5. Optimization matched, replace cmp with select's condition
  rewriter.replaceOp(cmpOp, selectOp.getCondition());
  return success();
}

// Detect when both operands of the cmpOp are triton::SplatOp. If so,
// replace the original comparison by comparing the underlying scalar values
// and then splatting (broadcasting) the scalar comparison result back to the
// original tensor shape.
// Example:
//   Input IR:
//     %splat_lhs = tt.splat %val1 : tensor<128xi32>
//     %splat_rhs = tt.splat %val2 : tensor<128xi32>
//     %cmp = arith.cmpi slt, %splat_lhs, %splat_rhs : tensor<128xi32>
//   Output IR:
//     %cmp_scalar = arith.cmpi slt, %val1, %val2
//     %splat_cmp = tt.splat %cmp_scalar : tensor<128xi1>
LogicalResult
SplatCmpConverter::matchAndRewrite(arith::CmpIOp cmpOp,
                                   PatternRewriter &rewriter) const {
  auto lhs = cmpOp.getLhs();
  auto rhs = cmpOp.getRhs();
  auto lhsSplatOp = lhs.getDefiningOp<triton::SplatOp>();
  auto rhsSplatOp = rhs.getDefiningOp<triton::SplatOp>();
  if (!lhsSplatOp || !rhsSplatOp) {
    return failure();
  }
  auto lhsSrc = lhsSplatOp.getSrc();
  auto rhsSrc = rhsSplatOp.getSrc();
  auto newCmpOp = rewriter.create<arith::CmpIOp>(
      cmpOp.getLoc(), cmpOp.getPredicate(), lhsSrc, rhsSrc);
  auto cmpType = dyn_cast<RankedTensorType>(cmpOp.getType());
  if (!cmpType) {
    return failure();
  }
  auto splatType =
      RankedTensorType::get(cmpType.getShape(), newCmpOp.getType());
  auto splatOp = rewriter.create<triton::SplatOp>(cmpOp.getLoc(), splatType,
                                                  newCmpOp.getResult());
  rewriter.replaceOp(cmpOp, splatOp.getResult());
  return success();
}

// Convert AddPtr when ptr is produced by tt.splat and offset is produced by
// tt.broadcast of a smaller-shaped value. Transformation:
//   %ptr_splat = tt.splat %ptr_src : tensor<Large x !tt.ptr<...>>
//   %offset_b = tt.broadcast %offset_src : tensor<Large x i32>
//   %res = tt.addptr %ptr_splat, %offset_b
// =>
//   %ptr_small_splat = tt.splat %ptr_src : tensor<Small x !tt.ptr<...>>
//   %add_small = tt.addptr %ptr_small_splat, %offset_src
//   %res = tt.broadcast %add_small : tensor<Large x !tt.ptr<...>>
LogicalResult
AddPtrSplatConverter::matchAndRewrite(triton::AddPtrOp addPtrOp,
                                      PatternRewriter &rewriter) const {
  // Match ptr produced by splat
  auto ptr = addPtrOp.getPtr();
  auto ptrSplatOp = ptr.getDefiningOp<triton::SplatOp>();
  if (!ptrSplatOp)
    return failure();

  // Match offset produced by broadcast
  auto offset = addPtrOp.getOffset();
  auto offsetBroadcastOp = offset.getDefiningOp<triton::BroadcastOp>();
  if (!offsetBroadcastOp)
    return failure();

  // Extract sources
  auto ptrSrc = ptrSplatOp.getSrc();
  auto offsetSrc = offsetBroadcastOp.getSrc();

  // Types must be ranked to reason about shapes
  auto resultType = dyn_cast<RankedTensorType>(addPtrOp.getResult().getType());
  auto offsetSrcType = dyn_cast<RankedTensorType>(offsetSrc.getType());
  if (!resultType || !offsetSrcType)
    return failure();

  // If offset source already has same shape as result, nothing to do
  if (resultType.getShape() == offsetSrcType.getShape())
    return failure();

  auto loc = addPtrOp.getLoc();

  // Build pointer tensor type corresponding to offset's (smaller) shape
  // ptrSrc should be a pointer type (element type)
  auto ptrElementType = dyn_cast<triton::PointerType>(ptrSrc.getType());
  auto smallPtrTensorType =
      RankedTensorType::get(offsetSrcType.getShape(), ptrElementType);

  // Create splat of ptrSrc to the smaller shape
  auto smallPtrSplat =
      rewriter.create<triton::SplatOp>(loc, smallPtrTensorType, ptrSrc);

  // Create addptr on the smaller shape
  auto smallAdd = rewriter.create<triton::AddPtrOp>(
      loc, smallPtrTensorType, smallPtrSplat.getResult(), offsetSrc);

  // Broadcast the small add result back to the original (larger) shape
  auto broadcasted = rewriter.create<triton::BroadcastOp>(loc, resultType,
                                                          smallAdd.getResult());

  rewriter.replaceOp(addPtrOp, broadcasted.getResult());
  return success();
}

// Move load before broadcast when possible:
// If load.ptr is a triton::BroadcastOp and
//  - load.mask is null; or
//  - load.mask is a triton::BroadcastOp,
// then rewrite the load in one of two ways depending on whether the ptr src
// and mask src shapes match:
//
// Case 1 (shapes match, original fast path):
//   %ptr_b  = tt.broadcast %ptr_src
//   %mask_b = tt.broadcast %mask_src        (src shapes identical)
//   %v = tt.load %ptr_b, %mask_b, %other_splat
// becomes:
//   %v_small = tt.load %ptr_src, %mask_src, %other_small
//   %v = tt.broadcast %v_small
//
// Case 2 (shapes differ, e.g. ptr and mask broadcast on different axes):
//   %ptr_b  = tt.broadcast %ptr_src         // e.g. [1,64] -> [256,64]
//   %mask_b = tt.broadcast %mask_src        // e.g. [256,1] -> [256,64]
//   %v = tt.load %ptr_b, %mask_b, %other
// becomes:
//   %v_small = tt.load %ptr_src             (no mask, no other — safe)
//   %v_bc    = tt.broadcast %v_small
//   %v       = arith.select %mask_b, %v_bc, %other
// The full mask is applied via select after broadcast, which downstream
// select analysis lowers to extract_slice / insert_slice.
LogicalResult
LoadBroadcastConverter::matchAndRewrite(triton::LoadOp loadOp,
                                        PatternRewriter &rewriter) const {
  // Match when ptr is defined by BroadcastOp
  Value ptr = loadOp.getPtr();
  auto ptrBroadcast = ptr.getDefiningOp<triton::BroadcastOp>();
  if (!ptrBroadcast)
    return failure();

  auto ptrSrcType = dyn_cast<RankedTensorType>(ptrBroadcast.getSrc().getType());
  if (!ptrSrcType)
    return failure();

  // mask can be null
  Value mask = loadOp.getMask();
  triton::BroadcastOp maskBroadcast = nullptr;
  Value maskSrc = nullptr;
  RankedTensorType maskSrcType = nullptr;
  if (mask) {
    maskBroadcast = mask.getDefiningOp<triton::BroadcastOp>();
    if (!maskBroadcast)
      return failure();
    maskSrc = maskBroadcast.getSrc();
    maskSrcType = dyn_cast<RankedTensorType>(maskSrc.getType());
    if (!maskSrcType)
      return failure();
  }

  Location loc = loadOp.getLoc();
  Value smallPtr = ptrBroadcast.getSrc();

  // Preserve other operands of load (like 'other', cache, evict, isVolatile)
  Value other = loadOp.getOther();
  auto cache = loadOp.getCache();
  auto evict = loadOp.getEvict();
  auto isVolatile = loadOp.getIsVolatile();
  auto resultType = dyn_cast<RankedTensorType>(loadOp.getResult().getType());
  if (!resultType)
    return failure();

  bool shapesMatch = !mask || (ptrSrcType.getShape() == maskSrcType.getShape());

  // no mask, or ptr/mask src shapes match
  if (shapesMatch) {
    Value newOther = other;
    if (other) {
      // Build a small-shaped 'other' constant matching ptr src shape.
      // Only splat dense constants are supported here (same as before).
      Attribute otherAttr;
      if (!matchPattern(other, m_Constant(&otherAttr)))
        return failure();
      auto denseOther = dyn_cast<DenseElementsAttr>(otherAttr);
      if (!denseOther)
        return failure();

      auto elemType = denseOther.getType().getElementType();
      if (!elemType)
        return failure();
      auto smallType = RankedTensorType::get(ptrSrcType.getShape(), elemType);

      DenseElementsAttr newDense;
      if (denseOther.isSplat()) {
        newDense = DenseElementsAttr::get(
            smallType, denseOther.getSplatValue<Attribute>());
      } else {
        return failure();
      }
      newOther = rewriter.create<arith::ConstantOp>(loc, newDense).getResult();
    }

    auto newLoad = rewriter.create<triton::LoadOp>(
        loc, smallPtr, maskSrc, newOther, cache, evict, isVolatile);
    auto broadcasted = rewriter.create<triton::BroadcastOp>(
        loc, resultType, newLoad.getResult());
    rewriter.replaceOp(loadOp, broadcasted.getResult());
    return success();
  }

  auto newLoad = rewriter.create<triton::LoadOp>(
      loc, smallPtr, /*mask=*/nullptr, /*other=*/nullptr, cache, evict,
      isVolatile);
  auto broadcasted = rewriter.create<triton::BroadcastOp>(loc, resultType,
                                                          newLoad.getResult());
  Value result = broadcasted.getResult();

  if (!other) {
    // default for masked-off elements is 0
    auto zeroAttr = DenseElementsAttr::get(
        resultType, rewriter.getZeroAttr(resultType.getElementType()));
    other = rewriter.create<arith::ConstantOp>(loc, zeroAttr).getResult();
  }
  auto selectOp = rewriter.create<arith::SelectOp>(loc, mask, result, other);
  rewriter.replaceOp(loadOp, selectOp.getResult());
  return success();
}

// Move store before broadcast when possible:
// If store.ptr is defined by a triton::BroadcastOp, the broadcast axes refer
// to the same memory locations (every pointer along a broadcast axis points
// to the same address), so storing once at the first index of each broadcast
// axis is sufficient. Rewrite
//   %ptr   = tt.broadcast %ptr_src            // e.g. [32,1] -> [32,32]
//   %mask  = tt.broadcast %mask_src?          // optional, same expansion
//   tt.store %ptr, %val, %mask?
// with
//   %val_small  = tensor.extract_slice %val[..0..]  // first index per
//                                                    // broadcast axis
//   %mask_small = %mask_src | tensor.extract_slice %mask
//   tt.store %ptr_src, %val_small, %mask_small?
LogicalResult
StoreBroadcastConverter::matchAndRewrite(triton::StoreOp storeOp,
                                         PatternRewriter &rewriter) const {
  // Match when ptr is defined by BroadcastOp
  Value ptr = storeOp.getPtr();
  auto ptrBroadcast = ptr.getDefiningOp<triton::BroadcastOp>();
  if (!ptrBroadcast)
    return failure();

  auto ptrSrcType = dyn_cast<RankedTensorType>(ptrBroadcast.getSrc().getType());
  auto resultType = dyn_cast<RankedTensorType>(ptr.getType());
  if (!ptrSrcType || !resultType)
    return failure();

  // Find the axes expanded by broadcast (src size 1 -> full size)
  auto srcShape = ptrSrcType.getShape();
  auto resultShape = resultType.getShape();
  SmallVector<int64_t> broadcastAxes;
  for (size_t i = 0; i < srcShape.size(); ++i) {
    if (srcShape[i] == 1 && resultShape[i] != 1)
      broadcastAxes.push_back(i);
  }
  if (broadcastAxes.empty())
    return failure();

  Location loc = storeOp.getLoc();
  SmallVector<OpFoldResult> offsets, sizes, strides;
  for (size_t i = 0; i < resultShape.size(); ++i) {
    offsets.push_back(rewriter.getIndexAttr(0));
    strides.push_back(rewriter.getIndexAttr(1));
    sizes.push_back(rewriter.getIndexAttr(resultShape[i]));
  }
  for (auto axis : broadcastAxes)
    sizes[axis] = rewriter.getIndexAttr(1);

  // Slice value along the broadcast axes (first index) if it has the full
  // broadcast shape; otherwise it must already match the small ptr shape.
  Value newValue = storeOp.getValue();
  if (auto valueType = dyn_cast<RankedTensorType>(newValue.getType())) {
    if (valueType.getShape() == resultShape) {
      newValue = rewriter.create<tensor::ExtractSliceOp>(loc, newValue, offsets,
                                                         sizes, strides);
    } else if (valueType.getShape() != srcShape) {
      return failure();
    }
  }

  // Slice mask the same way; reuse mask's broadcast source when its shape
  // matches the small ptr shape, avoiding a redundant extract_slice.
  Value newMask;
  if (Value mask = storeOp.getMask()) {
    auto maskType = dyn_cast<RankedTensorType>(mask.getType());
    if (!maskType)
      return failure();
    if (maskType.getShape() == resultShape) {
      Value maskSrc = nullptr;
      if (auto maskBroadcast = mask.getDefiningOp<triton::BroadcastOp>()) {
        auto maskSrcType =
            dyn_cast<RankedTensorType>(maskBroadcast.getSrc().getType());
        if (maskSrcType && maskSrcType.getShape() == srcShape)
          maskSrc = maskBroadcast.getSrc();
      }
      newMask =
          maskSrc ? maskSrc
                  : static_cast<Value>(rewriter.create<tensor::ExtractSliceOp>(
                        loc, mask, offsets, sizes, strides));
    } else if (maskType.getShape() != srcShape) {
      return failure();
    } else {
      newMask = mask;
    }
  }

  // Store the reduced slice; broadcast dims all alias the same address.
  if (newMask) {
    rewriter.create<triton::StoreOp>(loc, ptrBroadcast.getSrc(), newValue,
                                     newMask, storeOp.getBoundaryCheck(),
                                     storeOp.getCache(), storeOp.getEvict());
  } else {
    rewriter.create<triton::StoreOp>(loc, ptrBroadcast.getSrc(), newValue,
                                     storeOp.getBoundaryCheck(),
                                     storeOp.getCache(), storeOp.getEvict());
  }
  rewriter.eraseOp(storeOp);
  return success();
}

LogicalResult ZeroStrideMakeTensorPtrConverter::matchAndRewrite(
    triton::MakeTensorPtrOp op, PatternRewriter &rewriter) const {
  // Only rewrite when every stride is statically 0; any non-zero stride
  // would mean the dimension actually varies in memory and the broadcast
  // pattern doesn't apply.
  if (!llvm::all_of(op.getStrides(), [](Value stride) {
        Attribute attr;
        if (!matchPattern(stride, m_Constant(&attr)))
          return false;
        if (auto intAttr = dyn_cast<IntegerAttr>(attr))
          return intAttr.getValue() == 0;
        return false;
      })) {
    return failure();
  }

  auto isAllowedUser = [](Operation *user) {
    // FIXME: temporary solution
    // return isa<triton::LoadOp, triton::StoreOp, triton::AdvanceOp>(user);
    return isa<triton::LoadOp>(user);
  };
  if (!llvm::all_of(op->getUsers(), isAllowedUser))
    return failure();

  Location loc = op.getLoc();
  Value base = op.getBase();

  auto ptrType = cast<triton::PointerType>(op.getResult().getType());
  auto pointeeTensorTy = cast<RankedTensorType>(ptrType.getPointeeType());

  SmallVector<triton::LoadOp> loadUsers;
  SmallVector<triton::StoreOp> storeUsers;
  SmallVector<triton::AdvanceOp> advanceUsers;
  for (Operation *user : op->getUsers()) {
    if (auto loadOp = dyn_cast<triton::LoadOp>(user))
      loadUsers.push_back(loadOp);
    else if (auto storeOp = dyn_cast<triton::StoreOp>(user))
      storeUsers.push_back(storeOp);
    else if (auto advanceOp = dyn_cast<triton::AdvanceOp>(user))
      advanceUsers.push_back(advanceOp);
  }

  // Rewrite loads: load a scalar from `base` directly (tt.load accepts a
  // pointer-to-scalar type) and splat it to the block shape. The
  // "splat" Triton op is the intended way to broadcast a scalar value into
  // a tensor of arbitrary shape — equivalent in semantics to what
  // `tt.broadcast(0-D-tensor -> shape)` does, but without the intermediate
  // 0-D tensorization step.
  auto rewriteLoad = [&](triton::LoadOp loadOp) {
    rewriter.setInsertionPoint(loadOp);
    auto scalarLoad = rewriter.create<triton::LoadOp>(
        loc, base, /*mask=*/Value(), /*other=*/Value(), loadOp.getCache(),
        loadOp.getEvict(), loadOp.getIsVolatile());

    auto broadcasted = rewriter.create<triton::SplatOp>(loc, pointeeTensorTy,
                                                        scalarLoad.getResult());
    rewriter.replaceOp(loadOp, broadcasted.getResult());
  };

  for (triton::LoadOp loadOp : loadUsers)
    rewriteLoad(loadOp);

  rewriter.eraseOp(op);
  return success();
}

LogicalResult PromotePointerIterArgsPattern::matchAndRewrite(
    scf::ForOp forOp, PatternRewriter &rewriter) const {
  // 1. Check if the loop meets transformation conditions
  if (failed(matchLoop(forOp))) {
    return failure();
  }

  // 2. Collect pointer iteration arguments to be processed
  if (!failed(matchAndRewriteForAddPtr(forOp, rewriter))) {
    return success();
  }
  return matchAndRewriteAdvancePtr(forOp, rewriter);
}

// Transform a for loop that uses pointer iteration arguments into one that uses
// integer offsets instead. This pattern handles the specific case where:
// 1. The loop has pointer iteration arguments of type like
// tensor<1024x!tt.ptr<f32>>
// 2. Each pointer is used in a load/store operation and then incremented by
//    a constant offset via tt.addptr
// 3. The updated pointer (from addptr) is yielded back as the next iteration
// value
//
// The transformation converts:
//   scf.for iter_args(%ptr = %base_ptr) {
//     %val = tt.load %ptr
//     tt.store %other_ptr, %val
//     %new_ptr = tt.addptr %ptr, %offset
//     scf.yield %new_ptr
//   }
//
// Into:
//   scf.for iter_args(%offset_int = 0) {
//     %splat_offset = tt.splat %offset_int
//     %current_ptr = tt.addptr %base_ptr, %splat_offset
//     %val = tt.load %current_ptr
//     tt.store %other_ptr, %val
//     %new_offset = arith.addi %offset_int, %const_offset
//     scf.yield %new_offset
//   }
//
LogicalResult PromotePointerIterArgsPattern::matchAndRewriteForAddPtr(
    scf::ForOp forOp, PatternRewriter &rewriter) const {
  // 1. Collect pointer iteration arguments to be processed
  auto pointerArgsInfo = collectPointerIterArgs(forOp);
  if (pointerArgsInfo.empty()) {
    return failure();
  }
  // 2. Create new iteration argument types and initial values
  auto [newInitArgs, newIterArgTypes, indexMap] =
      createNewIterArgs(forOp, pointerArgsInfo, rewriter);
  // 3. Create the new for loop
  auto newForOp =
      createNewForLoop(forOp, newInitArgs, newIterArgTypes, rewriter);
  // 4. Rewrite the loop body
  if (failed(rewriteLoopBody(forOp, newForOp, pointerArgsInfo, indexMap,
                             rewriter))) {
    rewriter.eraseOp(newForOp);
    return failure();
  }
  // 5. Replace original loop results
  if (failed(replaceResults(forOp, newForOp, pointerArgsInfo, indexMap,
                            rewriter))) {
    rewriter.eraseOp(newForOp);
    return failure();
  }
  return success();
}

// Transform a for loop that uses pointer iteration arguments into one that uses
// integer offsets instead. This pattern handles the specific case where:
// 1. The loop has pointer iteration arguments of type like
// !tt.ptr<tensor<1024x>>
// 2. Each pointer is used in a load/store operation and then incremented by
//    a constant offset via tt.advanceptr
// 3. The updated pointer (from advanceptr) is yielded back as the next
// iteration value
//
// The transformation converts:
//   scf.for iter_args(%ptr1 = %base_ptr1, %ptr2 = %base_ptr2 ) {
//     %val = tt.load %ptr1
//     tt.store %base_ptr2, %val
//     %new_ptr1 = tt.adcanceptr %ptr1, %offset
//     %new_ptr2 = tt.adcanceptr %ptr2, %offset
//     scf.yield %new_ptr1,%new_ptr2
//   }
//
// Into:
// The transformation converts:
//   scf.for iter_args(%offset_int1 = 0,%offset_int2 = 0) {
//     %new_ptr1 = tt.adcanceptr %ptr1, %offset_int1
//     %new_ptr2 = tt.adcanceptr %ptr2, %offset_int1
//     %val = tt.load %new_ptr1
//     tt.store %new_ptr2, %val
//     %new_offset1 = arith.addi %offset_int, %const_offset
//     %new_offset2 = arith.addi %offset_int, %const_offset
//     scf.yield %new_offset1,%new_offset2
//   }
//
LogicalResult PromotePointerIterArgsPattern::matchAndRewriteAdvancePtr(
    scf::ForOp forOp, PatternRewriter &rewriter) const {
  // 1. Check if the loop meets transformation conditions
  if (failed(matchLoop(forOp))) {
    return failure();
  }
  // 2. Collect pointer iteration arguments to be processed
  auto pointerArgsInfo = collectPointerIterArgsForAdvancePtr(forOp);
  if (pointerArgsInfo.size() == 0) {
    return failure();
  }
  // 3. Create new iteration argument types and initial values
  auto [newInitArgs, newIterArgTypes, indexMap] =
      createNewIterArgsForAdvancePtr(forOp, pointerArgsInfo, rewriter);

  // 4. Create the new for loop
  auto newForOp =
      createNewForLoop(forOp, newInitArgs, newIterArgTypes, rewriter);
  // 5. Rewrite the loop body
  if (failed(rewriteLoopBodyForAdvancePtr(forOp, newForOp, pointerArgsInfo,
                                          indexMap, rewriter))) {
    rewriter.eraseOp(newForOp);
    return failure();
  }
  rewriter.replaceOp(forOp, newForOp);
  return success();
}

LogicalResult PromotePointerIterArgsPattern::matchLoop(scf::ForOp forOp) const {
  auto lowerBound = forOp.getLowerBound();
  auto upperBound = forOp.getUpperBound();
  auto step = forOp.getStep();
  if (!matchPattern(lowerBound, m_Constant()) ||
      !matchPattern(upperBound, m_Constant()) ||
      !matchPattern(step, m_Constant())) {
    return failure();
  }
  return success();
}

SmallVector<PromotePointerIterArgsPattern::PointerArgInfo>
PromotePointerIterArgsPattern::collectPointerIterArgs(scf::ForOp forOp) const {
  SmallVector<PointerArgInfo> result;
  auto &loopBody = *forOp.getBody();
  for (auto [idx, iterArg] : llvm::enumerate(forOp.getRegionIterArgs())) {
    if (isPointerIterArg(iterArg)) {
      auto info = analyzePointerIterArg(iterArg, loopBody);
      if (info.has_value()) {
        info->oldIndex = static_cast<unsigned>(idx),
        info->basePointer = forOp.getInitArgs()[idx],
        result.push_back(info.value());
      }
    }
  }
  return result;
}

SmallVector<PromotePointerIterArgsPattern::PointerArgInfo>
PromotePointerIterArgsPattern::collectPointerIterArgsForAdvancePtr(
    scf::ForOp forOp) const {
  SmallVector<PointerArgInfo> result;
  auto &loopBody = *forOp.getBody();

  for (auto [idx, iterArg] : llvm::enumerate(forOp.getRegionIterArgs())) {
    if (isa<triton::PointerType>(iterArg.getType())) {
      auto info = analyzePointerIterArgForAdvancePtr(iterArg, loopBody);
      if (info.has_value()) {
        info->oldIndex = static_cast<unsigned>(idx);
        info->basePointer = forOp.getInitArgs()[idx];
        result.push_back(info.value());
      }
    }
  }
  return result;
}

bool PromotePointerIterArgsPattern::isPointerIterArg(Value iterArg) const {
  auto ptrType = dyn_cast<TensorType>(iterArg.getType());
  return ptrType && isa<triton::PointerType>(ptrType.getElementType());
}

std::optional<PromotePointerIterArgsPattern::PointerArgInfo>
PromotePointerIterArgsPattern::analyzePointerIterArg(Value iterArg,
                                                     Block &loopBody) const {
  int memCount =
      0; // Count of memory operations (load/store) using this pointer
  int addPtrCount = 0;          // Count of addptr operations on this pointer
  Value addPtrResult = nullptr; // Result of the addptr operation
  Value offset = nullptr;       // Offset value used in addptr
  Value addPtrValue = nullptr;  // The addptr operation result value

  for (auto &op : loopBody) {
    TypeSwitch<Operation *>(&op)
        .Case<triton::LoadOp, triton::StoreOp>([&](auto memoryOp) {
          // Check if this memory operation uses the pointer we're analyzing
          if (memoryOp.getPtr() == iterArg)
            ++memCount;
        })
        .Case<triton::AddPtrOp>([&](auto addPtrOp) {
          // Check if this addptr operation updates the pointer we're analyzing
          if (addPtrOp.getPtr() == iterArg) {
            ++addPtrCount;
            addPtrResult = addPtrOp.getResult();
            offset = addPtrOp.getOffset();
            addPtrValue = addPtrOp.getResult();
          }
        })
        .Default([](auto) {}); // Ignore other operations
  }

  // Check the terminator to see if the addptr result is yielded
  auto yieldOp = dyn_cast<scf::YieldOp>(loopBody.getTerminator());
  if (!yieldOp)
    return std::nullopt;

  bool isYielded = false;
  for (auto operand : yieldOp.getOperands()) {
    if (operand == addPtrResult) {
      isYielded = true;
      break;
    }
  }

  // Pattern matched if:
  // 1. Exactly one addptr operation on this pointer
  // 2. At least one memory operation using this pointer
  // 3. The addptr result is yielded
  if (addPtrCount == 1 && memCount >= 1 && isYielded) {
    return PointerArgInfo{
        .oldIndex = 0,
        .basePointer = nullptr, // Will be set in collectPointerIterArgs
        .offsetValue = offset,
        .newIterArg = nullptr, // Will be set in createNewIterArgs
        .addPtrValue = addPtrValue};
  }
  return std::nullopt;
}

std::optional<PromotePointerIterArgsPattern::PointerArgInfo>
PromotePointerIterArgsPattern::analyzePointerIterArgForAdvancePtr(
    Value iterArg, Block &loopBody) const {
  int memCount =
      0; // Count of memory operations (load/store) using this pointer
  int advancePtrCount = 0; // Count of advanceptr operations on this pointer
  Value advancePtrResult = nullptr; // Result of the addptr operation
  SmallVector<Value> offsetValues;
  Value advancePtrValue = nullptr; // The addptr operation result value
  int nonZeroConstant = 0;         // Number of non-zero constants
  for (auto &op : loopBody) {
    TypeSwitch<Operation *>(&op)
        .Case<triton::LoadOp, triton::StoreOp>([&](auto memoryOp) {
          // Check if this memory operation uses the pointer we're analyzing
          if (memoryOp.getPtr() == iterArg)
            ++memCount;
        })
        .Case<triton::AdvanceOp>([&](triton::AdvanceOp advancePtrOp) {
          // Check if this addptr operation updates the pointer we're analyzing
          if (advancePtrOp.getPtr() == iterArg) {
            ++advancePtrCount;
            advancePtrResult = advancePtrOp.getResult();
            for (auto offsetVal : advancePtrOp.getOffsets()) {
              if (auto offsetInt = getConstantIntValue(offsetVal)) {
                if (*offsetInt != 0) {
                  ++nonZeroConstant;
                }
                offsetValues.push_back(offsetVal);
              }
            }
            advancePtrValue = advancePtrOp.getResult();
          }
        })
        .Default([](auto) {}); // Ignore other operations
  }
  // Check the terminator to see if the addptr result is yielded
  auto yieldOp = dyn_cast<scf::YieldOp>(loopBody.getTerminator());
  if (!yieldOp)
    return std::nullopt;
  bool isYielded = false;
  for (auto operand : yieldOp.getOperands()) {
    if (operand == advancePtrResult) {
      isYielded = true;
      break;
    }
  }
  // Pattern matched if:
  // 1. Exactly one addptr operation on this pointer
  // 2. At least one memory operation using this pointer
  // 3. The addptr result is yielded
  if (advancePtrCount == 1 && nonZeroConstant == 1 && memCount >= 1 &&
      isYielded) {
    return PointerArgInfo{
        .oldIndex = 0,
        .basePointer = nullptr, // Will be set in collectPointerIterArgs
        .offsetValue = nullptr,
        .newIterArg = nullptr, // Will be set in createNewIterArgs
        .addPtrValue = advancePtrValue,
        .offsetValues = offsetValues};
  }
  return std::nullopt;
}

std::tuple<SmallVector<Value>, SmallVector<Type>, DenseMap<unsigned, unsigned>>
PromotePointerIterArgsPattern::createNewIterArgs(
    scf::ForOp forOp, ArrayRef<PointerArgInfo> pointerArgs,
    PatternRewriter &rewriter) const {
  SmallVector<Value> newInitArgs;
  SmallVector<Type> newIterArgTypes;
  DenseMap<unsigned, unsigned> indexMap;

  for (unsigned i = 0; i < forOp.getInitArgs().size(); ++i) {
    if (isPointerArgIndex(pointerArgs, i)) {
      // Replace pointer with integer offset (initialized to 0)
      Value zero = rewriter.create<arith::ConstantIntOp>(forOp.getLoc(), 0,
                                                         INT_TYPE_BIT_WIDTH);
      newInitArgs.push_back(zero);
      newIterArgTypes.push_back(rewriter.getIntegerType(INT_TYPE_BIT_WIDTH));
    } else {
      // Preserve original argument unchanged
      newInitArgs.push_back(forOp.getInitArgs()[i]);
      newIterArgTypes.push_back(forOp.getInitArgs()[i].getType());
    }
    // Identity mapping: argument count and order unchanged，
    // may change in future
    indexMap[i] = i;
  }

  return {newInitArgs, newIterArgTypes, indexMap};
}

std::tuple<SmallVector<Value>, SmallVector<Type>, DenseMap<unsigned, unsigned>>
PromotePointerIterArgsPattern::createNewIterArgsForAdvancePtr(
    scf::ForOp forOp, SmallVector<PointerArgInfo> &pointerArgs,
    PatternRewriter &rewriter) const {
  DenseMap<unsigned, unsigned> indexMap;
  SmallVector<Value> newInitArgs;
  SmallVector<Type> newIterArgTypes;
  SmallVector<Value> newInitArgsTemp;
  SmallVector<Type> newIterArgTypesTemp;

  for (unsigned i = 0; i < forOp.getInitArgs().size(); ++i) {
    PointerArgInfo *infoTemp = nullptr;
    bool isMatch = false;

    // find the matching pointerArg
    for (size_t k = 0; k < pointerArgs.size(); ++k) {
      auto &info = pointerArgs[k];
      if (info.oldIndex == i) {
        infoTemp = &info;
        isMatch = true;
        break;
      }
    }

    if (!isMatch) {
      // Preserve original argument unchanged
      newInitArgsTemp.push_back(forOp.getInitArgs()[i]);
      newIterArgTypesTemp.push_back(forOp.getInitArgs()[i].getType());
    } else if (infoTemp != nullptr) {
      // Replace pointer with integer offset (initialized to 0)
      SmallVector<Value> newInitArgs;
      SmallVector<Type> newIterArgTypes;

      if (infoTemp->offsetValues.empty()) {
        Value zero =
            rewriter.create<arith::ConstantIntOp>(forOp.getLoc(), 0, 32);
        newInitArgs.push_back(zero);
        newIterArgTypes.push_back(rewriter.getIntegerType(INT_TYPE_BIT_WIDTH));
        newInitArgsTemp.push_back(zero);
        newIterArgTypesTemp.push_back(
            rewriter.getIntegerType(INT_TYPE_BIT_WIDTH));
      } else {
        Value zero =
            rewriter.create<arith::ConstantIntOp>(forOp.getLoc(), 0, 32);
        for (size_t j = 0; j < infoTemp->offsetValues.size(); ++j) {
          Value offset = infoTemp->offsetValues[j];
          if (auto maskInt = getConstantIntValue(offset)) {
            newInitArgs.push_back(zero);
            newIterArgTypes.push_back(
                rewriter.getIntegerType(INT_TYPE_BIT_WIDTH));
            if (*maskInt == 0) {
              continue;
            }
            newInitArgsTemp.push_back(zero);
            newIterArgTypesTemp.push_back(
                rewriter.getIntegerType(INT_TYPE_BIT_WIDTH));
          }
        }
      }

      infoTemp->newInitArgs = newInitArgs;
      infoTemp->newIterArgTypes = newIterArgTypes;
    } else {
      newInitArgsTemp.push_back(forOp.getInitArgs()[i]);
      newIterArgTypesTemp.push_back(forOp.getInitArgs()[i].getType());
    }
    // may change in future
    indexMap[i] = i;
  }
  return {newInitArgsTemp, newIterArgTypesTemp, indexMap};
}

scf::ForOp PromotePointerIterArgsPattern::createNewForLoop(
    scf::ForOp forOp, ArrayRef<Value> newInitArgs,
    ArrayRef<Type> newIterArgTypes, PatternRewriter &rewriter) const {
  return rewriter.create<scf::ForOp>(forOp.getLoc(), forOp.getLowerBound(),
                                     forOp.getUpperBound(), forOp.getStep(),
                                     newInitArgs);
}

LogicalResult PromotePointerIterArgsPattern::rewriteLoopBody(
    scf::ForOp oldForOp, scf::ForOp newForOp,
    SmallVector<PointerArgInfo> &pointerArgs,
    DenseMap<unsigned, unsigned> &indexMap, PatternRewriter &rewriter) const {
  Block &oldBody = *oldForOp.getBody();
  Block &newBody = *newForOp.getBody();

  rewriter.setInsertionPointToStart(&newBody);

  // Create IR mapping that maps original values to their transformed
  // equivalents
  IRMapping mapping =
      createIRMapping(oldForOp, newForOp, pointerArgs, indexMap, rewriter);
  // Clone instructions from original loop body, applying the mapping
  return cloneInstructions(oldBody, newBody, pointerArgs, indexMap, mapping,
                           rewriter);
}

LogicalResult PromotePointerIterArgsPattern::rewriteLoopBodyForAdvancePtr(
    scf::ForOp oldForOp, scf::ForOp newForOp,
    SmallVector<PointerArgInfo> &pointerArgs,
    DenseMap<unsigned, unsigned> &indexMap, PatternRewriter &rewriter) const {
  Block &oldBody = *oldForOp.getBody();
  Block &newBody = *newForOp.getBody();
  rewriter.setInsertionPointToStart(&newBody);
  // Create IR mapping that maps original values to their transformed
  // equivalents
  IRMapping mapping = createIRMappingForAdvancePtr(
      oldForOp, newForOp, pointerArgs, indexMap, rewriter);
  // Clone instructions from original loop body, applying the mapping
  return cloneInstructionsForAdvancePtr(oldBody, newBody, pointerArgs, indexMap,
                                        mapping, rewriter);
}

IRMapping PromotePointerIterArgsPattern::createIRMapping(
    scf::ForOp oldForOp, scf::ForOp newForOp,
    SmallVector<PointerArgInfo> &pointerArgs,
    DenseMap<unsigned, unsigned> &indexMap, PatternRewriter &rewriter) const {
  IRMapping mapping;
  mapping.map(oldForOp.getInductionVar(), newForOp.getInductionVar());

  // Process iteration arguments
  for (unsigned i = 0; i < oldForOp.getRegionIterArgs().size(); ++i) {
    Value oldIterArg = oldForOp.getRegionIterArgs()[i];
    Value newIterArg = newForOp.getRegionIterArgs()[indexMap[i]];

    if (isPointerArgIndex(pointerArgs, i)) {
      // Update the PointerArgInfo with the new integer iteration argument
      for (auto &info : pointerArgs) {
        if (info.oldIndex == i) {
          info.newIterArg = newIterArg;
          break;
        }
      }

      // Map original pointer argument to a reconstructed pointer
      mapping.map(oldIterArg,
                  rebuildPointer(oldForOp, pointerArgs, i, rewriter));
    } else {
      // Direct mapping for non-pointer arguments
      mapping.map(oldIterArg, newIterArg);
    }
  }
  return mapping;
}

IRMapping PromotePointerIterArgsPattern::createIRMappingForAdvancePtr(
    scf::ForOp oldForOp, scf::ForOp newForOp,
    SmallVector<PointerArgInfo> &pointerArgs,
    DenseMap<unsigned, unsigned> &indexMap, PatternRewriter &rewriter) const {
  IRMapping mapping;
  mapping.map(oldForOp.getInductionVar(), newForOp.getInductionVar());
  // Process iteration arguments
  for (unsigned i = 0; i < oldForOp.getRegionIterArgs().size(); ++i) {
    Value oldIterArg = oldForOp.getRegionIterArgs()[i];
    Value newIterArg = newForOp.getRegionIterArgs()[indexMap[i]];
    if (isPointerArgIndex(pointerArgs, i)) {
      // Update the PointerArgInfo with the new integer iteration argument
      for (auto &info : pointerArgs) {
        if (info.oldIndex == i) {
          info.newIterArg = newIterArg;
          break;
        }
      }
      Value newIterArgTemp =
          rebuildPointerForAdvancePtr(oldForOp, pointerArgs, i, rewriter);
      // Map original pointer argument to a reconstructed pointer
      mapping.map(oldIterArg, newIterArgTemp);
    } else {
      // Direct mapping for non-pointer arguments
      mapping.map(oldIterArg, newIterArg);
    }
  }
  return mapping;
}

bool PromotePointerIterArgsPattern::isPointerArgIndex(
    ArrayRef<PointerArgInfo> pointerArgs, unsigned idx) const {
  for (auto &info : pointerArgs) {
    if (info.oldIndex == idx)
      return true;
  }
  return false;
}

Value PromotePointerIterArgsPattern::rebuildPointer(
    scf::ForOp forOp, ArrayRef<PointerArgInfo> pointerArgs, unsigned idx,
    PatternRewriter &rewriter) const {
  const PointerArgInfo *info = nullptr;
  for (auto &argInfo : pointerArgs) {
    if (argInfo.oldIndex == idx) {
      info = &argInfo;
      break;
    }
  }
  if (!info)
    return nullptr;
  // Create splat operation to broadcast integer offset to tensor shape
  auto baseType = info->basePointer.getType();
  Value splatOffset = nullptr;
  if (auto rankedType = dyn_cast<RankedTensorType>(baseType)) {
    // Get the shape of the original tensor
    auto shape = rankedType.getShape();

    splatOffset = rewriter.create<triton::SplatOp>(
        forOp.getLoc(), RankedTensorType::get(shape, rewriter.getI32Type()),
        info->newIterArg);
  } else {
    return nullptr;
  }
  // Create addptr operation: base pointer + splatted offset
  return rewriter.create<triton::AddPtrOp>(forOp.getLoc(),
                                           info->basePointer.getType(),
                                           info->basePointer, splatOffset);
}

Value PromotePointerIterArgsPattern::rebuildPointerForAdvancePtr(
    scf::ForOp forOp, ArrayRef<PointerArgInfo> pointerArgs, unsigned idx,
    PatternRewriter &rewriter) const {
  const PointerArgInfo *info = nullptr;
  for (auto &argInfo : pointerArgs) {
    if (argInfo.oldIndex == idx) {
      info = &argInfo;
      break;
    }
  }
  if (!info) {
    return nullptr;
  }
  SmallVector<Value> advanceOpOld;
  advanceOpOld.push_back(info->basePointer);
  for (size_t i = 0; i < info->offsetValues.size(); ++i) {
    if (auto maskInt = getConstantIntValue(info->offsetValues[i])) {
      if (*maskInt != 0) {
        advanceOpOld.push_back(info->newIterArg);
        continue;
      }
    }
    advanceOpOld.push_back(info->offsetValues[i]);
  }
  auto advanceOp = rewriter.create<triton::AdvanceOp>(
      forOp.getLoc(), info->basePointer.getType(), advanceOpOld);
  return advanceOp;
}

LogicalResult PromotePointerIterArgsPattern::cloneInstructions(
    Block &oldBody, Block &newBody, ArrayRef<PointerArgInfo> pointerArgs,
    DenseMap<unsigned, unsigned> &indexMap, IRMapping &mapping,
    PatternRewriter &rewriter) const {
  // Collect all operations from the old loop body except the terminator
  SmallVector<Operation *> toClone;
  for (auto &op : oldBody.without_terminator()) {
    toClone.push_back(&op);
  }
  // Build a set of addptr operations to skip (those that update pointer
  // iteration arguments)
  DenseSet<Value> addPtrOpsToSkip;
  for (const auto &info : pointerArgs) {
    if (info.addPtrValue) {
      addPtrOpsToSkip.insert(info.addPtrValue);
    }
  }
  // Clone all operations except the skipped addptr operations
  for (auto *op : toClone) {
    // Only skip addptr operations that are updating pointer iteration arguments
    if (auto addPtrOp = dyn_cast<triton::AddPtrOp>(op)) {
      if (addPtrOpsToSkip.contains(addPtrOp.getResult())) {
        continue;
      }
    }
    rewriter.clone(*op, mapping);
  }
  // Handle the yield terminator separately
  auto yieldOp = dyn_cast<scf::YieldOp>(oldBody.getTerminator());
  if (!yieldOp) {
    return failure();
  }
  return cloneYieldOp(yieldOp, pointerArgs, indexMap, mapping, rewriter);
}

LogicalResult PromotePointerIterArgsPattern::cloneInstructionsForAdvancePtr(
    Block &oldBody, Block &newBody, ArrayRef<PointerArgInfo> pointerArgs,
    DenseMap<unsigned, unsigned> &indexMap, IRMapping &mapping,
    PatternRewriter &rewriter) const {
  // Collect all operations from the old loop body except the terminator
  SmallVector<Operation *> toClone;
  for (auto &op : oldBody.without_terminator()) {
    toClone.push_back(&op);
  }
  // Build a set of addptr operations to skip (those that update pointer
  // iteration arguments)
  DenseSet<Value> advancePtrOpsToSkip;
  for (const auto &info : pointerArgs) {
    if (info.addPtrValue) {
      advancePtrOpsToSkip.insert(info.addPtrValue);
    }
  }
  // Clone all operations except the skipped addptr operations
  for (auto *op : toClone) {
    if (auto advancePtrOp = dyn_cast<triton::AdvanceOp>(op)) {
      if (advancePtrOpsToSkip.contains(advancePtrOp.getResult())) {
        continue;
      }
    }
    auto clonedOp = rewriter.clone(*op, mapping);
  }
  // Handle the yield terminator separately
  auto yieldOp = dyn_cast<scf::YieldOp>(oldBody.getTerminator());
  if (!yieldOp) {
    return failure();
  }
  return cloneYieldOpForAdvancePtr(yieldOp, pointerArgs, indexMap, mapping,
                                   rewriter);
}

LogicalResult PromotePointerIterArgsPattern::cloneYieldOpForAdvancePtr(
    scf::YieldOp yieldOp, ArrayRef<PointerArgInfo> pointerArgs,
    DenseMap<unsigned, unsigned> &indexMap, IRMapping &mapping,
    PatternRewriter &rewriter) const {
  SmallVector<Value> newOperands;
  // Process each operand of the original yield operation
  for (unsigned i = 0; i < yieldOp.getNumOperands(); ++i) {
    auto operand = yieldOp.getOperand(i);
    if (isPointerArgIndex(pointerArgs, i)) {
      // For pointer arguments being promoted: create integer addition
      SmallVector<Value> intResult =
          createOffsetsForAdvancePtr(i, pointerArgs, indexMap, rewriter);
      for (size_t i = 0; i < intResult.size(); ++i) {
        newOperands.push_back(intResult[i]);
      }
    } else {
      // For other arguments: use the value from the IR mapping
      Value mappedValue = mapping.lookupOrDefault(operand);
      newOperands.push_back(mappedValue);
    }
  }
  // Validate that all new operands are non-null
  for (size_t i = 0; i < newOperands.size(); ++i) {
    auto v = newOperands[i];
    if (!v) {
      return failure();
    }
  }
  // Create the new yield operation in the transformed loop
  auto newYieldOp =
      rewriter.create<scf::YieldOp>(yieldOp.getLoc(), newOperands);
  return success();
}

LogicalResult PromotePointerIterArgsPattern::cloneYieldOp(
    scf::YieldOp yieldOp, ArrayRef<PointerArgInfo> pointerArgs,
    DenseMap<unsigned, unsigned> &indexMap, IRMapping &mapping,
    PatternRewriter &rewriter) const {
  SmallVector<Value> newOperands;
  // Process each operand of the original yield operation
  for (unsigned i = 0; i < yieldOp.getNumOperands(); ++i) {
    if (isPointerArgIndex(pointerArgs, i)) {
      // For pointer arguments being promoted: create integer addition
      Value intResult = createIntegerAdd(i, pointerArgs, indexMap, rewriter);
      newOperands.push_back(intResult);
    } else {
      // For other arguments: use the value from the IR mapping
      newOperands.push_back(mapping.lookupOrDefault(yieldOp.getOperand(i)));
    }
  }
  // Validate that all new operands are non-null
  for (auto v : newOperands) {
    if (!v) {
      return failure();
    }
  }
  // Create the new yield operation in the transformed loop
  rewriter.create<scf::YieldOp>(yieldOp.getLoc(), newOperands);
  return success();
}

SmallVector<Value> PromotePointerIterArgsPattern::createOffsetsForAdvancePtr(
    unsigned idx, ArrayRef<PointerArgInfo> pointerArgs,
    DenseMap<unsigned, unsigned> &indexMap, PatternRewriter &rewriter) const {
  const PointerArgInfo *info = nullptr;
  for (auto &argInfo : pointerArgs) {
    if (argInfo.oldIndex == idx) {
      info = &argInfo;
      break;
    }
  }
  if (!info) {
    return SmallVector<Value>();
  }

  if (info->offsetValues.empty()) {
    return SmallVector<Value>();
  }
  SmallVector<Value> offsets;
  // Process all offset values in the collection
  for (size_t i = 0; i < info->offsetValues.size(); ++i) {
    Value offset = info->offsetValues[i];
    Value newArg = info->newInitArgs[i];
    // Get a location for creating constants
    Location loc = offset.getLoc();
    // Check if this offset is a constant
    Attribute offsetAttr;
    if (matchPattern(offset, m_Constant(&offsetAttr))) {
      // Case 1: Integer attribute (scalar constant)
      if (auto intAttr = dyn_cast<IntegerAttr>(offsetAttr)) {
        if (intAttr.getInt() == 0) {
          continue;
        }

        Value result =
            rewriter.create<arith::AddIOp>(loc, info->newIterArg, offset);
        offsets.push_back(result);
      }
    }
  }
  return offsets;
}

Value PromotePointerIterArgsPattern::createIntegerAdd(
    unsigned idx, ArrayRef<PointerArgInfo> pointerArgs,
    DenseMap<unsigned, unsigned> &indexMap, PatternRewriter &rewriter) const {
  const PointerArgInfo *info = nullptr;
  for (auto &argInfo : pointerArgs) {
    if (argInfo.oldIndex == idx) {
      info = &argInfo;
      break;
    }
  }
  if (!info)
    return nullptr;
  // Try to extract constant offset value
  Attribute offsetAttr;
  if (matchPattern(info->offsetValue, m_Constant(&offsetAttr))) {
    Location loc = info->offsetValue.getLoc();
    // Case 1: Integer attribute (scalar constant)
    if (auto intAttr = dyn_cast<IntegerAttr>(offsetAttr)) {
      Value constOffset = rewriter.create<arith::ConstantIntOp>(
          loc, intAttr.getInt(), INT_TYPE_BIT_WIDTH);
      return rewriter.create<arith::AddIOp>(loc, info->newIterArg, constOffset);
    }
    // Case 2: DenseElementsAttr (tensor constant)
    if (auto denseAttr = dyn_cast<DenseElementsAttr>(offsetAttr)) {
      // Check if it's a splat (all elements are the same)
      if (denseAttr.isSplat()) {
        // For integer-type DenseElementsAttr
        if (denseAttr.getElementType().isInteger(INT_TYPE_BIT_WIDTH)) {
          auto splatValue = denseAttr.getSplatValue<APInt>();
          Value constOffset = rewriter.create<arith::ConstantIntOp>(
              loc, splatValue.getZExtValue(), INT_TYPE_BIT_WIDTH);
          return rewriter.create<arith::AddIOp>(loc, info->newIterArg,
                                                constOffset);
        }
      } else {
        // If not a splat, but has only one element, we can still handle it
        if (denseAttr.getNumElements() == 1) {
          auto firstElement = *denseAttr.getValues<APInt>().begin();
          Value constOffset = rewriter.create<arith::ConstantIntOp>(
              loc, firstElement.getZExtValue(), INT_TYPE_BIT_WIDTH);
          return rewriter.create<arith::AddIOp>(loc, info->newIterArg,
                                                constOffset);
        }
      }
    }
  }

  // Return nullptr if offset is not a constant (pattern only handles constant
  // offsets)
  return nullptr;
}

LogicalResult PromotePointerIterArgsPattern::replaceResults(
    scf::ForOp oldForOp, scf::ForOp newForOp,
    ArrayRef<PointerArgInfo> pointerArgs,
    DenseMap<unsigned, unsigned> &indexMap, PatternRewriter &rewriter) const {
  SmallVector<Value> newResults;

  for (unsigned i = 0; i < oldForOp.getNumResults(); ++i) {
    if (isPointerArgIndex(pointerArgs, i)) {
      Value ptrResult = reconstructPointer(
          oldForOp, i, newForOp.getResult(indexMap[i]), pointerArgs, rewriter);
      newResults.push_back(ptrResult);
    } else {
      newResults.push_back(newForOp.getResult(indexMap[i]));
    }
  }

  for (auto v : newResults) {
    if (!v) {
      return failure();
    }
  }
  rewriter.replaceOp(oldForOp, newResults);
  return success();
}

SmallVector<Value> PromotePointerIterArgsPattern::reconstructPointerForAdvance(
    scf::ForOp forOp, unsigned idx, Value intResult,
    ArrayRef<PointerArgInfo> pointerArgs, PatternRewriter &rewriter) const {
  const PointerArgInfo *info = nullptr;
  for (auto &argInfo : pointerArgs) {
    if (argInfo.oldIndex == idx) {
      info = &argInfo;
      break;
    }
  }
  if (!info) {
    return SmallVector<Value>();
  }
  // Create splat operation to broadcast integer result to tensor shape
  auto baseType = info->basePointer.getType();
  SmallVector<Value> addPtr;
  // Create a tensor with the same shape, where all elements are the integer
  // result
  SmallVector<Value> newInitArgs = info->newInitArgs;
  for (size_t i = 0; i < newInitArgs.size(); ++i) {
    if (auto maskInt = getConstantIntValue(newInitArgs[i])) {
      if (*maskInt == 0) {
        continue;
      }
      Value constOffset = rewriter.create<arith::ConstantIntOp>(
          forOp.getLoc(), *maskInt, INT_TYPE_BIT_WIDTH);
      Value addPtrResult = rewriter.create<arith::AddIOp>(
          forOp.getLoc(), newInitArgs[i], constOffset);
      addPtr.push_back(addPtrResult);
    }
  }
  return addPtr;
}

Value PromotePointerIterArgsPattern::reconstructPointer(
    scf::ForOp forOp, unsigned idx, Value intResult,
    ArrayRef<PointerArgInfo> pointerArgs, PatternRewriter &rewriter) const {
  const PointerArgInfo *info = nullptr;
  for (auto &argInfo : pointerArgs) {
    if (argInfo.oldIndex == idx) {
      info = &argInfo;
      break;
    }
  }
  if (!info)
    return nullptr;

  // Create splat operation to broadcast integer result to tensor shape
  auto baseType = info->basePointer.getType();
  Value splatOffset = nullptr;
  if (auto rankedType = dyn_cast<RankedTensorType>(baseType)) {
    // Get the shape of the original tensor
    auto shape = rankedType.getShape();

    splatOffset = rewriter.create<triton::SplatOp>(
        forOp.getLoc(), RankedTensorType::get(shape, rewriter.getI32Type()),
        intResult);
  } else {
    return nullptr;
  }

  // Create a tensor with the same shape, where all elements are the integer
  // result
  return rewriter.create<triton::AddPtrOp>(forOp.getLoc(),
                                           info->basePointer.getType(),
                                           info->basePointer, splatOffset);
}

void SimplifyTensorIterArgsPattern::ShapeChainInfo::dump() const {
  LLVM_DEBUG({
    llvm::dbgs() << "ShapeChainInfo:\n";
    llvm::dbgs() << "  base: " << base << "\n";
    llvm::dbgs() << "  chain:\n";
    for (Operation *op : chain) {
      llvm::dbgs() << "    " << *op << "\n";
    }
  });
}

void SimplifyTensorIterArgsPattern::CandidateInfo::dump() const {
  LLVM_DEBUG({
    llvm::dbgs() << "CandidateInfo:\n";
    llvm::dbgs() << "  idx: " << idx << "\n";
    llvm::dbgs() << "  ShapeInfo: \n";
    shapeInfo.dump();
    for (Operation *op : arithOps) {
      llvm::dbgs() << "  ArithOp: " << *op << "\n";
    }
  });
}

Value SimplifyTensorIterArgsPattern::cloneShapeChain(
    Location loc, Value base, ArrayRef<Operation *> chain,
    PatternRewriter &rewriter) const {
  Value cur = base;
  for (Operation *op : chain) {
    if (auto splat = dyn_cast<triton::SplatOp>(op)) {
      auto dstTy = cast<RankedTensorType>(splat.getType());
      cur = rewriter.create<triton::SplatOp>(loc, dstTy, cur);
      continue;
    }
    if (auto bcast = dyn_cast<triton::BroadcastOp>(op)) {
      auto dstTy = cast<RankedTensorType>(bcast.getType());
      cur = rewriter.create<triton::BroadcastOp>(loc, dstTy, cur);
      continue;
    }
    if (auto expand = dyn_cast<triton::ExpandDimsOp>(op)) {
      auto dstTy = cast<RankedTensorType>(expand.getType());
      cur = rewriter.create<triton::ExpandDimsOp>(loc, dstTy, cur,
                                                  expand.getAxis());
      continue;
    }
    return Value();
  }
  return cur;
}

bool SimplifyTensorIterArgsPattern::isBlockArgumentFromAnotherForLoop(
    Value v) const {
  auto barg = dyn_cast<BlockArgument>(v);
  if (!barg) {
    return false;
  }

  Block *owner = barg.getOwner();
  if (!owner) {
    return false;
  }

  auto parentFor = dyn_cast_or_null<scf::ForOp>(owner->getParentOp());
  if (!parentFor) {
    return false;
  }

  // Only handle block args that belong to the body block of scf.for.
  if (&parentFor.getRegion().front() != owner) {
    return false;
  }

  // scf.for body block args layout:
  //   arg#0                : induction variable
  //   arg#1..arg#N         : region iter args
  unsigned argNo = barg.getArgNumber();
  if (argNo == 0) {
    // IV: not an iter arg init source.
    return false;
  }

  unsigned iterIdx = argNo - 1;
  if (iterIdx >= parentFor.getInitArgs().size()) {
    return false;
  }

  return true;
}

Value SimplifyTensorIterArgsPattern::normalizeInitArgForShapePeel(
    Value v) const {
  if (!isBlockArgumentFromAnotherForLoop(v)) {
    // Not a block argument from another for loop, return as is.
    return v;
  }
  auto barg = dyn_cast<BlockArgument>(v);
  if (!barg) {
    return v;
  }

  Block *owner = barg.getOwner();
  if (!owner) {
    return v;
  }

  auto parentFor = dyn_cast_or_null<scf::ForOp>(owner->getParentOp());
  if (!parentFor) {
    return v;
  }

  unsigned argNo = barg.getArgNumber();
  if (argNo == 0) {
    return v;
  }

  unsigned iterIdx = argNo - 1;
  if (iterIdx >= parentFor.getInitArgs().size()) {
    return v;
  }
  // For the simple nested relay case:
  // inner.initArg = outer.regionIterArg  ==> resolve to outer.initArg
  return parentFor.getInitArgs()[iterIdx];
}

std::optional<SimplifyTensorIterArgsPattern::RelayMapM1>
SimplifyTensorIterArgsPattern::getRelayMapM1(scf::ForOp innerFor,
                                             scf::ForOp outerFor,
                                             unsigned innerIdx) const {
  if (innerIdx >= innerFor.getInitArgs().size() ||
      innerIdx >= innerFor.getNumResults()) {
    return std::nullopt;
  }

  // 1) inner initArg -> outer iterArg idx
  Value innerInit = innerFor.getInitArgs()[innerIdx];
  if (!isBlockArgumentFromAnotherForLoop(innerInit)) {
    return std::nullopt;
  }

  auto barg = dyn_cast<BlockArgument>(innerInit);
  if (!barg) {
    return std::nullopt;
  }

  unsigned outerInitIdx = barg.getArgNumber() - 1;
  if (outerInitIdx >= outerFor.getRegionIterArgs().size()) {
    return std::nullopt;
  }

  // NEW: ensure outer iterArg lane is used only as the mapped inner initArg.
  // i.e. no extra users in outer body.
  Value outerIterArg = outerFor.getRegionIterArgs()[outerInitIdx];
  if (!outerIterArg.hasOneUse()) {
    return std::nullopt;
  }
  OpOperand &onlyUse = *outerIterArg.getUses().begin();
  if (onlyUse.getOwner() != innerFor.getOperation() ||
      onlyUse.get() != innerFor.getInitArgs()[innerIdx]) {
    return std::nullopt;
  }

  // 2) inner result -> outer yield slot
  if (!outerFor.getBody() || !outerFor.getBody()->mightHaveTerminator()) {
    return std::nullopt;
  }
  auto outerYield = dyn_cast<scf::YieldOp>(outerFor.getBody()->getTerminator());
  if (!outerYield) {
    return std::nullopt;
  }

  // only handle the case where inner result directly feeds into outer yield
  // operand, without any intermediate use/def.
  Value innerRes = innerFor.getResult(innerIdx);
  std::optional<unsigned> outerYieldIdx;
  for (unsigned k = 0; k < outerYield.getNumOperands(); ++k) {
    if (outerYield.getOperand(k) == innerRes) {
      outerYieldIdx = k;
      break;
    }
  }
  if (!outerYieldIdx.has_value()) {
    return std::nullopt;
  }

  if (outerInitIdx != *outerYieldIdx) {
    // The position of iterArg and yield operand should be the same in this
    // simple relay case.
    return std::nullopt;
  }

  return RelayMapM1{
      .innerIdx = innerIdx,
      .outerInitIdx = outerInitIdx,
      .outerYieldIdx = *outerYieldIdx,
  };
}

void SimplifyTensorIterArgsPattern::splitCandidatesByRelay(
    SmallVector<SimplifyTensorIterArgsPattern::CandidateInfo> all,
    SmallVector<SimplifyTensorIterArgsPattern::CandidateInfo> &locals,
    SmallVector<SimplifyTensorIterArgsPattern::CandidateInfo> &relays) const {
  for (const auto &c : all) {
    if (c.relayMap.has_value()) {
      relays.push_back(c);
    } else {
      locals.push_back(c);
    }
  }
}

std::optional<SimplifyTensorIterArgsPattern::ShapeChainInfo>
SimplifyTensorIterArgsPattern::peelShapeChain(Value v) const {
  ShapeChainInfo info;
  Value cur = v;
  SmallVector<Operation *> rev;
  while (Operation *def = cur.getDefiningOp()) {
    if (isa<triton::BroadcastOp, triton::SplatOp, triton::ExpandDimsOp>(def)) {
      rev.push_back(def);
      cur = def->getOperand(0);
      continue;
    }
    break;
  }
  if (rev.empty()) {
    return std::nullopt;
  }
  std::reverse(rev.begin(), rev.end());
  info.base = cur;
  info.chain = std::move(rev);
  return info;
}

bool SimplifyTensorIterArgsPattern::isArithWithConst(Operation *op,
                                                     Value curVal,
                                                     Value &nextVal,
                                                     Value &constVal) const {
  nextVal = Value();
  constVal = Value();

  Value lhs;
  Value rhs;
  if (!extractBinaryArithOperands(op, lhs, rhs)) {
    return false;
  }

  bool lhsIsConst = matchPattern(lhs, m_Constant());
  bool rhsIsConst = matchPattern(rhs, m_Constant());
  if (lhsIsConst == rhsIsConst) {
    return false;
  }

  if (lhs == curVal && rhsIsConst) {
    nextVal = op->getResult(0);
    constVal = rhs;
    return true;
  }
  if (rhs == curVal && lhsIsConst) {
    nextVal = op->getResult(0);
    constVal = lhs;
    return true;
  }

  return false;
}

Value SimplifyTensorIterArgsPattern::getNewConstLikeOperand(
    Value cst, Type targetTy, PatternRewriter &rewriter) const {
  Attribute attr;
  if (!matchPattern(cst, m_Constant(&attr))) {
    return Value();
  }
  auto tensorTy = dyn_cast<RankedTensorType>(targetTy);
  if (!tensorTy) {
    return Value();
  }
  // Only support arith.constant dense right now.
  auto dense = dyn_cast<DenseElementsAttr>(attr);
  if (!dense || !dense.isSplat()) {
    return Value();
  }
  auto splatAttr =
      DenseElementsAttr::get(tensorTy, dense.getSplatValue<Attribute>());
  return rewriter.create<arith::ConstantOp>(cst.getLoc(), tensorTy, splatAttr);
}

bool SimplifyTensorIterArgsPattern::canBuildConstLikeOperand(
    Value cst, Type targetTy) const {
  Attribute attr;
  if (!matchPattern(cst, m_Constant(&attr))) {
    return false;
  }
  auto tensorTy = dyn_cast<RankedTensorType>(targetTy);
  if (!tensorTy) {
    return false;
  }
  auto dense = dyn_cast<DenseElementsAttr>(attr);
  return dense && dense.isSplat();
}

LogicalResult SimplifyTensorIterArgsPattern::collectReverseLinearYieldPath(
    Value yielded, Value iterArg,
    SmallVectorImpl<Operation *> &opsInExecOrder) const {
  SmallVector<Operation *> revOps;
  Value cur = yielded;

  while (cur != iterArg) {
    Operation *def = cur.getDefiningOp();
    if (!def || isa<scf::ForOp, scf::YieldOp>(def)) {
      return failure();
    }

    Value lhs;
    Value rhs;
    if (!extractBinaryArithOperands(def, lhs, rhs)) {
      return failure();
    }

    bool lhsIsConst = matchPattern(lhs, m_Constant());
    bool rhsIsConst = matchPattern(rhs, m_Constant());
    if (lhsIsConst == rhsIsConst) {
      return failure();
    }

    Value upstream = lhsIsConst ? rhs : lhs;
    if (!upstream) {
      return failure();
    }

    // Safety: only accept a strict linear chain.
    // The current def result must be consumed exclusively by `cur`.
    // If there is any extra user, rewriting this lane may break semantics.
    if (def->getNumResults() != 1) {
      return failure();
    }
    Value defRes = def->getResult(0);
    if (!defRes.hasOneUse()) {
      return failure();
    }
    OpOperand &onlyUse = *defRes.getUses().begin();
    if (onlyUse.get() != cur) {
      return failure();
    }

    revOps.push_back(def);
    cur = upstream;
  }

  opsInExecOrder.assign(revOps.rbegin(), revOps.rend());
  return success();
}

LogicalResult SimplifyTensorIterArgsPattern::matchAndRewrite(
    scf::ForOp forOp, PatternRewriter &rewriter) const {
  LLVM_DEBUG({
    llvm::dbgs() << "Now Handling For Op: \n";
    forOp.dump();
  });

  // if you want only simplify iter args once to avoid infinite pattern
  // application, return failure when meeting done label
  if (forOp->hasAttr(kSimplifiedAttr)) {
    LLVM_DEBUG({
      llvm::dbgs() << "This For Op has been simplified before.\n";
      forOp.dump();
    });
  }
  // If the forOp has been marked as failed before, it means we attempted to
  // simplify it but couldn't, so we should not try again.
  if (forOp->hasAttr(kFailedAttr) || forOp->hasAttr(kIncompleteAttr)) {
    return failure();
  }

  Block &oldBody = *forOp.getBody();
  if (!oldBody.mightHaveTerminator()) {
    return failure();
  }
  auto oldYield = dyn_cast<scf::YieldOp>(oldBody.getTerminator());
  if (!oldYield) {
    return failure();
  }

  SmallVector<CandidateInfo> candidates;
  auto regionIterArgs = forOp.getRegionIterArgs();
  auto initArgs = forOp.getInitArgs();

  for (unsigned i = 0; i < regionIterArgs.size(); ++i) {
    Value iterArg = regionIterArgs[i];
    Value initArg = initArgs[i];
    Value yielded = oldYield.getOperand(i);

    auto iterTy = dyn_cast<RankedTensorType>(iterArg.getType());
    if (!iterTy || !iterTy.hasStaticShape()) {
      continue;
    }

    // if initArg comes from another iterArg of an outer loop, get the ultimate
    // source initArg. This handles the common nested relay pattern where inner
    // loop iter arg is directly yielded from outer loop iter arg without
    // modification. multiple for-loop levels of relay may require more complex
    // data flow analysis to resolve the ultimate source initArg.
    Value normalizedInitArg = normalizeInitArgForShapePeel(initArg);

    auto shapeInfoOpt = peelShapeChain(normalizedInitArg);
    if (!shapeInfoOpt.has_value()) {
      continue;
    }
    auto shapeInfo = *shapeInfoOpt;

    SmallVector<Operation *> chainOps;
    if (failed(collectReverseLinearYieldPath(yielded, iterArg, chainOps))) {
      continue;
    }

    std::optional<RelayMapM1> relayMap;
    if (auto outerFor = dyn_cast_or_null<scf::ForOp>(forOp->getParentOp())) {
      relayMap = getRelayMapM1(forOp, outerFor, i);
    }

    if (!relayMap.has_value() && isBlockArgumentFromAnotherForLoop(initArg)) {
      LLVM_DEBUG({
        llvm::dbgs()
            << "Init arg is a block argument from another for loop, but does "
               "not form a simple relay pattern. Skipping candidate.\n";
      });
      continue;
    }

    candidates.push_back(CandidateInfo{
        .idx = i,
        .shapeInfo = std::move(shapeInfo),
        .arithOps = std::move(chainOps),
        .relayMap = relayMap,
    });
  }

  if (candidates.empty()) {
    return failure();
  }

  // Pre-validate all candidate chains before mutating IR.
  // This avoids creating partial new IR and then returning failure().
  for (auto &c : candidates) {
    auto baseTensorTy = dyn_cast<RankedTensorType>(c.shapeInfo.base.getType());
    if (!baseTensorTy) {
      return failure();
    }

    Value iterCur = regionIterArgs[c.idx];
    for (Operation *oldOp : c.arithOps) {
      Value nextVal, oldConst;
      if (!isArithWithConst(oldOp, iterCur, nextVal, oldConst)) {
        return failure();
      }
      if (!canBuildConstLikeOperand(oldConst, baseTensorTy)) {
        return failure();
      }
      iterCur = nextVal;
    }
  }

  if (!isSafeToRewriteLanesByResultUses(forOp, candidates)) {
    return failure();
  }

  LLVM_DEBUG({
    llvm::dbgs() << "Now Simplify For Op: \n";
    forOp.dump();
    llvm::dbgs() << "Found " << candidates.size()
                 << " candidate iter args to simplify in for loop at "
                 << forOp.getLoc() << "\n";
    for (auto &c : candidates) {
      c.dump();
    }
  });

  SmallVector<CandidateInfo> localCandidates;
  SmallVector<CandidateInfo> relayCandidates;
  splitCandidatesByRelay(candidates, localCandidates, relayCandidates);

  FailureOr<scf::ForOp> newForRes = rewriteForWithLocalCandidates(
      forOp, localCandidates, /*outerCaptureMap=*/nullptr, rewriter);
  if (failed(newForRes)) {
    return failure();
  }
  auto newFor = *newForRes;

  // case A: has relay -> continue relay pipeline
  // If there are relay candidates, we need to rewrite the innerFor and outerFor
  // together to maintain the relay relationship. The innerFor needs to be
  // rewritten because the iterArg shape is changed after peeling, and the
  // outerFor needs to be rewritten together to maintain the relay relationship
  // between inner and outer iter args.
  if (!relayCandidates.empty()) {
    LLVM_DEBUG({
      llvm::dbgs() << "Nested For-loop args need to be rewritten to maintain "
                      "relay relationship\n";
    });
    scf::ForOp oldFor = forOp;
    if (failed(rewriteForWithRelayCandidates(newFor, oldFor, relayCandidates,
                                             rewriter))) {
      rewriter.eraseOp(newFor);
      return failure();
    }
    return success();
  }

  LLVM_DEBUG({
    llvm::dbgs() << "Only local candidates found, committing local rewrite\n";
  });
  // case B: no relay -> commit local rewrite now
  if (newFor->hasAttr(kIncompleteAttr)) {
    newFor->removeAttr(kIncompleteAttr);
  }
  newFor->setAttr(kSimplifiedAttr, rewriter.getUnitAttr());
  LLVM_DEBUG({
    llvm::dbgs() << "Successfully rewrote ForOp to: \n";
    newFor.dump();
  });
  rewriter.replaceOp(forOp, newFor.getResults());
  return success();
}

bool SimplifyTensorIterArgsPattern::extractBinaryArithOperands(
    Operation *op, Value &lhs, Value &rhs) const {
  if (auto v = dyn_cast<arith::AddIOp>(op)) {
    lhs = v.getLhs();
    rhs = v.getRhs();
    return true;
  }
  if (auto v = dyn_cast<arith::SubIOp>(op)) {
    lhs = v.getLhs();
    rhs = v.getRhs();
    return true;
  }
  if (auto v = dyn_cast<arith::MulIOp>(op)) {
    lhs = v.getLhs();
    rhs = v.getRhs();
    return true;
  }
  if (auto v = dyn_cast<arith::DivSIOp>(op)) {
    lhs = v.getLhs();
    rhs = v.getRhs();
    return true;
  }

  if (auto v = dyn_cast<arith::AddFOp>(op)) {
    lhs = v.getLhs();
    rhs = v.getRhs();
    return true;
  }
  if (auto v = dyn_cast<arith::SubFOp>(op)) {
    lhs = v.getLhs();
    rhs = v.getRhs();
    return true;
  }
  if (auto v = dyn_cast<arith::MulFOp>(op)) {
    lhs = v.getLhs();
    rhs = v.getRhs();
    return true;
  }
  if (auto v = dyn_cast<arith::DivFOp>(op)) {
    lhs = v.getLhs();
    rhs = v.getRhs();
    return true;
  }

  return false;
}

Value SimplifyTensorIterArgsPattern::createSameBinaryArithOp(
    Operation *oldOp, Location loc, Value lhs, Value rhs,
    PatternRewriter &rewriter) const {
  if (isa<arith::AddIOp>(oldOp))
    return rewriter.create<arith::AddIOp>(loc, lhs, rhs).getResult();
  if (isa<arith::SubIOp>(oldOp))
    return rewriter.create<arith::SubIOp>(loc, lhs, rhs).getResult();
  if (isa<arith::MulIOp>(oldOp))
    return rewriter.create<arith::MulIOp>(loc, lhs, rhs).getResult();
  if (isa<arith::DivSIOp>(oldOp))
    return rewriter.create<arith::DivSIOp>(loc, lhs, rhs).getResult();

  if (isa<arith::AddFOp>(oldOp))
    return rewriter.create<arith::AddFOp>(loc, lhs, rhs).getResult();
  if (isa<arith::SubFOp>(oldOp))
    return rewriter.create<arith::SubFOp>(loc, lhs, rhs).getResult();
  if (isa<arith::MulFOp>(oldOp))
    return rewriter.create<arith::MulFOp>(loc, lhs, rhs).getResult();
  if (isa<arith::DivFOp>(oldOp))
    return rewriter.create<arith::DivFOp>(loc, lhs, rhs).getResult();

  return Value();
}

FailureOr<scf::ForOp>
SimplifyTensorIterArgsPattern::rewriteForWithLocalCandidates(
    scf::ForOp forOp,
    ArrayRef<SimplifyTensorIterArgsPattern::CandidateInfo> candidates,
    const IRMapping *outerCaptureMap, PatternRewriter &rewriter) const {
  // This function is used to rewrite the forOp with local candidate.
  // The caller should have already validated that the candidate's shape chain
  // can be peeled and the arithmetic chain can be rebuilt with constants.
  if (candidates.empty()) {
    return forOp;
  }

  Block &oldBody = *forOp.getBody();
  if (!oldBody.mightHaveTerminator()) {
    return failure();
  }
  auto oldYield = dyn_cast<scf::YieldOp>(oldBody.getTerminator());
  if (!oldYield) {
    return failure();
  }

  auto regionIterArgs = forOp.getRegionIterArgs();
  auto initArgs = forOp.getInitArgs();

  DenseMap<unsigned, const CandidateInfo *> candMap;
  for (auto &c : candidates) {
    candMap[c.idx] = &c;
  }

  DenseMap<Operation *, unsigned> arithOpToCandIdx;
  DenseSet<Operation *> candidateArithSet;
  for (auto &c : candidates) {
    for (Operation *op : c.arithOps) {
      candidateArithSet.insert(op);
      arithOpToCandIdx[op] = c.idx;
    }
  }

  // Build new init args (candidate iter arg uses base)
  SmallVector<Value> newInitArgs;
  newInitArgs.reserve(initArgs.size());
  for (unsigned i = 0; i < initArgs.size(); ++i) {
    if (auto it = candMap.find(i); it != candMap.end()) {
      newInitArgs.push_back(it->second->shapeInfo.base);
      continue;
    }
    Value v = initArgs[i];
    if (outerCaptureMap) {
      if (Value mapped = outerCaptureMap->lookupOrNull(v)) {
        v = mapped;
      }
    }
    newInitArgs.push_back(v);
  }

  auto newFor = rewriter.create<scf::ForOp>(
      forOp.getLoc(), forOp.getLowerBound(), forOp.getUpperBound(),
      forOp.getStep(), newInitArgs);
  newFor->setAttr(kIncompleteAttr, rewriter.getUnitAttr());

  auto failAfterCreate = [&]() -> LogicalResult {
    rewriter.eraseOp(newFor);
    return failure();
  };

  IRMapping mapping;
  mapping.map(forOp.getInductionVar(), newFor.getInductionVar());

  // map region args
  for (unsigned i = 0; i < regionIterArgs.size(); ++i) {
    Value oldArg = regionIterArgs[i];
    Value newArg = newFor.getRegionIterArgs()[i];
    mapping.map(oldArg, newArg);
  }

  rewriter.setInsertionPointToStart(newFor.getBody());

  // Materialize candidate iter-shape values and remap old iterArg semantic
  // values.
  for (auto &c : candidates) {
    Value baseArg = newFor.getRegionIterArgs()[c.idx];
    Value mat =
        cloneShapeChain(forOp.getLoc(), baseArg, c.shapeInfo.chain, rewriter);
    if (!mat) {
      return failAfterCreate();
    }
    mapping.map(regionIterArgs[c.idx], mat);
  }

  // Track current base-domain running value for each candidate lane.
  DenseMap<unsigned, Value> baseCurByCand;
  for (auto &c : candidates) {
    baseCurByCand[c.idx] = newFor.getRegionIterArgs()[c.idx];
  }

  // Replay old body in-order:
  // - candidate arith op: rebuild now and map old result -> new result
  // - other op: clone with mapping
  for (Operation &op : oldBody.without_terminator()) {
    if (candidateArithSet.contains(&op)) {
      auto itIdx = arithOpToCandIdx.find(&op);
      if (itIdx == arithOpToCandIdx.end()) {
        return failAfterCreate();
      }
      unsigned candIdx = itIdx->second;

      Value baseCur = baseCurByCand[candIdx];
      auto baseTensorTy = dyn_cast<RankedTensorType>(baseCur.getType());
      if (!baseTensorTy) {
        return failAfterCreate();
      }

      Value lhs;
      Value rhs;
      if (!extractBinaryArithOperands(&op, lhs, rhs)) {
        return failAfterCreate();
      }

      bool lhsIsCst = matchPattern(lhs, m_Constant());
      bool rhsIsCst = matchPattern(rhs, m_Constant());
      if (lhsIsCst == rhsIsCst) {
        return failAfterCreate();
      }

      Value oldConst = lhsIsCst ? lhs : rhs;
      Value baseConst =
          getNewConstLikeOperand(oldConst, baseTensorTy, rewriter);
      if (!baseConst) {
        return failAfterCreate();
      }

      Value newRes = createSameBinaryArithOp(&op, op.getLoc(), baseCur,
                                             baseConst, rewriter);
      if (!newRes) {
        return failAfterCreate();
      }

      mapping.map(op.getResult(0), newRes);
      baseCurByCand[candIdx] = newRes;
      continue;
    }

    // Non-candidate op: clone as-is with mapping.
    if (outerCaptureMap) {
      for (Value operand : op.getOperands()) {
        if (mapping.lookupOrNull(operand)) {
          continue; // already mapped (iv/iterArg/local rebuilt values)
        }
        if (Value mappedOuter = outerCaptureMap->lookupOrNull(operand)) {
          mapping.map(operand, mappedOuter);
        }
      }
    }
    rewriter.clone(op, mapping);
  }

  // Rebuild yield.
  SmallVector<Value> newYieldOperands;
  newYieldOperands.reserve(oldYield.getNumOperands());
  for (unsigned i = 0; i < oldYield.getNumOperands(); ++i) {
    if (candMap.count(i)) {
      auto it = baseCurByCand.find(i);
      if (it == baseCurByCand.end() || !it->second) {
        return failAfterCreate();
      }
      newYieldOperands.push_back(it->second);
    } else {
      Value mapped = mapping.lookupOrDefault(oldYield.getOperand(i));
      if (!mapped) {
        return failAfterCreate();
      }
      newYieldOperands.push_back(mapped);
    }
  }
  rewriter.create<scf::YieldOp>(oldYield.getLoc(), newYieldOperands);
  if (newFor->hasAttr(kIncompleteAttr)) {
    newFor->removeAttr(kIncompleteAttr);
  }
  newFor->setAttr(kSimplifiedAttr, rewriter.getUnitAttr());
  return newFor;
}

bool SimplifyTensorIterArgsPattern::isSafeToRewriteLanesByResultUses(
    scf::ForOp forOp,
    ArrayRef<SimplifyTensorIterArgsPattern::CandidateInfo> candidates) const {
  if (candidates.empty()) {
    return false;
  }

  SmallVector<unsigned> laneIdxs;
  laneIdxs.reserve(candidates.size());
  for (const auto &c : candidates) {
    laneIdxs.push_back(c.idx);
  }

  auto parentOuter = dyn_cast_or_null<scf::ForOp>(forOp->getParentOp());
  if (!parentOuter || !parentOuter.getBody() ||
      !parentOuter.getBody()->mightHaveTerminator()) {
    // No parent loop or parent loop body is malformed, be conservative and
    // return false to avoid unsafe rewrite.
    return false;
  }
  auto parentOuterYield =
      parentOuter
          ? dyn_cast<scf::YieldOp>(parentOuter.getBody()->getTerminator())
          : nullptr;

  for (const auto &c : candidates) {
    unsigned idx = c.idx;
    if (idx >= forOp.getNumResults()) {
      return false;
    }

    Value r = forOp.getResult(idx);
    if (r.use_empty()) {
      continue;
    }

    // relay case: the rewritten lane can only be used by the corresponding
    // outer yield operand in the parent loop, and cannot have any other uses.
    if (c.relayMap.has_value()) {
      if (!parentOuter || !parentOuterYield) {
        return false;
      }
      unsigned outerYieldIdx = c.relayMap->outerYieldIdx;
      if (outerYieldIdx >= parentOuterYield.getNumOperands()) {
        return false;
      }

      for (OpOperand &use : r.getUses()) {
        auto y = dyn_cast<scf::YieldOp>(use.getOwner());
        if (!y || y != parentOuterYield) {
          return false;
        }
        if (use.getOperandNumber() != outerYieldIdx) {
          return false;
        }
      }

      // outer result cannot have any uses outside of the parent loop's yield
      // operand
      unsigned outerLane = c.relayMap->outerYieldIdx;
      if (outerLane >= parentOuter.getNumResults()) {
        return false;
      }
      if (!parentOuter.getResult(outerLane).use_empty()) {
        return false;
      }

      continue;
    }

    // local case: the rewritten lane cannot have any uses outside of the forOp
    // results
    return false;
  }

  return true;
}

LogicalResult SimplifyTensorIterArgsPattern::precheckRelayCandidates(
    scf::ForOp innerFor,
    ArrayRef<SimplifyTensorIterArgsPattern::CandidateInfo> relayCandidates,
    scf::ForOp &outerForOut) const {
  if (relayCandidates.empty()) {
    return failure();
  }

  auto outerFor = dyn_cast_or_null<scf::ForOp>(innerFor->getParentOp());
  if (!outerFor) {
    return failure();
  }

  DenseSet<unsigned> usedOuterInitIdx;
  DenseSet<unsigned> usedOuterYieldIdx;
  if (!outerFor.getBody() || !outerFor.getBody()->mightHaveTerminator()) {
    return failure();
  }
  auto outerYield = dyn_cast<scf::YieldOp>(outerFor.getBody()->getTerminator());
  if (!outerYield) {
    return failure();
  }

  for (const auto &c : relayCandidates) {
    if (!c.relayMap.has_value()) {
      return failure();
    }
    const auto &m = *c.relayMap;

    // must be the same level mapping for the current innerFor
    if (m.innerIdx != c.idx) {
      return failure();
    }

    // M1 strict lane
    if (m.outerInitIdx != m.outerYieldIdx) {
      return failure();
    }

    if (m.outerInitIdx >= outerFor.getInitArgs().size()) {
      return failure();
    }
    if (m.outerYieldIdx >= outerYield.getNumOperands()) {
      return failure();
    }

    // Conflict check: outer lane cannot be occupied multiple times
    // different inner iter args mapped to the same outer iter arg or yield
    // operand is not supported in this simple relay case, as it may require
    // more complex data flow analysis to resolve the ultimate source init arg
    // and final yield operand.
    if (!usedOuterInitIdx.insert(m.outerInitIdx).second) {
      return failure();
    }
    if (!usedOuterYieldIdx.insert(m.outerYieldIdx).second) {
      return failure();
    }
  }

  outerForOut = outerFor;
  return success();
}

FailureOr<scf::ForOp>
SimplifyTensorIterArgsPattern::rewriteInnerForWithRelayCandidates(
    scf::ForOp innerFor,
    ArrayRef<SimplifyTensorIterArgsPattern::CandidateInfo> relayCandidates,
    const IRMapping *outerCaptureMap, PatternRewriter &rewriter) const {
  if (!innerFor || relayCandidates.empty()) {
    return failure();
  }

  // Reuse local-like rebuild on the already-correct inner init args.
  // IMPORTANT: For relay mode, caller must ensure innerFor init args on relay
  // lanes are already wired to new outer iter args.
  return rewriteForWithLocalCandidates(innerFor, relayCandidates,
                                       outerCaptureMap, rewriter);
}

FailureOr<scf::ForOp>
SimplifyTensorIterArgsPattern::rewriteOuterForWithRelayCandidates(
    scf::ForOp innerFor, scf::ForOp oldInnerFor, scf::ForOp outerFor,
    ArrayRef<SimplifyTensorIterArgsPattern::CandidateInfo> relayCandidates,
    PatternRewriter &rewriter) const {
  if (!outerFor || !innerFor || relayCandidates.empty()) {
    return failure();
  }

  Block &oldOuterBody = *outerFor.getBody();
  if (!oldOuterBody.mightHaveTerminator()) {
    return failure();
  }
  auto oldOuterYield = dyn_cast<scf::YieldOp>(oldOuterBody.getTerminator());
  if (!oldOuterYield) {
    return failure();
  }

  // Build new outer init args: relay lanes switch to base.
  SmallVector<Value> newOuterInitArgs(outerFor.getInitArgs().begin(),
                                      outerFor.getInitArgs().end());
  for (const auto &c : relayCandidates) {
    if (!c.relayMap.has_value()) {
      return failure();
    }
    const auto &m = *c.relayMap;
    if (m.outerInitIdx >= newOuterInitArgs.size()) {
      return failure();
    }
    newOuterInitArgs[m.outerInitIdx] = c.shapeInfo.base;
  }

  rewriter.setInsertionPoint(outerFor);
  auto newOuterFor = rewriter.create<scf::ForOp>(
      outerFor.getLoc(), outerFor.getLowerBound(), outerFor.getUpperBound(),
      outerFor.getStep(), newOuterInitArgs);
  newOuterFor->setAttr(kIncompleteAttr, rewriter.getUnitAttr());

  auto failAfterCreate = [&]() -> FailureOr<scf::ForOp> {
    rewriter.eraseOp(newOuterFor);
    return failure();
  };

  IRMapping outerMap;
  outerMap.map(outerFor.getInductionVar(), newOuterFor.getInductionVar());
  for (unsigned i = 0; i < outerFor.getRegionIterArgs().size(); ++i) {
    outerMap.map(outerFor.getRegionIterArgs()[i],
                 newOuterFor.getRegionIterArgs()[i]);
  }

  rewriter.setInsertionPointToStart(newOuterFor.getBody());

  scf::ForOp rewrittenInnerInNewOuter;

  // Clone outer body with special handling for old inner.
  for (Operation &op : oldOuterBody.without_terminator()) {
    // Fast path: not the anchor inner op we want to rebuild.
    if (&op != innerFor.getOperation()) {
      if (&op == oldInnerFor.getOperation()) {
        // Drop old inner only when it is different from anchor inner.
        continue;
      }
      rewriter.clone(op, outerMap);
      continue;
    }

    // Build relay-wired inner init args:
    // relay lane: inner init <- new outer iterArg[outerInitIdx]
    // non-relay lane: mapped old inner init
    SmallVector<Value> relayWiredInnerInitArgs;
    relayWiredInnerInitArgs.reserve(innerFor.getInitArgs().size());

    DenseMap<unsigned, unsigned> innerIdxToOuterInitIdx;
    for (const auto &c : relayCandidates) {
      const auto &m = *c.relayMap;
      innerIdxToOuterInitIdx[m.innerIdx] = m.outerInitIdx;
    }

    for (unsigned i = 0; i < innerFor.getInitArgs().size(); ++i) {
      auto it = innerIdxToOuterInitIdx.find(i);
      if (it != innerIdxToOuterInitIdx.end()) {
        unsigned outerInitIdx = it->second;
        if (outerInitIdx >= newOuterFor.getRegionIterArgs().size()) {
          return failAfterCreate();
        }
        relayWiredInnerInitArgs.push_back(
            newOuterFor.getRegionIterArgs()[outerInitIdx]);
      } else {
        relayWiredInnerInitArgs.push_back(
            outerMap.lookupOrDefault(innerFor.getInitArgs()[i]));
      }
    }

    // Build inner relay candidates whose base comes from new outer iter args.
    SmallVector<CandidateInfo> relayCandidatesForInner;
    relayCandidatesForInner.reserve(relayCandidates.size());
    for (const auto &c : relayCandidates) {
      CandidateInfo cc = c; // copy

      if (!cc.relayMap.has_value()) {
        return failAfterCreate();
      }
      unsigned outerInitIdx = cc.relayMap->outerInitIdx;
      if (outerInitIdx >= newOuterFor.getRegionIterArgs().size()) {
        return failAfterCreate();
      }

      // relay semantics: inner base is new outer iter arg on mapped lane
      cc.shapeInfo.base = newOuterFor.getRegionIterArgs()[outerInitIdx];
      relayCandidatesForInner.push_back(std::move(cc));
    }

    FailureOr<scf::ForOp> rewrittenInnerRes = failure();
    {
      PatternRewriter::InsertionGuard guard(rewriter);
      rewrittenInnerRes = rewriteInnerForWithRelayCandidates(
          innerFor, relayCandidatesForInner, &outerMap, rewriter);
    }
    if (failed(rewrittenInnerRes)) {
      return failAfterCreate();
    }
    rewrittenInnerInNewOuter = *rewrittenInnerRes;

    // Map old inner results to rewritten inner results for outer cloning/yield
    // mapping.
    if (innerFor.getNumResults() != rewrittenInnerInNewOuter.getNumResults()) {
      return failAfterCreate();
    }
    for (unsigned r = 0; r < innerFor.getNumResults(); ++r) {
      outerMap.map(innerFor.getResult(r),
                   rewrittenInnerInNewOuter.getResult(r));
    }
  }

  if (!rewrittenInnerInNewOuter) {
    return failAfterCreate();
  }

  // Rebuild outer yield; relay lanes forced from rewritten inner results.
  SmallVector<Value> newOuterYieldOps;
  newOuterYieldOps.reserve(oldOuterYield.getNumOperands());
  DenseMap<unsigned, unsigned> outerYieldToInnerIdx;
  for (const auto &c : relayCandidates) {
    const auto &m = *c.relayMap;
    outerYieldToInnerIdx[m.outerYieldIdx] = m.innerIdx;
  }

  for (unsigned i = 0; i < oldOuterYield.getNumOperands(); ++i) {
    auto it = outerYieldToInnerIdx.find(i);
    if (it != outerYieldToInnerIdx.end()) {
      unsigned innerIdx = it->second;
      if (innerIdx >= rewrittenInnerInNewOuter.getNumResults()) {
        return failAfterCreate();
      }
      newOuterYieldOps.push_back(rewrittenInnerInNewOuter.getResult(innerIdx));
    } else {
      newOuterYieldOps.push_back(
          outerMap.lookupOrDefault(oldOuterYield.getOperand(i)));
    }
  }

  rewriter.setInsertionPointToEnd(newOuterFor.getBody());
  rewriter.create<scf::YieldOp>(oldOuterYield.getLoc(), newOuterYieldOps);

  if (newOuterFor->hasAttr(kIncompleteAttr)) {
    newOuterFor->removeAttr(kIncompleteAttr);
  }
  newOuterFor->setAttr(kSimplifiedAttr, rewriter.getUnitAttr());
  return newOuterFor;
}

LogicalResult SimplifyTensorIterArgsPattern::rewriteForWithRelayCandidates(
    scf::ForOp newfor, scf::ForOp oldFor,
    ArrayRef<CandidateInfo> relayCandidates, PatternRewriter &rewriter) const {
  // forOp is innerFor (already local-rewritten or to-be-relay-rewritten)
  scf::ForOp innerFor = newfor;
  scf::ForOp outerFor;
  if (failed(precheckRelayCandidates(innerFor, relayCandidates, outerFor))) {
    return failure();
  }

  FailureOr<scf::ForOp> newOuterRes = rewriteOuterForWithRelayCandidates(
      innerFor, oldFor, outerFor, relayCandidates, rewriter);
  if (failed(newOuterRes)) {
    return failure();
  }

  scf::ForOp newOuterFor = *newOuterRes;
  LLVM_DEBUG({
    llvm::dbgs() << "Successfully rewrote outer ForOp to: \n";
    newOuterFor.dump();
  });

  // Only replace outer; old inner is nested under old outer and will be removed
  // together.
  rewriter.replaceOp(outerFor, newOuterFor.getResults());
  return success();
}

bool IfYieldAddHoistConverter::isSupportedTensorResultType(Type type) const {
  auto tensorType = dyn_cast<RankedTensorType>(type);
  return tensorType && !isa<triton::PointerType>(tensorType.getElementType());
}

bool IfYieldAddHoistConverter::isDefinedOutsideIf(Value value,
                                                  scf::IfOp ifOp) const {
  if (auto blockArg = dyn_cast<BlockArgument>(value)) {
    Block *owner = blockArg.getOwner();
    return owner && owner->getParentOp() != ifOp.getOperation();
  }

  Operation *defOp = value.getDefiningOp();
  return defOp && !ifOp->isAncestor(defOp);
}

bool IfYieldAddHoistConverter::extractAddendFromAddExpr(
    Value maybeAddExpr, Value baseValue, Value &addendOut) const {
  if (auto addi = maybeAddExpr.getDefiningOp<arith::AddIOp>()) {
    if (addi.getLhs() == baseValue) {
      addendOut = addi.getRhs();
      return true;
    }
    if (addi.getRhs() == baseValue) {
      addendOut = addi.getLhs();
      return true;
    }
    return false;
  }

  if (auto addf = maybeAddExpr.getDefiningOp<arith::AddFOp>()) {
    if (addf.getLhs() == baseValue) {
      addendOut = addf.getRhs();
      return true;
    }
    if (addf.getRhs() == baseValue) {
      addendOut = addf.getLhs();
      return true;
    }
    return false;
  }

  return false;
}

Value IfYieldAddHoistConverter::buildZeroTensorLikeType(
    Type laneType, Location loc, PatternRewriter &rewriter) const {
  auto tensorType = dyn_cast<RankedTensorType>(laneType);
  if (!tensorType)
    return Value();

  Type elemType = tensorType.getElementType();
  Attribute zeroElemAttr;
  if (isa<FloatType>(elemType)) {
    zeroElemAttr = rewriter.getFloatAttr(elemType, 0.0);
  } else if (elemType.isIntOrIndex()) {
    zeroElemAttr = rewriter.getIntegerAttr(elemType, 0);
  } else {
    return Value();
  }

  auto zeroTensorAttr = DenseElementsAttr::get(tensorType, zeroElemAttr);
  return rewriter.create<arith::ConstantOp>(loc, laneType, zeroTensorAttr);
}

bool IfYieldAddHoistConverter::tryRewriteSingleLane(
    unsigned laneIdx, Value baseBranchYield, Value addExprBranchYield,
    bool baseInThenBranch, Type laneType, scf::IfOp ifOp,
    PatternRewriter &rewriter, SmallVectorImpl<Value> &updatedThenYieldOperands,
    SmallVectorImpl<Value> &updatedElseYieldOperands,
    SmallVectorImpl<Value> &hoistedBasePerLane,
    SmallVectorImpl<bool> &laneRewrittenFlags) const {
  if (!isDefinedOutsideIf(baseBranchYield, ifOp))
    return false;

  Value addendValue;
  if (!extractAddendFromAddExpr(addExprBranchYield, baseBranchYield,
                                addendValue))
    return false;
  if (!addendValue || addendValue.getType() != laneType)
    return false;

  Value zeroTensor = buildZeroTensorLikeType(laneType, ifOp.getLoc(), rewriter);
  if (!zeroTensor)
    return false;

  if (baseInThenBranch) {
    updatedThenYieldOperands[laneIdx] = zeroTensor;
    updatedElseYieldOperands[laneIdx] = addendValue;
  } else {
    updatedThenYieldOperands[laneIdx] = addendValue;
    updatedElseYieldOperands[laneIdx] = zeroTensor;
  }

  hoistedBasePerLane[laneIdx] = baseBranchYield;
  laneRewrittenFlags[laneIdx] = true;
  return true;
}

// scf.if lane rewrite:
// one branch yields A, the other yields A + B
// => if yields 0 / B, then outside do A + if_result
LogicalResult
IfYieldAddHoistConverter::matchAndRewrite(scf::IfOp ifOp,
                                          PatternRewriter &rewriter) const {
  if (ifOp.getNumResults() == 0) {
    return failure();
  }

  // Robust guard for malformed / emptied regions.
  Block *thenBlockPtr = ifOp.thenBlock();
  Block *elseBlockPtr = ifOp.elseBlock();
  if (!thenBlockPtr || !elseBlockPtr) {
    return failure();
  }
  if (!thenBlockPtr->mightHaveTerminator() ||
      !elseBlockPtr->mightHaveTerminator()) {
    return failure();
  }

  auto thenYieldOp = dyn_cast<scf::YieldOp>(ifOp.thenBlock()->getTerminator());
  auto elseYieldOp = dyn_cast<scf::YieldOp>(ifOp.elseBlock()->getTerminator());
  if (!thenYieldOp || !elseYieldOp) {
    return failure();
  }

  Location loc = ifOp.getLoc();
  bool anyLaneRewritten = false;

  SmallVector<Value> updatedThenYieldOperands(thenYieldOp.getOperands().begin(),
                                              thenYieldOp.getOperands().end());
  SmallVector<Value> updatedElseYieldOperands(elseYieldOp.getOperands().begin(),
                                              elseYieldOp.getOperands().end());

  // for each lane, record whether we need post-if add with baseYield
  SmallVector<Value> hoistedBasePerLane(ifOp.getNumResults(), Value());
  SmallVector<bool> laneRewrittenFlags(ifOp.getNumResults(), false);

  for (unsigned laneIdx = 0; laneIdx < ifOp.getNumResults(); ++laneIdx) {
    Type laneType = ifOp.getResultTypes()[laneIdx];
    if (!isSupportedTensorResultType(laneType)) {
      continue;
    }

    Value thenYieldVal = thenYieldOp.getOperand(laneIdx);
    Value elseYieldVal = elseYieldOp.getOperand(laneIdx);

    // Assume that only one of the two branches can have the add pattern, and
    // try both ways to find a rewrite opportunity.
    bool rewritten =
        tryRewriteSingleLane(
            laneIdx, thenYieldVal, elseYieldVal, /* baseInThenBranch= */ true,
            laneType, ifOp, rewriter, updatedThenYieldOperands,
            updatedElseYieldOperands, hoistedBasePerLane, laneRewrittenFlags) ||
        tryRewriteSingleLane(
            laneIdx, elseYieldVal, thenYieldVal, /* baseInThenBranch= */ false,
            laneType, ifOp, rewriter, updatedThenYieldOperands,
            updatedElseYieldOperands, hoistedBasePerLane, laneRewrittenFlags);

    anyLaneRewritten |= rewritten;
  }

  if (!anyLaneRewritten) {
    return failure();
  }

  rewriter.setInsertionPoint(ifOp);
  auto rewrittenIfOp = rewriter.create<scf::IfOp>(loc, ifOp.getResultTypes(),
                                                  ifOp.getCondition(),
                                                  /* withElseRegion= */ true);

  {
    IRMapping thenMapping;
    Block *oldThenBlock = ifOp.thenBlock();
    Block *newThenBlock = rewrittenIfOp.thenBlock();

    rewriter.setInsertionPointToStart(newThenBlock);
    for (Operation &op : oldThenBlock->without_terminator()) {
      rewriter.clone(op, thenMapping);
    }

    SmallVector<Value> mappedThenYieldOperands;
    mappedThenYieldOperands.reserve(updatedThenYieldOperands.size());
    for (Value v : updatedThenYieldOperands) {
      mappedThenYieldOperands.push_back(thenMapping.lookupOrDefault(v));
    }

    rewriter.create<scf::YieldOp>(loc, mappedThenYieldOperands);
  }

  {
    IRMapping elseMapping;
    Block *oldElseBlock = ifOp.elseBlock();
    Block *newElseBlock = rewrittenIfOp.elseBlock();

    rewriter.setInsertionPointToStart(newElseBlock);
    for (Operation &op : oldElseBlock->without_terminator()) {
      rewriter.clone(op, elseMapping);
    }

    SmallVector<Value> mappedElseYieldOperands;
    mappedElseYieldOperands.reserve(updatedElseYieldOperands.size());
    for (Value v : updatedElseYieldOperands) {
      mappedElseYieldOperands.push_back(elseMapping.lookupOrDefault(v));
    }

    rewriter.create<scf::YieldOp>(loc, mappedElseYieldOperands);
  }

  rewriter.setInsertionPointAfter(rewrittenIfOp);
  SmallVector<Value> finalResults;
  finalResults.reserve(ifOp.getNumResults());

  for (unsigned laneIdx = 0; laneIdx < ifOp.getNumResults(); ++laneIdx) {
    if (!laneRewrittenFlags[laneIdx]) {
      finalResults.push_back(rewrittenIfOp.getResult(laneIdx));
      continue;
    }

    Value hoistedBase = hoistedBasePerLane[laneIdx];
    Value laneDelta = rewrittenIfOp.getResult(laneIdx);

    auto laneTensorType = dyn_cast<RankedTensorType>(laneDelta.getType());
    if (!laneTensorType)
      return failure();

    Type elemType = laneTensorType.getElementType();
    Value reconstructedLane;
    if (isa<FloatType>(elemType)) {
      reconstructedLane =
          rewriter.create<arith::AddFOp>(loc, hoistedBase, laneDelta);
    } else if (elemType.isIntOrIndex()) {
      reconstructedLane =
          rewriter.create<arith::AddIOp>(loc, hoistedBase, laneDelta);
    } else {
      return failure();
    }

    finalResults.push_back(reconstructedLane);
  }

  rewriter.replaceOp(ifOp, finalResults);
  return success();
}

} // namespace CannonicalizerConverter
