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

#include "ascend/include/Utils/InterleaveOptimization.h"
#include "ascend/include/Utils/Utils.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Utils/StaticValueUtils.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypeInterfaces.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/Interfaces/ViewLikeInterface.h"
#include "mlir/Support/LogicalResult.h"

#include "mlir/IR/Operation.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/Debug.h"
#include <cassert>
#include <limits>
#include <optional>
#include <utility>

namespace mlir {
namespace triton {
// For origin MemRefType of ReinterpretCastOp under interleave state, here wanna
// adjust its shape info by expanding last dimension double.
static MemRefType expandInterleaveMemRefType(
    MemRefType originType,
    std::optional<OpFoldResult> castOffset = std::nullopt) {
  // Double the last dimension shape
  SmallVector<int64_t> shape(originType.getShape());
  shape.back() = shape.back() * 2;

  // Adjuest layout attribute
  StridedLayoutAttr originLayout =
      llvm::dyn_cast<StridedLayoutAttr>(originType.getLayout());
  // If offset is static, just reset it to 0.
  auto offset = originLayout.getOffset() == ShapedType::kDynamic
                    ? originLayout.getOffset()
                    : 0;
  // A reinterpret_cast result type must agree with its offset operand.  The
  // original interleaved view can carry layout offset 0 while the normalized
  // base is a non-zero static offset, so use the new cast offset whenever it
  // is available.
  if (castOffset) {
    if (auto staticOffset = getConstantIntValue(*castOffset))
      offset = staticOffset.value();
    else
      offset = ShapedType::kDynamic;
  }
  // Set last dimension stride to 1
  SmallVector<int64_t> stride(originLayout.getStrides());
  stride.back() = 1;

  return MemRefType::get(
      shape, originType.getElementType(),
      StridedLayoutAttr::get(originType.getContext(), offset, stride));
}

// The last dimension of an interleaved view has stride 2.  The address can be
// expressed as an arbitrary base plus a lane selector (0 for even, 1 for odd):
//
//   base + 2 * lane + {0, 1}
//
// The base is allowed to contain static and dynamic offsets.  In particular,
// a constant such as the RoPE region offset (+64) is part of the base, not an
// odd-lane marker.  Keep the analysis side-effect free so callers can safely
// return failure and use the normal lowering path.
struct DeinterleaveOffsetInfo {
  SmallVector<Value> baseTerms;
  int64_t baseConstant = 0;
  IndexMode indexMode = IndexMode::EVEN_MODE;
  // Reuse the original SSA base whenever the outermost +1 unambiguously
  // selects the odd lane. This preserves the existing offset def-use chain
  // instead of recreating an equivalent arithmetic expression during type
  // conversion.
  std::optional<OpFoldResult> materializedBaseOffset;
};

static bool addStaticOffset(int64_t offset, int64_t &sum) {
  if ((offset > 0 && sum > std::numeric_limits<int64_t>::max() - offset) ||
      (offset < 0 && sum < std::numeric_limits<int64_t>::min() - offset))
    return false;
  sum += offset;
  return true;
}

static bool collectDeinterleaveOffsetTerms(OpFoldResult offset,
                                           SmallVectorImpl<Value> &baseTerms,
                                           int64_t &staticOffset) {
  if (auto constant = getConstantIntValue(offset))
    return addStaticOffset(constant.value(), staticOffset);

  auto value = llvm::dyn_cast<Value>(offset);
  if (!value)
    return false;

  if (auto addOp = value.getDefiningOp<arith::AddIOp>()) {
    return collectDeinterleaveOffsetTerms(addOp.getLhs(), baseTerms,
                                          staticOffset) &&
           collectDeinterleaveOffsetTerms(addOp.getRhs(), baseTerms,
                                          staticOffset);
  }

  // The expression below this value is an opaque base term.  Do not recurse
  // into it: a constant in a multiply/addptr/base-address subgraph is not a
  // lane selector.
  baseTerms.push_back(value);
  return true;
}

static FailureOr<DeinterleaveOffsetInfo>
normalizeDeinterleaveOffset(OpFoldResult originOffset) {
  DeinterleaveOffsetInfo result;
  int64_t staticOffset = 0;
  if (!collectDeinterleaveOffsetTerms(originOffset, result.baseTerms,
                                      staticOffset))
    return failure();

  for (Value term : result.baseTerms) {
    if (!isa<IndexType>(term.getType()))
      return failure();
  }

  // Keep the base pair-aligned relative to the original view.  This also
  // supports folded constants, e.g. +65 becomes base +64 and ODD_MODE.
  int64_t laneOffset = staticOffset % 2;
  if (laneOffset < 0)
    laneOffset += 2;
  result.baseConstant = staticOffset - laneOffset;
  result.indexMode =
      laneOffset == 0 ? IndexMode::EVEN_MODE : IndexMode::ODD_MODE;

  if (auto value = llvm::dyn_cast<Value>(originOffset)) {
    Value base = value;
    if (result.indexMode == IndexMode::ODD_MODE) {
      auto addOp = value.getDefiningOp<arith::AddIOp>();
      if (addOp) {
        if (auto lhsConstant = getConstantIntValue(addOp.getLhs());
            lhsConstant && lhsConstant.value() == 1) {
          base = addOp.getRhs();
          result.materializedBaseOffset = base;
        } else if (auto rhsConstant = getConstantIntValue(addOp.getRhs());
                   rhsConstant && rhsConstant.value() == 1) {
          base = addOp.getLhs();
          result.materializedBaseOffset = base;
        }
      }
    } else {
      result.materializedBaseOffset = base;
    }
  }
  return result;
}

static bool hasEquivalentDeinterleaveBase(const DeinterleaveOffsetInfo &lhs,
                                          const DeinterleaveOffsetInfo &rhs) {
  if (lhs.baseConstant != rhs.baseConstant ||
      lhs.baseTerms.size() != rhs.baseTerms.size())
    return false;

  // Addition is commutative.  Compare the opaque terms as a multiset so that
  // equivalent add trees with a different operand order remain optimizable.
  SmallVector<bool> matched(rhs.baseTerms.size(), false);
  for (Value lhsTerm : lhs.baseTerms) {
    bool found = false;
    for (auto [index, rhsTerm] : llvm::enumerate(rhs.baseTerms)) {
      if (!matched[index] && lhsTerm == rhsTerm) {
        matched[index] = true;
        found = true;
        break;
      }
    }
    if (!found)
      return false;
  }
  return true;
}

static OpFoldResult
materializeDeinterleaveBaseOffset(const DeinterleaveOffsetInfo &offsetInfo,
                                  OpBuilder &builder, Location loc) {
  if (offsetInfo.materializedBaseOffset)
    return *offsetInfo.materializedBaseOffset;
  if (offsetInfo.baseTerms.empty())
    return builder.getIndexAttr(offsetInfo.baseConstant);

  Value base = offsetInfo.baseTerms.front();
  for (Value term : llvm::drop_begin(offsetInfo.baseTerms))
    base = builder.create<arith::AddIOp>(loc, base, term);

  if (offsetInfo.baseConstant != 0) {
    Value constant = builder.create<arith::ConstantOp>(
        loc, builder.getIndexAttr(offsetInfo.baseConstant));
    base = builder.create<arith::AddIOp>(loc, base, constant);
  }
  return base;
}

LogicalResult
DeinterleaveStatusOptimization(triton::LoadOp op,
                               triton::LoadOp::Adaptor adaptor,
                               ConversionPatternRewriter &rewriter) {
  auto ptr = adaptor.getPtr();
  if (auto reinterpretCast = ptr.getDefiningOp<memref::ReinterpretCastOp>()) {
    auto loc = op.getLoc();

    // 1. Normalize the original view offset, then create the expanded cast.
    auto originCastOffset = reinterpretCast.getConstifiedMixedOffset();
    auto castSize = reinterpretCast.getConstifiedMixedSizes();
    auto castStride = reinterpretCast.getConstifiedMixedStrides();
    // Actually, `castSize` is always constant value as `MemRefType` result
    if (auto lastDimSize = getConstantIntValue(castSize.back())) {
      castSize.back() = rewriter.getIndexAttr(lastDimSize.value() * 2);
    } else {
      return failure();
    }
    // Last element of castStride is also constant value as prerequisite
    // is that last dimension stride of casted memref type is always 2.
    castStride.back() = rewriter.getIndexAttr(1);
    auto offsetInfo = normalizeDeinterleaveOffset(originCastOffset);
    if (failed(offsetInfo))
      return failure();
    OpFoldResult castOffset =
        materializeDeinterleaveBaseOffset(*offsetInfo, rewriter, loc);
    auto srcType =
        expandInterleaveMemRefType(reinterpretCast.getType(), castOffset);
    auto newCastOp = rewriter.create<memref::ReinterpretCastOp>(
        loc, srcType, reinterpretCast.getViewSource(), castOffset, castSize,
        castStride);

    // 3. Create new memref allocOp
    auto newAllocOp = rewriter.create<memref::AllocOp>(
        loc, MemRefType::get(srcType.getShape(), srcType.getElementType()));

    // 4. Implement memref copy and bufferization back to tensor
    rewriter.create<memref::CopyOp>(loc, newCastOp.getResult(), newAllocOp);
    Value newTensor = rewriter.create<bufferization::ToTensorOp>(
        loc,
        RankedTensorType::get(srcType.getShape(), srcType.getElementType()),
        newAllocOp, true /* restrict */, true /* writable */);

    // 5. Implement tensor extract_slice to represent deinterleave
    // Here use `castOffset` to determine whether even index deinterleave or
    // odd index.
    SmallVector<OpFoldResult> extractOffsets(srcType.getRank(),
                                             rewriter.getIndexAttr(0));
    SmallVector<OpFoldResult> extractStrides(srcType.getRank(),
                                             rewriter.getIndexAttr(1));
    SmallVector<OpFoldResult> extractSizes = llvm::to_vector(
        llvm::map_range(srcType.getShape(), [&](int64_t dim) -> OpFoldResult {
          return rewriter.getIndexAttr(dim);
        }));

    // Adjust extract_slice shape
    switch (offsetInfo->indexMode) {
    case IndexMode::EVEN_MODE:
      extractOffsets.back() = rewriter.getIndexAttr(0);
      break;
    case IndexMode::ODD_MODE:
      extractOffsets.back() = rewriter.getIndexAttr(1);
      break;
    }
    extractStrides.back() = rewriter.getIndexAttr(2);
    extractSizes.back() = rewriter.getIndexAttr(srcType.getShape().back() / 2);

    Value deinterleaveSlice = rewriter.create<tensor::ExtractSliceOp>(
        loc, newTensor, extractOffsets, extractSizes, extractStrides);

    rewriter.replaceOp(op, deinterleaveSlice);
    return success();
  }

  return failure();
}

LogicalResult DeinterleaveStatusWithMaskOptimization(
    triton::LoadOp op, triton::LoadOp::Adaptor adaptor,
    ConversionPatternRewriter &rewriter, MaskState &mstate, Value localMem) {
  auto ptr = adaptor.getPtr();
  if (auto reinterpretCast = ptr.getDefiningOp<memref::ReinterpretCastOp>()) {
    auto loc = op.getLoc();

    // 1. Normalize the original view offset, then create the expanded cast.
    auto originCastOffset = reinterpretCast.getConstifiedMixedOffset();
    auto castSize = reinterpretCast.getConstifiedMixedSizes();
    auto castStride = reinterpretCast.getConstifiedMixedStrides();

    if (auto lastDimSize = getConstantIntValue(castSize.back())) {
      castSize.back() = rewriter.getIndexAttr(lastDimSize.value() * 2);
    } else {
      return failure();
    }
    castStride.back() = rewriter.getIndexAttr(1);
    auto offsetInfo = normalizeDeinterleaveOffset(originCastOffset);
    if (failed(offsetInfo))
      return failure();
    OpFoldResult castOffset =
        materializeDeinterleaveBaseOffset(*offsetInfo, rewriter, loc);
    auto srcType =
        expandInterleaveMemRefType(reinterpretCast.getType(), castOffset);

    auto newCastOp = rewriter.create<memref::ReinterpretCastOp>(
        loc, srcType, reinterpretCast.getViewSource(), castOffset, castSize,
        castStride);

    // 3. Create new memref allocOp
    // To reuse existing linalg::fill, here need to change insertion point
    auto savedInsertPoint = rewriter.saveInsertionPoint();
    rewriter.setInsertionPointAfterValue(localMem);
    auto newAllocOp = rewriter.create<memref::AllocOp>(
        loc, MemRefType::get(srcType.getShape(), srcType.getElementType()));
    rewriter.restoreInsertionPoint(savedInsertPoint);

    // 4. Broadcast other value by linalg.fill if necessary
    auto other = op.getOther();
    // While deinterleave optimization will just adjust last dimension info
    // and origin mask state wouldn't involve last dimension. Therefore in
    // current `scf.if + linalg.fill` combination, condition of `if` could be
    // kept and just replace linalg.fill'
    if (other) {
      assert(localMem.hasOneUse() &&
             llvm::isa<linalg::FillOp>(*(localMem.getUsers().begin())));
      auto originFillOp =
          llvm::dyn_cast<linalg::FillOp>(*(localMem.getUsers().begin()));

      assert(llvm::isa<scf::IfOp>(originFillOp->getParentOp()));
      auto ifOp = llvm::dyn_cast<scf::IfOp>(originFillOp->getParentOp());

      auto newFillOp = ifOp.getThenBodyBuilder().create<linalg::FillOp>(
          originFillOp.getLoc(), originFillOp.getInputs(),
          ValueRange{newAllocOp});
      rewriter.replaceOp(originFillOp, newFillOp);
    }

    // 5. Implement new subview, memref copy and bufferization back to tensor
    SmallVector<OpFoldResult> subviewStrides(srcType.getRank(),
                                             rewriter.getIndexAttr(1));
    SmallVector<OpFoldResult> subviewOffsets = mstate.offsets;
    SmallVector<OpFoldResult> subviewSizes = mstate.dims;
    // Just adjust last dimension size to double
    std::optional<int64_t> originSubviewLastDim =
        getConstantIntValue(subviewSizes.back());
    assert(originSubviewLastDim.has_value());
    subviewSizes.back() =
        rewriter.getIndexAttr(originSubviewLastDim.value() * 2);

    auto argSubviewType = memref::SubViewOp::inferResultType(
        srcType, subviewOffsets, subviewSizes, subviewStrides);
    // alloca subview type doesn't carry layout attribute
    auto allocSubviewType = memref::SubViewOp::inferResultType(
        newAllocOp.getType(), subviewOffsets, subviewSizes, subviewStrides);

    memref::SubViewOp srcSubview = rewriter.create<memref::SubViewOp>(
        loc, llvm::cast<MemRefType>(argSubviewType), newCastOp, subviewOffsets,
        subviewSizes, subviewStrides);
    memref::SubViewOp dstSubview = rewriter.create<memref::SubViewOp>(
        loc, llvm::cast<MemRefType>(allocSubviewType), newAllocOp,
        subviewOffsets, subviewSizes, subviewStrides);
    rewriter.create<memref::CopyOp>(loc, srcSubview, dstSubview);
    Value newTensor = rewriter.create<bufferization::ToTensorOp>(
        loc,
        RankedTensorType::get(srcType.getShape(), srcType.getElementType()),
        newAllocOp, true /* restrict */, true /* writable */);

    // 6. Implement tensor extract_slice to represent deinterleave
    // Here use `castOffset` to determine whether even index deinterleave or
    // odd index.
    SmallVector<OpFoldResult> extractOffsets(srcType.getRank(),
                                             rewriter.getIndexAttr(0));
    SmallVector<OpFoldResult> extractStrides(srcType.getRank(),
                                             rewriter.getIndexAttr(1));
    SmallVector<OpFoldResult> extractSizes = llvm::to_vector(
        llvm::map_range(srcType.getShape(), [&](int64_t dim) -> OpFoldResult {
          return rewriter.getIndexAttr(dim);
        }));

    switch (offsetInfo->indexMode) {
    case IndexMode::EVEN_MODE:
      extractOffsets.back() = rewriter.getIndexAttr(0);
      break;
    case IndexMode::ODD_MODE:
      extractOffsets.back() = rewriter.getIndexAttr(1);
      break;
    }
    extractStrides.back() = rewriter.getIndexAttr(2);
    extractSizes.back() = rewriter.getIndexAttr(srcType.getShape().back() / 2);

    Value deinterleaveSlice = rewriter.create<tensor::ExtractSliceOp>(
        loc, newTensor, extractOffsets, extractSizes, extractStrides);

    rewriter.replaceOp(op, deinterleaveSlice);
    return success();
  }
  return failure();
}

LogicalResult
InterleaveStatusOptimization(SmallVector<Operation *> materializeVec) {
  OpBuilder builder(materializeVec[1]);
  auto loc = materializeVec[1]->getLoc();

  auto firstReinterpretCastOp =
      llvm::dyn_cast<bufferization::MaterializeInDestinationOp>(
          materializeVec[0])
          .getDest()
          .getDefiningOp<memref::ReinterpretCastOp>();
  auto secondReinterpretCastOp =
      llvm::dyn_cast<bufferization::MaterializeInDestinationOp>(
          materializeVec[1])
          .getDest()
          .getDefiningOp<memref::ReinterpretCastOp>();

  assert(firstReinterpretCastOp && secondReinterpretCastOp);
  // Judge whether two `ReinterpretCastOp` shape satisfy interleave state
  // a. both size are equal
  if (!isEqualConstantIntOrValueArray(
          firstReinterpretCastOp.getConstifiedMixedSizes(),
          secondReinterpretCastOp.getConstifiedMixedSizes())) {
    return failure();
  }
  // b. both strides are equal
  if (!isEqualConstantIntOrValueArray(
          firstReinterpretCastOp.getConstifiedMixedStrides(),
          secondReinterpretCastOp.getConstifiedMixedStrides())) {
    return failure();
  }
  // c. both offsets should satisfy tricky rule
  auto firstOriginCastOffset =
      firstReinterpretCastOp.getConstifiedMixedOffset();
  auto secondOriginCastOffset =
      secondReinterpretCastOp.getConstifiedMixedOffset();
  auto firstOffsetInfo = normalizeDeinterleaveOffset(firstOriginCastOffset);
  auto secondOffsetInfo = normalizeDeinterleaveOffset(secondOriginCastOffset);
  if (failed(firstOffsetInfo) || failed(secondOffsetInfo) ||
      !hasEquivalentDeinterleaveBase(*firstOffsetInfo, *secondOffsetInfo) ||
      firstOffsetInfo->indexMode == secondOffsetInfo->indexMode)
    return failure();

  std::pair<IndexMode, IndexMode> indexModeRecord = {
      firstOffsetInfo->indexMode, secondOffsetInfo->indexMode};
  OpFoldResult newCastOffset =
      materializeDeinterleaveBaseOffset(*firstOffsetInfo, builder, loc);

  // Create new op
  // 1. Get new destination memref type
  auto dstType = expandInterleaveMemRefType(firstReinterpretCastOp.getType(),
                                            newCastOffset);

  // 2. New tensor::EmptyOp
  auto emptyTensor = builder.create<tensor::EmptyOp>(loc, dstType.getShape(),
                                                     dstType.getElementType());

  // 3. New insert_slice from materialization source into new empty tensor
  SmallVector<OpFoldResult> insertOffsets(dstType.getRank(),
                                          builder.getIndexAttr(0));
  SmallVector<OpFoldResult> insertStrides(dstType.getRank(),
                                          builder.getIndexAttr(1));
  SmallVector<OpFoldResult> insertSizes = llvm::to_vector(
      llvm::map_range(dstType.getShape(), [&](int64_t dim) -> OpFoldResult {
        return builder.getIndexAttr(dim);
      }));
  insertStrides.back() = builder.getIndexAttr(2);
  insertSizes.back() = builder.getIndexAttr(dstType.getShape().back() / 2);
  if (indexModeRecord.first == IndexMode::ODD_MODE) {
    insertOffsets.back() = builder.getIndexAttr(1);
  } else {
    insertOffsets.back() = builder.getIndexAttr(0);
  }
  auto insertFirst = builder.create<tensor::InsertSliceOp>(
      loc,
      llvm::dyn_cast<bufferization::MaterializeInDestinationOp>(
          materializeVec[0])
          .getSource(),
      emptyTensor.getResult(), insertOffsets, insertSizes, insertStrides);

  if (indexModeRecord.second == IndexMode::ODD_MODE) {
    insertOffsets.back() = builder.getIndexAttr(1);
  } else {
    insertOffsets.back() = builder.getIndexAttr(0);
  }
  auto insertSecond = builder.create<tensor::InsertSliceOp>(
      loc,
      llvm::dyn_cast<bufferization::MaterializeInDestinationOp>(
          materializeVec[1])
          .getSource(),
      insertFirst.getResult(), insertOffsets, insertSizes, insertStrides);

  // 4. Reinterpret_cast block arg
  auto newCastSize = firstReinterpretCastOp.getConstifiedMixedSizes();
  auto newCastStride = firstReinterpretCastOp.getConstifiedMixedStrides();
  newCastSize.back() = builder.getIndexAttr(dstType.getShape().back());
  newCastStride.back() = builder.getIndexAttr(1);
  auto newCastOp = builder.create<memref::ReinterpretCastOp>(
      loc, dstType, firstReinterpretCastOp.getViewSource(), newCastOffset,
      newCastSize, newCastStride);

  // 5. Create new bufferization::MaterializeInDestinationOp
  auto newStoreOp = builder.create<bufferization::MaterializeInDestinationOp>(
      loc, insertSecond.getResult(), newCastOp.getResult());
  // Setting writable is necessary as dst is memref type
  newStoreOp.setWritable(true);

  // 6. Erase origin materialization
  materializeVec[0]->erase();
  materializeVec[1]->erase();

  return success();
}

LogicalResult
InterleaveStatusWithMaskOptimization(SmallVector<Operation *> materializeVec) {
  OpBuilder builder(materializeVec[1]);

  auto firstSubviewOpOfReCast =
      llvm::dyn_cast<bufferization::MaterializeInDestinationOp>(
          materializeVec[0])
          .getDest()
          .getDefiningOp<memref::SubViewOp>();
  auto firstSrcExtractSlice =
      llvm::dyn_cast<bufferization::MaterializeInDestinationOp>(
          materializeVec[0])
          .getSource()
          .getDefiningOp<tensor::ExtractSliceOp>();
  auto firstReinterpretCastOp = firstSubviewOpOfReCast.getSource()
                                    .getDefiningOp<memref::ReinterpretCastOp>();

  auto secondSubviewOpOfReCast =
      llvm::dyn_cast<bufferization::MaterializeInDestinationOp>(
          materializeVec[1])
          .getDest()
          .getDefiningOp<memref::SubViewOp>();
  auto secondSrcExtractSlice =
      llvm::dyn_cast<bufferization::MaterializeInDestinationOp>(
          materializeVec[1])
          .getSource()
          .getDefiningOp<tensor::ExtractSliceOp>();
  auto secondReinterpretCastOp =
      secondSubviewOpOfReCast.getSource()
          .getDefiningOp<memref::ReinterpretCastOp>();

  // 1. Both source shapes of subview and extract_slice are equal
  if (firstSubviewOpOfReCast.getSourceType().getShape() !=
      firstSrcExtractSlice.getSourceType().getShape())
    return failure();
  if (secondSubviewOpOfReCast.getSourceType().getShape() !=
      secondSrcExtractSlice.getSourceType().getShape())
    return failure();
  if (firstSubviewOpOfReCast.getSourceType().getShape() !=
      secondSubviewOpOfReCast.getSourceType().getShape())
    return failure();

  // 2. both mask state are equal
  std::function<bool(OpFoldResult, OpFoldResult)> cmpFunc =
      mlir::isEqualConstantIntOrValue;
  if (!mlir::detail::sameOffsetsSizesAndStrides(firstSubviewOpOfReCast,
                                                firstSrcExtractSlice, cmpFunc))
    return failure();
  if (!mlir::detail::sameOffsetsSizesAndStrides(secondSubviewOpOfReCast,
                                                secondSrcExtractSlice, cmpFunc))
    return failure();
  if (!mlir::detail::sameOffsetsSizesAndStrides(
          firstSubviewOpOfReCast, secondSubviewOpOfReCast, cmpFunc))
    return failure();

  // 3. Still judge whether two `ReinterpretCastOp` shape satisfy request
  // a. both size are equal
  if (!isEqualConstantIntOrValueArray(
          firstReinterpretCastOp.getConstifiedMixedSizes(),
          secondReinterpretCastOp.getConstifiedMixedSizes()))
    return failure();
  // b. both strides are equal
  if (!isEqualConstantIntOrValueArray(
          firstReinterpretCastOp.getConstifiedMixedStrides(),
          secondReinterpretCastOp.getConstifiedMixedStrides()))
    return failure();
  // c. both offsets should satisfy tricky rule
  auto firstOriginCastOffset =
      firstReinterpretCastOp.getConstifiedMixedOffset();
  auto secondOriginCastOffset =
      secondReinterpretCastOp.getConstifiedMixedOffset();
  auto firstOffsetInfo = normalizeDeinterleaveOffset(firstOriginCastOffset);
  auto secondOffsetInfo = normalizeDeinterleaveOffset(secondOriginCastOffset);
  if (failed(firstOffsetInfo) || failed(secondOffsetInfo) ||
      !hasEquivalentDeinterleaveBase(*firstOffsetInfo, *secondOffsetInfo) ||
      firstOffsetInfo->indexMode == secondOffsetInfo->indexMode)
    return failure();

  std::pair<IndexMode, IndexMode> indexModeRecord = {
      firstOffsetInfo->indexMode, secondOffsetInfo->indexMode};
  OpFoldResult newCastOffset = materializeDeinterleaveBaseOffset(
      *firstOffsetInfo, builder, materializeVec[1]->getLoc());
  auto loc = materializeVec[1]->getLoc();

  // Create new op
  // 1. Get new destination memref type
  auto dstType = expandInterleaveMemRefType(firstReinterpretCastOp.getType(),
                                            newCastOffset);

  // 2. New tensor::EmptyOp
  auto emptyTensor = builder.create<tensor::EmptyOp>(loc, dstType.getShape(),
                                                     dstType.getElementType());

  // 3. New insert_slice from extract_slice source into new empty tensor
  SmallVector<OpFoldResult> insertOffsets(dstType.getRank(),
                                          builder.getIndexAttr(0));
  SmallVector<OpFoldResult> insertStrides(dstType.getRank(),
                                          builder.getIndexAttr(1));
  SmallVector<OpFoldResult> insertSizes = llvm::to_vector(
      llvm::map_range(dstType.getShape(), [&](int64_t dim) -> OpFoldResult {
        return builder.getIndexAttr(dim);
      }));
  insertStrides.back() = builder.getIndexAttr(2);
  insertSizes.back() = builder.getIndexAttr(dstType.getShape().back() / 2);
  if (indexModeRecord.first == IndexMode::ODD_MODE) {
    insertOffsets.back() = builder.getIndexAttr(1);
  } else {
    insertOffsets.back() = builder.getIndexAttr(0);
  }
  auto insertFirst = builder.create<tensor::InsertSliceOp>(
      loc, firstSrcExtractSlice.getSource(), emptyTensor.getResult(),
      insertOffsets, insertSizes, insertStrides);

  if (indexModeRecord.second == IndexMode::ODD_MODE) {
    insertOffsets.back() = builder.getIndexAttr(1);
  } else {
    insertOffsets.back() = builder.getIndexAttr(0);
  }
  auto insertSecond = builder.create<tensor::InsertSliceOp>(
      loc, secondSrcExtractSlice.getSource(), insertFirst.getResult(),
      insertOffsets, insertSizes, insertStrides);

  // 4. To enable store with mask, create new extract_slice
  SmallVector<OpFoldResult> extractOffsets =
      firstSrcExtractSlice.getMixedOffsets();
  SmallVector<OpFoldResult> extractStrides =
      firstSrcExtractSlice.getMixedStrides();
  SmallVector<OpFoldResult> extractSizes = firstSrcExtractSlice.getMixedSizes();
  if (!llvm::isa<Attribute>(extractSizes.back())) {
    return failure();
  }
  extractSizes.back() = builder.getIndexAttr(
      getConstantIntValue(extractSizes.back()).value() * 2);
  auto newSrcExtractSlice = builder.create<tensor::ExtractSliceOp>(
      loc, insertSecond.getResult(), extractOffsets, extractSizes,
      extractStrides);

  // 5. Reinterpret_cast block arg
  auto newCastSize = firstReinterpretCastOp.getConstifiedMixedSizes();
  auto newCastStride = firstReinterpretCastOp.getConstifiedMixedStrides();
  newCastSize.back() = builder.getIndexAttr(dstType.getShape().back());
  newCastStride.back() = builder.getIndexAttr(1);
  auto newCastOp = builder.create<memref::ReinterpretCastOp>(
      loc, dstType, firstReinterpretCastOp.getViewSource(), newCastOffset,
      newCastSize, newCastStride);

  // 6. Create new memref::SubViewOp of above new reinterpret_cast
  // Here could reuse shape info of new extract_slice
  auto dstSubviewType = memref::SubViewOp::inferResultType(
      dstType, extractOffsets, extractSizes, extractStrides);
  auto newSubviewOpOfReCast = builder.create<memref::SubViewOp>(
      loc, llvm::cast<MemRefType>(dstSubviewType), newCastOp, extractOffsets,
      extractSizes, extractStrides);

  // 7. Create new bufferization::MaterializeInDestinationOp
  auto newStoreOp = builder.create<bufferization::MaterializeInDestinationOp>(
      loc, newSrcExtractSlice.getResult(), newSubviewOpOfReCast.getResult());
  // Setting writable is necessary as dst is memref type
  newStoreOp.setWritable(true);

  // 8. Erase origin operation
  materializeVec[0]->erase();
  materializeVec[1]->erase();
  if (firstSubviewOpOfReCast->use_empty()) {
    firstSubviewOpOfReCast->erase();
  }
  if (firstSrcExtractSlice->use_empty()) {
    firstSrcExtractSlice->erase();
  }
  if (secondSubviewOpOfReCast->use_empty()) {
    secondSubviewOpOfReCast->erase();
  }
  if (secondSrcExtractSlice->use_empty()) {
    secondSrcExtractSlice->erase();
  }

  return success();
}

} // namespace triton
} // namespace mlir
