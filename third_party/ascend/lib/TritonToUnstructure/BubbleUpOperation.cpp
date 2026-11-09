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

#include "TritonToUnstructure/BubbleUpOperation.h"
#include "Utils/Utils.h"

#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "mlir/Transforms/Passes.h"

#define DEBUG_TYPE "triton-bubble-up-operation"

template <typename ExtractOpTy>
BubbleUpExtract<ExtractOpTy>::BubbleUpExtract(MLIRContext *context,
                                              bool enableAggressiveMode)
    : OpRewritePattern<ExtractOpTy>(context),
      enableAggressiveMode(enableAggressiveMode) {}

template <typename ExtractOpTy>
LogicalResult
BubbleUpExtract<ExtractOpTy>::matchAndRewrite(ExtractOpTy op,
                                              PatternRewriter &rewriter) const {
  Value tensorValue;
  if constexpr (std::is_same_v<ExtractOpTy, tensor::ExtractOp>) {
    tensorValue = op.getTensor();
  } else if constexpr (std::is_same_v<ExtractOpTy, tensor::ExtractSliceOp>) {
    tensorValue = op.getSource();
    if (tensorValue.getType() == op.getResult().getType()) {
      rewriter.replaceAllUsesWith(op.getResult(), tensorValue);
      rewriter.eraseOp(op);
      return success();
    }
  } else {
    llvm_unreachable("Unhandled case");
  }
  auto funcOp = op->template getParentOfType<triton::FuncOp>();
  auto parentOp = tensorValue.getDefiningOp();
  auto loc = op.getLoc();

  if (!parentOp || (!enableAggressiveMode && !parentOp->hasOneUse())) {
    return failure();
  }

  // Keep the vector mask attached to discrete memory accesses. Depending on
  // which mask conversion produced the select, the marker is either on the
  // select itself or on the generated load loop that supplies its value.
  if (auto selectOp = dyn_cast<arith::SelectOp>(parentOp);
      selectOp && isa<ShapedType>(selectOp.getCondition().getType())) {
    auto isDiscreteLoadResult = [](Value value) {
      auto forOp = value.getDefiningOp<scf::ForOp>();
      return forOp && forOp->hasAttr("ExtractedLoadOrStore");
    };
    if (selectOp->hasAttr(ConverterUtils::discreteAttrName) ||
        isDiscreteLoadResult(selectOp.getTrueValue()) ||
        isDiscreteLoadResult(selectOp.getFalseValue()))
      return failure();
  }

  LLVM_DEBUG({
    auto &os = llvm::dbgs();
    os << "Before bubble up\n" << op << '\n' << funcOp << "\n";
  });

  if (auto extsiOp = dyn_cast<arith::ExtSIOp>(parentOp)) {
    bubbleUpOperation(op, extsiOp, loc, rewriter);
  } else if (auto addIOp = dyn_cast<arith::AddIOp>(parentOp)) {
    bubbleUpIntBinaryOp(op, addIOp, loc, rewriter);
  } else if (auto subIOp = dyn_cast<arith::SubIOp>(parentOp)) {
    bubbleUpIntBinaryOp(op, subIOp, loc, rewriter);
  } else if (auto mulIOp = dyn_cast<arith::MulIOp>(parentOp)) {
    bubbleUpIntBinaryOp(op, mulIOp, loc, rewriter);
  } else if (auto divSIOp = dyn_cast<arith::DivSIOp>(parentOp)) {
    bubbleUpIntBinaryOp(op, divSIOp, loc, rewriter);
  } else if (auto remSIOp = dyn_cast<arith::RemSIOp>(parentOp)) {
    bubbleUpIntBinaryOp(op, remSIOp, loc, rewriter);
  } else if (auto maxSIOp = dyn_cast<arith::MaxSIOp>(parentOp)) {
    bubbleUpIntBinaryOp(op, maxSIOp, loc, rewriter);
  } else if (auto minSIOp = dyn_cast<arith::MinSIOp>(parentOp)) {
    bubbleUpIntBinaryOp(op, minSIOp, loc, rewriter);
  } else if (auto andIOp = dyn_cast<arith::AndIOp>(parentOp)) {
    bubbleUpIntBinaryOp(op, andIOp, loc, rewriter);
  } else if (auto orIOp = dyn_cast<arith::OrIOp>(parentOp)) {
    bubbleUpIntBinaryOp(op, orIOp, loc, rewriter);
  } else if (auto cmpIOp = dyn_cast<arith::CmpIOp>(parentOp)) {
    bubbleUpOperation(op, cmpIOp, loc, rewriter);
  } else if (auto selectOp = dyn_cast<arith::SelectOp>(parentOp)) {
    bubbleUpOperation(op, selectOp, loc, rewriter);
  } else if (auto truncFOp = dyn_cast<arith::TruncFOp>(parentOp)) {
    bubbleUpOperation(op, truncFOp, loc, rewriter);
  } else if (auto extFOp = dyn_cast<arith::ExtFOp>(parentOp)) {
    bubbleUpOperation(op, extFOp, loc, rewriter);
  } else if (auto fpTosiOp = dyn_cast<arith::FPToSIOp>(parentOp)) {
    bubbleUpOperation(op, fpTosiOp, loc, rewriter);
  } else if (auto siTofpOp = dyn_cast<arith::SIToFPOp>(parentOp)) {
    bubbleUpOperation(op, siTofpOp, loc, rewriter);
  } else if (auto clampFOp = dyn_cast<triton::ClampFOp>(parentOp)) {
    bubbleUpOperation(op, clampFOp, loc, rewriter);
  } else if (auto addFOp = dyn_cast<arith::AddFOp>(parentOp)) {
    bubbleUpFloatBinaryOp<arith::AddFOp>(op, addFOp, loc, rewriter);
  } else if (auto subFOp = dyn_cast<arith::SubFOp>(parentOp)) {
    bubbleUpFloatBinaryOp<arith::SubFOp>(op, subFOp, loc, rewriter);
  } else if (auto mulFOp = dyn_cast<arith::MulFOp>(parentOp)) {
    bubbleUpFloatBinaryOp<arith::MulFOp>(op, mulFOp, loc, rewriter);
  } else if (auto divFOp = dyn_cast<arith::DivFOp>(parentOp)) {
    bubbleUpFloatBinaryOp(op, divFOp, loc, rewriter);
  } else if (auto minNumFOp = dyn_cast<arith::MinNumFOp>(parentOp)) {
    bubbleUpFloatBinaryOp<arith::MinNumFOp>(op, minNumFOp, loc, rewriter);
  } else if (auto maxNumFOp = dyn_cast<arith::MaxNumFOp>(parentOp)) {
    bubbleUpFloatBinaryOp<arith::MaxNumFOp>(op, maxNumFOp, loc, rewriter);
  } else if (auto cmpFOp = dyn_cast<arith::CmpFOp>(parentOp)) {
    bubbleUpOperation(op, cmpFOp, loc, rewriter);
  } else if (auto broadCastOp = dyn_cast<triton::BroadcastOp>(parentOp)) {
    bubbleUpOperation(op, broadCastOp, loc, rewriter);
  } else if (auto expandDimsOp = dyn_cast<triton::ExpandDimsOp>(parentOp)) {
    bubbleUpOperation(op, expandDimsOp, loc, rewriter);
  } else if (auto splatOp = dyn_cast<triton::SplatOp>(parentOp)) {
    bubbleUpOperation(op, splatOp, loc, rewriter);
  } else if (auto makeRangeOp = dyn_cast<triton::MakeRangeOp>(parentOp)) {
    bubbleUpOperation(op, makeRangeOp, loc, rewriter);
  } else if (auto addPtrOp = dyn_cast<triton::AddPtrOp>(parentOp)) {
    bubbleUpOperation(op, addPtrOp, loc, rewriter);
  } else if (auto floorOp = dyn_cast<math::FloorOp>(parentOp)) {
    bubbleUpOperation(op, floorOp, loc, rewriter);
  } else if (auto ceilOp = dyn_cast<math::CeilOp>(parentOp)) {
    bubbleUpOperation(op, ceilOp, loc, rewriter);
  } else if (auto extractSliceOp = dyn_cast<tensor::ExtractSliceOp>(parentOp)) {
    if constexpr (std::is_same_v<ExtractOpTy, tensor::ExtractOp>) {
      bubbleUpOperation(op, extractSliceOp, loc, rewriter);
    } else {
      return failure();
    }
  } else {
    return failure();
  }
  if (parentOp->use_empty())
    rewriter.eraseOp(parentOp);

  LLVM_DEBUG({
    auto &os = llvm::dbgs();
    os << "After bubble up\n" << funcOp << '\n';
  });

  return success();
}

template <typename ExtractOpTy>
Value BubbleUpExtract<ExtractOpTy>::createExtractOp(
    ExtractOpTy op, Value value, Location loc,
    PatternRewriter &rewriter) const {
  llvm_unreachable("Unhandled extract operation");
}

template <>
Value BubbleUpExtract<tensor::ExtractOp>::createExtractOp(
    tensor::ExtractOp op, Value value, Location loc,
    PatternRewriter &rewriter) const {
  auto extractedOp =
      rewriter.create<tensor::ExtractOp>(loc, value, op.getIndices());
  extractedOp->setAttr(ConverterUtils::discreteAttrName,
                       UnitAttr::get(rewriter.getContext()));
  return extractedOp;
}

template <>
Value BubbleUpExtract<tensor::ExtractSliceOp>::createExtractOp(
    tensor::ExtractSliceOp op, Value value, Location loc,
    PatternRewriter &rewriter) const {
  auto extractedType = getExtractSlicedType(
      op.getMixedSizes(), op.getDroppedDims(), getElementTypeOrSelf(value));
  auto extractedOp = rewriter.create<tensor::ExtractSliceOp>(
      loc, extractedType, value, op.getMixedOffsets(), op.getMixedSizes(),
      op.getMixedStrides());
  extractedOp->setAttr(ConverterUtils::discreteAttrName,
                       UnitAttr::get(rewriter.getContext()));
  return extractedOp;
}

template <typename ExtractOpTy>
template <typename BinOpTy>
void BubbleUpExtract<ExtractOpTy>::bubbleUpIntBinaryOp(
    ExtractOpTy op, BinOpTy binOp, Location loc,
    PatternRewriter &rewriter) const {
  auto lhs = createExtractOp(op, binOp.getLhs(), loc, rewriter);
  auto rhs = createExtractOp(op, binOp.getRhs(), loc, rewriter);
  LLVM_DEBUG({
    auto &os = llvm::dbgs();
    os << "Binary\n" << *op << '\n' << binOp << '\n';
  });
  rewriter.replaceOpWithNewOp<BinOpTy>(op, lhs, rhs);
}

template <typename ExtractOpTy>
template <typename BinOpTy>
void BubbleUpExtract<ExtractOpTy>::bubbleUpFloatBinaryOp(
    ExtractOpTy op, BinOpTy binOp, Location loc,
    PatternRewriter &rewriter) const {
  auto lhs = createExtractOp(op, binOp.getLhs(), loc, rewriter);
  auto rhs = createExtractOp(op, binOp.getRhs(), loc, rewriter);
  rewriter.replaceOpWithNewOp<BinOpTy>(op, lhs, rhs);
}

template <typename ExtractOpTy>
void BubbleUpExtract<ExtractOpTy>::bubbleUpOperation(
    ExtractOpTy op, arith::ExtSIOp parentOp, Location loc,
    PatternRewriter &rewriter) const {
  auto in = createExtractOp(op, parentOp.getIn(), loc, rewriter);
  rewriter.replaceOpWithNewOp<arith::ExtSIOp>(op, op.getResult().getType(), in);
}

template <typename ExtractOpTy>
void BubbleUpExtract<ExtractOpTy>::bubbleUpOperation(
    ExtractOpTy op, arith::CmpIOp parentOp, Location loc,
    PatternRewriter &rewriter) const {
  auto lhs = createExtractOp(op, parentOp.getLhs(), loc, rewriter);
  auto rhs = createExtractOp(op, parentOp.getRhs(), loc, rewriter);
  rewriter.replaceOpWithNewOp<arith::CmpIOp>(op, parentOp.getPredicateAttr(),
                                             lhs, rhs);
}

template <typename ExtractOpTy>
void BubbleUpExtract<ExtractOpTy>::bubbleUpOperation(
    ExtractOpTy op, arith::SelectOp parentOp, Location loc,
    PatternRewriter &rewriter) const {
  Value condition = parentOp.getCondition();
  if (isa<ShapedType>(condition.getType()))
    condition = createExtractOp(op, condition, loc, rewriter);
  Value trueValue = createExtractOp(op, parentOp.getTrueValue(), loc, rewriter);
  Value falseValue =
      createExtractOp(op, parentOp.getFalseValue(), loc, rewriter);
  rewriter.replaceOpWithNewOp<arith::SelectOp>(op, condition, trueValue,
                                               falseValue);
}

template <>
void BubbleUpExtract<tensor::ExtractOp>::bubbleUpOperation(
    tensor::ExtractOp op, triton::BroadcastOp parentOp, Location loc,
    PatternRewriter &rewriter) const {
  auto src = parentOp.getSrc();
  auto srcShape = cast<RankedTensorType>(src.getType()).getShape();
  SmallVector<Value> newIndices;
  for (const auto &[index, shape] :
       llvm::zip_equal(op.getIndices(), srcShape)) {
    if (shape == 1) {
      newIndices.push_back(
          rewriter.create<arith::ConstantOp>(loc, rewriter.getIndexAttr(0)));
    } else {
      newIndices.push_back(index);
    }
  }
  auto extractedOp = rewriter.create<tensor::ExtractOp>(loc, src, newIndices);
  extractedOp->setAttr(ConverterUtils::discreteAttrName,
                       UnitAttr::get(rewriter.getContext()));
  rewriter.replaceOp(op, extractedOp);
}

template <>
void BubbleUpExtract<tensor::ExtractSliceOp>::bubbleUpOperation(
    tensor::ExtractSliceOp op, triton::BroadcastOp parentOp, Location loc,
    PatternRewriter &rewriter) const {
  auto src = parentOp.getSrc();
  auto srcShape = cast<RankedTensorType>(src.getType()).getShape();
  SmallVector<OpFoldResult> newOffsets;
  SmallVector<OpFoldResult> newSizes;
  bool isScalarLikeSrc = true;
  for (const auto &[offset, size, shape] :
       llvm::zip_equal(op.getMixedOffsets(), op.getMixedSizes(), srcShape)) {
    if (shape == 1) {
      newOffsets.push_back(rewriter.getIndexAttr(0));
      newSizes.push_back(rewriter.getIndexAttr(1));
    } else {
      newOffsets.push_back(offset);
      newSizes.push_back(size);
    }
    if (getConstantIntValue(newSizes.back()).value_or(-1) != 1)
      isScalarLikeSrc = false;
  }
  auto extractedType = getExtractSlicedType(newSizes, op.getDroppedDims(),
                                            getElementTypeOrSelf(src));
  auto extractedOp = rewriter.create<tensor::ExtractSliceOp>(
      loc, extractedType, src, newOffsets, newSizes, op.getMixedStrides());
  extractedOp->setAttr(ConverterUtils::discreteAttrName,
                       UnitAttr::get(rewriter.getContext()));
  if (isScalarLikeSrc) {
    SmallVector<Value> indices(
        extractedType.getRank(),
        rewriter.create<arith::ConstantOp>(loc, rewriter.getIndexAttr(0)));
    auto extractedValue =
        rewriter.create<tensor::ExtractOp>(loc, extractedOp, indices);
    rewriter.replaceOpWithNewOp<triton::SplatOp>(op, op.getResult().getType(),
                                                 extractedValue);
  } else {
    rewriter.replaceOpWithNewOp<triton::BroadcastOp>(
        op, op.getResult().getType(), extractedOp);
  }
}

template <>
void BubbleUpExtract<tensor::ExtractOp>::bubbleUpOperation(
    tensor::ExtractOp op, triton::ExpandDimsOp parentOp, Location loc,
    PatternRewriter &rewriter) const {
  auto src = parentOp.getSrc();
  SmallVector<Value> newIndices;
  for (const auto index : llvm::enumerate(op.getIndices())) {
    if (index.index() != parentOp.getAxis())
      newIndices.push_back(index.value());
  }
  auto extractedOp = rewriter.create<tensor::ExtractOp>(loc, src, newIndices);
  extractedOp->setAttr(ConverterUtils::discreteAttrName,
                       UnitAttr::get(rewriter.getContext()));
  rewriter.replaceOp(op, extractedOp);
}

template <>
void BubbleUpExtract<tensor::ExtractSliceOp>::bubbleUpOperation(
    tensor::ExtractSliceOp op, triton::ExpandDimsOp parentOp, Location loc,
    PatternRewriter &rewriter) const {
  auto src = parentOp.getSrc();
  auto srcShape = cast<RankedTensorType>(src.getType()).getShape();
  auto offsets = op.getMixedOffsets();
  auto sizes = op.getMixedSizes();
  auto strides = op.getMixedStrides();
  auto axis = parentOp.getAxis();
  int64_t axisOffset = 0;
  SmallVector<OpFoldResult> newOffsets;
  SmallVector<OpFoldResult> newSizes;
  SmallVector<OpFoldResult> newStrides;
  auto droppedDim = op.getDroppedDims();
  llvm::SmallBitVector srcDroppedDims;
  for (size_t i = 0; i <= srcShape.size(); i++) {
    if (i != axis) {
      newOffsets.push_back(offsets[i]);
      newSizes.push_back(sizes[i]);
      newStrides.push_back(strides[i]);
      srcDroppedDims.push_back(droppedDim[i]);
      if (i < axis && droppedDim[i])
        axisOffset++;
    }
  }
  auto extractedType =
      getExtractSlicedType(newSizes, srcDroppedDims, getElementTypeOrSelf(src));
  if (extractedType == src.getType()) {
    rewriter.replaceOp(op, src);
  } else {
    auto extractedOp = rewriter.create<tensor::ExtractSliceOp>(
        loc, extractedType, src, newOffsets, newSizes, newStrides);
    extractedOp->setAttr(ConverterUtils::discreteAttrName,
                         UnitAttr::get(rewriter.getContext()));
    rewriter.replaceOpWithNewOp<triton::ExpandDimsOp>(op, extractedOp,
                                                      axis - axisOffset);
  }
}

template <>
void BubbleUpExtract<tensor::ExtractOp>::bubbleUpOperation(
    tensor::ExtractOp op, triton::SplatOp parentOp, Location loc,
    PatternRewriter &rewriter) const {
  auto src = parentOp.getSrc();
  rewriter.replaceOp(op, src);
}

template <>
void BubbleUpExtract<tensor::ExtractSliceOp>::bubbleUpOperation(
    tensor::ExtractSliceOp op, triton::SplatOp parentOp, Location loc,
    PatternRewriter &rewriter) const {
  auto src = parentOp.getSrc();
  rewriter.replaceOpWithNewOp<triton::SplatOp>(
      op, cast<RankedTensorType>(op.getResult().getType()), src);
}

template <>
void BubbleUpExtract<tensor::ExtractOp>::bubbleUpOperation(
    tensor::ExtractOp op, triton::MakeRangeOp parentOp, Location loc,
    PatternRewriter &rewriter) const {
  auto resultType = cast<RankedTensorType>(parentOp.getResult().getType());
  int32_t start = parentOp.getStart();
  Value idx = op.getIndices()[0];
  Value result = rewriter.create<arith::IndexCastOp>(
      op.getLoc(), resultType.getElementType(), idx);
  if (start != 0) {
    Value startVal = rewriter.create<arith::ConstantOp>(
        op.getLoc(),
        rewriter.getIntegerAttr(resultType.getElementType(), start));
    result = rewriter.create<arith::AddIOp>(op.getLoc(), result, startVal);
  }
  rewriter.replaceOp(op, result);
}

template <>
void BubbleUpExtract<tensor::ExtractSliceOp>::bubbleUpOperation(
    tensor::ExtractSliceOp op, triton::MakeRangeOp parentOp, Location loc,
    PatternRewriter &rewriter) const {
  auto resultType = cast<RankedTensorType>(parentOp.getResult().getType());
  auto idxOfr = op.getMixedOffsets()[0];
  Value idx = getValueOrCreateConstantIndexOp(rewriter, op.getLoc(), idxOfr);
  idx = rewriter.create<arith::IndexCastOp>(op.getLoc(),
                                            resultType.getElementType(), idx);
  rewriter.replaceOpWithNewOp<triton::SplatOp>(op, op.getResult().getType(),
                                               idx);
}

template <typename ExtractOpTy>
void BubbleUpExtract<ExtractOpTy>::bubbleUpOperation(
    ExtractOpTy op, triton::AddPtrOp parentOp, Location loc,
    PatternRewriter &rewriter) const {
  auto ptr = createExtractOp(op, parentOp.getPtr(), loc, rewriter);
  auto offset = createExtractOp(op, parentOp.getOffset(), loc, rewriter);
  rewriter.replaceOpWithNewOp<triton::AddPtrOp>(op, ptr.getType(), ptr, offset);
}

template <typename ExtractOpTy>
void BubbleUpExtract<ExtractOpTy>::bubbleUpOperation(
    ExtractOpTy op, arith::TruncFOp parentOp, Location loc,
    PatternRewriter &rewriter) const {
  auto in = createExtractOp(op, parentOp.getIn(), loc, rewriter);
  rewriter.replaceOpWithNewOp<arith::TruncFOp>(op, op.getResult().getType(),
                                               in);
}

template <typename ExtractOpTy>
void BubbleUpExtract<ExtractOpTy>::bubbleUpOperation(
    ExtractOpTy op, arith::ExtFOp parentOp, Location loc,
    PatternRewriter &rewriter) const {
  auto in = createExtractOp(op, parentOp.getIn(), loc, rewriter);
  rewriter.replaceOpWithNewOp<arith::ExtFOp>(op, op.getResult().getType(), in);
}

template <typename ExtractOpTy>
void BubbleUpExtract<ExtractOpTy>::bubbleUpOperation(
    ExtractOpTy op, arith::FPToSIOp parentOp, Location loc,
    PatternRewriter &rewriter) const {
  auto in = createExtractOp(op, parentOp.getIn(), loc, rewriter);
  rewriter.replaceOpWithNewOp<arith::FPToSIOp>(op, op.getResult().getType(),
                                               in);
}

template <typename ExtractOpTy>
void BubbleUpExtract<ExtractOpTy>::bubbleUpOperation(
    ExtractOpTy op, arith::SIToFPOp parentOp, Location loc,
    PatternRewriter &rewriter) const {
  auto in = createExtractOp(op, parentOp.getIn(), loc, rewriter);
  rewriter.replaceOpWithNewOp<arith::SIToFPOp>(op, op.getResult().getType(),
                                               in);
}

template <typename ExtractOpTy>
void BubbleUpExtract<ExtractOpTy>::bubbleUpOperation(
    ExtractOpTy op, triton::ClampFOp parentOp, Location loc,
    PatternRewriter &rewriter) const {
  auto x = createExtractOp(op, parentOp.getX(), loc, rewriter);
  auto min = createExtractOp(op, parentOp.getMin(), loc, rewriter);
  auto max = createExtractOp(op, parentOp.getMax(), loc, rewriter);
  rewriter.replaceOpWithNewOp<triton::ClampFOp>(op, x, min, max,
                                                parentOp.getPropagateNan());
}

template <typename ExtractOpTy>
void BubbleUpExtract<ExtractOpTy>::bubbleUpOperation(
    ExtractOpTy op, arith::CmpFOp parentOp, Location loc,
    PatternRewriter &rewriter) const {
  auto lhs = createExtractOp(op, parentOp.getLhs(), loc, rewriter);
  auto rhs = createExtractOp(op, parentOp.getRhs(), loc, rewriter);
  rewriter.replaceOpWithNewOp<arith::CmpFOp>(op, parentOp.getPredicateAttr(),
                                             lhs, rhs);
}

template <typename ExtractOpTy>
void BubbleUpExtract<ExtractOpTy>::bubbleUpOperation(
    ExtractOpTy op, math::FloorOp parentOp, Location loc,
    PatternRewriter &rewriter) const {
  auto operand = createExtractOp(op, parentOp.getOperand(), loc, rewriter);
  rewriter.replaceOpWithNewOp<math::FloorOp>(op, operand,
                                             parentOp.getFastmath());
}

template <typename ExtractOpTy>
void BubbleUpExtract<ExtractOpTy>::bubbleUpOperation(
    ExtractOpTy op, math::CeilOp parentOp, Location loc,
    PatternRewriter &rewriter) const {
  auto operand = createExtractOp(op, parentOp.getOperand(), loc, rewriter);
  rewriter.replaceOpWithNewOp<math::CeilOp>(op, operand,
                                            parentOp.getFastmath());
}

template <>
void BubbleUpExtract<tensor::ExtractOp>::bubbleUpOperation(
    tensor::ExtractOp op, tensor::ExtractSliceOp parentOp, Location loc,
    PatternRewriter &rewriter) const {
  SmallVector<Value> indices;
  SmallVector<Value> newIndices;
  auto indiceIter = op.getIndices().begin();
  auto droppedDims = parentOp.getDroppedDims();
  for (auto [idx, offset] : llvm::enumerate(parentOp.getMixedOffsets())) {
    if (droppedDims[idx]) {
      auto zeroIdx = rewriter.create<arith::ConstantOp>(
          op.getLoc(), rewriter.getIndexAttr(0));
      indices.push_back(zeroIdx);
    } else {
      indices.push_back(*indiceIter);
      ++indiceIter;
    }
  }
  for (const auto &[offset, index] :
       llvm::zip_equal(parentOp.getMixedOffsets(), indices)) {
    Value offsetVal =
        getValueOrCreateConstantIndexOp(rewriter, op.getLoc(), offset);
    newIndices.push_back(
        rewriter.create<arith::AddIOp>(op.getLoc(), offsetVal, index));
  }
  rewriter
      .replaceOpWithNewOp<tensor::ExtractOp>(op, parentOp.getSource(),
                                             newIndices)
      ->setAttr(ConverterUtils::discreteAttrName,
                UnitAttr::get(rewriter.getContext()));
}

BubbleUpOperationPass::BubbleUpOperationPass(
    const BubbleUpOperationOptions &options)
    : BubbleUpOperationBase(options) {}

void BubbleUpOperationPass::runOnOperation() {
  ModuleOp moduleOp = getOperation();
  MLIRContext *ctx = &getContext();

  RewritePatternSet patterns(ctx);
  patterns.add<BubbleUpExtract<tensor::ExtractOp>,
               BubbleUpExtract<tensor::ExtractSliceOp>>(ctx,
                                                        enableAggressiveMode);

  if (failed(applyPatternsGreedily(moduleOp, std::move(patterns)))) {
    moduleOp->emitError("failed to apply Patterns");
    signalPassFailure();
  }

  PassManager pm(&getContext(), moduleOp.getOperationName());
  pm.addPass(createCSEPass());
  pm.addPass(createCanonicalizerPass());
  if (failed(runPipeline(pm, getOperation()))) {
    signalPassFailure();
  }
}

std::unique_ptr<OperationPass<ModuleOp>>
triton::createBubbleUpOperationPass(const BubbleUpOperationOptions &options) {
  return std::make_unique<BubbleUpOperationPass>(options);
}
