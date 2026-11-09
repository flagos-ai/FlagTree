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

#include "TritonToGraph/LegacyMemoryAccess/StridedLoadStoreRewrite.h"
#include "TritonMemoryAccess/LoadStoreMaskAnalysis.h"
#include "TritonMemoryAccess/MemoryAccessTags.h"
#include "TritonToStructured/PtrAnalysis.h"
#include "Utils/Utils.h"

#include "Dialect/TritonAscend/IR/TritonAscendDialect.h"
#include "TritonToGraph/GraphOptimization.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Utils/StaticValueUtils.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Matchers.h"

#include "llvm/ADT/Twine.h"
#include "llvm/Support/Debug.h"

#include <cstdint>
#include <cstdlib>
#include <functional>

#define DEBUG_TYPE "graph-optimize"

namespace StridedLoadStoreRewrite {

using namespace mlir;
using namespace triton;

namespace {

// V1 fast-path supports up to 5D tensors, mirroring
// UnstructureConversionPass::tryRewriteIndirectFastPath.
constexpr size_t kFastPathRankLimit = 5;

// Funnel for the eight rewrite variants, which differ only in how the pointer
// was formed and which op they produce.
static void logMatch(Location loc, const llvm::Twine &pointerKind,
                     const llvm::Twine &rewrite, int64_t lastStride) {
  LLVM_DEBUG(
      llvm::dbgs() << "[" DEBUG_TYPE "] matched graph optimization rule "
                   << static_cast<unsigned>(
                          cfg::GraphOptimizationRuleId::StridedLoadStoreRewrite)
                   << " ("
                   << cfg::getGraphOptimizationRuleName(
                          cfg::GraphOptimizationRuleId::StridedLoadStoreRewrite)
                   << ") at " << loc << ": [" << pointerKind << "] " << rewrite
                   << " lastStride=" << lastStride << "\n");
}

// Returns true iff `v` is a static integer constant with |v| > 1.
static bool isStaticConstAbsGtOne(Value v) {
  IntegerAttr scalarAttr;
  if (matchPattern(v, m_Constant(&scalarAttr)))
    return std::abs(scalarAttr.getValue().getSExtValue()) > 1;
  DenseElementsAttr denseAttr;
  if (matchPattern(v, m_Constant(&denseAttr)) && denseAttr.isSplat() &&
      denseAttr.getElementType().isInteger())
    return std::abs(denseAttr.getSplatValue<llvm::APInt>().getSExtValue()) > 1;
  // Transparently see through tt.splat of a scalar constant.
  if (auto splatOp = v.getDefiningOp<triton::SplatOp>())
    return isStaticConstAbsGtOne(splatOp.getSrc());
  return false;
}

static std::optional<int64_t> getStaticConstInt(Value v) {
  IntegerAttr scalarAttr;
  if (matchPattern(v, m_Constant(&scalarAttr)))
    return scalarAttr.getValue().getSExtValue();
  DenseElementsAttr denseAttr;
  if (matchPattern(v, m_Constant(&denseAttr)) && denseAttr.isSplat() &&
      denseAttr.getElementType().isInteger())
    return denseAttr.getSplatValue<llvm::APInt>().getSExtValue();
  if (auto splatOp = v.getDefiningOp<triton::SplatOp>())
    return getStaticConstInt(splatOp.getSrc());
  return std::nullopt;
}

static std::optional<int64_t> getStaticMaskUpperBound(Value mask) {
  if (!mask)
    return std::nullopt;
  if (auto cmp = mask.getDefiningOp<arith::CmpIOp>()) {
    auto bound = getStaticConstInt(cmp.getRhs());
    if (!bound)
      return std::nullopt;
    if (cmp.getPredicate() == arith::CmpIPredicate::slt)
      return *bound;
    if (cmp.getPredicate() == arith::CmpIPredicate::sle)
      return *bound + 1;
    return std::nullopt;
  }
  if (auto andOp = mask.getDefiningOp<arith::AndIOp>()) {
    auto lhsBound = getStaticMaskUpperBound(andOp.getLhs());
    auto rhsBound = getStaticMaskUpperBound(andOp.getRhs());
    if (lhsBound && rhsBound)
      return std::min(*lhsBound, *rhsBound);
    return lhsBound ? lhsBound : rhsBound;
  }
  return std::nullopt;
}

static bool
shouldRouteMaskedSingleTilePow2ToIndirect(Value mask,
                                          RankedTensorType tensorType) {
  if (!mask || tensorType.getRank() != 1)
    return false;
  int64_t blockSize = tensorType.getShape()[0];
  if (ShapedType::isDynamic(blockSize))
    return false;
  auto upperBound = getStaticMaskUpperBound(mask);
  return upperBound && *upperBound <= blockSize;
}

// Cheaply detect tensor-level static stride > 1 before running PtrAnalysis,
// which mutates IR. Dynamic stride stays on the structured SIMD path, and
// scalar offset arithmetic does not affect per-element stride.
static bool offsetMayContainStrideGtOne(Value offset, int depthBudget = 16) {
  if (depthBudget <= 0) {
    return true; // Give up cheaply and let PtrAnalysis decide downstream.
  }
  if (!isa<RankedTensorType>(offset.getType())) {
    return false;
  }
  Operation *defOp = offset.getDefiningOp();
  if (!defOp) {
    return false;
  }
  if (auto mul = dyn_cast<arith::MulIOp>(defOp)) {
    if (isStaticConstAbsGtOne(mul.getLhs()) ||
        isStaticConstAbsGtOne(mul.getRhs())) {
      return true;
    }
    return offsetMayContainStrideGtOne(mul.getLhs(), depthBudget - 1) ||
           offsetMayContainStrideGtOne(mul.getRhs(), depthBudget - 1);
  }
  // arith.shli %a, %k effectively multiplies by 2^k; treat shift by >=1 as
  // "may contain stride > 1".
  if (auto shl = dyn_cast<arith::ShLIOp>(defOp)) {
    APInt c;
    if (matchPattern(shl.getRhs(), m_ConstantInt(&c)) &&
        c.getSExtValue() >= 1) {
      return true;
    }
    DenseElementsAttr denseAttr;
    if (matchPattern(shl.getRhs(), m_Constant(&denseAttr)) &&
        denseAttr.isSplat() && denseAttr.getElementType().isInteger() &&
        denseAttr.getSplatValue<llvm::APInt>().getSExtValue() >= 1) {
      return true;
    }
    return offsetMayContainStrideGtOne(shl.getLhs(), depthBudget - 1);
  }
  for (Value operand : defOp->getOperands()) {
    if (offsetMayContainStrideGtOne(operand, depthBudget - 1)) {
      return true;
    }
  }
  return false;
}

// Walk through shape-only wrappers to find the underlying scalar !tt.ptr<T>.
// ChunkCoalescing lifts invariant pointer tensors as
// broadcast(expand_dims(splat(ptr))), which is still a scalar base pointer for
// indirect access construction.
static Value getScalarBasePtr(Value tensorPtr, int depthBudget = 8) {
  if (depthBudget <= 0)
    return Value();
  if (auto splatOp = tensorPtr.getDefiningOp<triton::SplatOp>()) {
    Value src = splatOp.getSrc();
    if (isa<triton::PointerType>(src.getType())) {
      return src;
    }
  }
  if (auto broadcastOp = tensorPtr.getDefiningOp<triton::BroadcastOp>())
    return getScalarBasePtr(broadcastOp.getSrc(), depthBudget - 1);
  if (auto expandDimsOp = tensorPtr.getDefiningOp<triton::ExpandDimsOp>())
    return getScalarBasePtr(expandDimsOp.getSrc(), depthBudget - 1);
  return Value();
}

// Ensure the per-element offset tensor has i64 element type, matching the
// convention used elsewhere (UnstructureConversionPass::parseAddPtr).
static Value ensureI64OffsetTensor(Value offsetTensor, Location loc,
                                   PatternRewriter &rewriter) {
  auto tensorTy = dyn_cast<RankedTensorType>(offsetTensor.getType());
  if (!tensorTy)
    return Value();
  auto eltTy = dyn_cast<IntegerType>(tensorTy.getElementType());
  if (!eltTy)
    return Value();
  if (eltTy.getWidth() == 64)
    return offsetTensor;
  auto newTy =
      RankedTensorType::get(tensorTy.getShape(), rewriter.getIntegerType(64));
  return rewriter.create<arith::ExtSIOp>(loc, newTy, offsetTensor);
}

// Promote a scalar to i64. Handles i32/i64 integers and index types.
static Value ensureI64Scalar(Value v, Location loc, PatternRewriter &rewriter) {
  Type ty = v.getType();
  if (auto intTy = dyn_cast<IntegerType>(ty)) {
    if (intTy.getWidth() == 64)
      return v;
    return rewriter.create<arith::ExtSIOp>(loc, rewriter.getI64Type(), v);
  }
  if (isa<IndexType>(ty)) {
    return rewriter.create<arith::IndexCastOp>(loc, rewriter.getI64Type(), v);
  }
  return Value(); // Unsupported scalar type.
}

static LogicalResult unwrapScalarAddPtrChain(Value scalarPtr, Value &src,
                                             Value &scalarOffset, Location loc,
                                             PatternRewriter &rewriter) {
  src = scalarPtr;
  scalarOffset = Value();
  while (auto addPtrOp = src.getDefiningOp<triton::AddPtrOp>()) {
    if (isa<RankedTensorType>(addPtrOp.getPtr().getType()))
      break;
    if (!scalarOffset)
      scalarOffset = rewriter.create<arith::ConstantOp>(
          loc, rewriter.getI64IntegerAttr(0));
    Value offset = ensureI64Scalar(addPtrOp.getOffset(), loc, rewriter);
    if (!offset)
      return failure();
    scalarOffset = rewriter.create<arith::AddIOp>(loc, scalarOffset, offset);
    src = addPtrOp.getPtr();
  }
  return success();
}

static Value addScalarOffsetToTensor(Value offsetTensor, Value scalarOffset,
                                     Location loc, PatternRewriter &rewriter) {
  if (!scalarOffset)
    return offsetTensor;
  APInt scalarOffsetConst;
  if (matchPattern(scalarOffset, m_ConstantInt(&scalarOffsetConst)) &&
      scalarOffsetConst.isZero())
    return offsetTensor;
  auto tensorType = cast<RankedTensorType>(offsetTensor.getType());
  Value scalarOffsetTensor =
      rewriter.create<triton::SplatOp>(loc, tensorType, scalarOffset);
  return rewriter.create<arith::AddIOp>(loc, offsetTensor, scalarOffsetTensor);
}

static Value getStrideLoadOtherScalar(triton::LoadOp op,
                                      RankedTensorType resultType,
                                      PatternRewriter &rewriter) {
  auto loc = op.getLoc();
  Type elementType = resultType.getElementType();
  if (Value other = op.getOther()) {
    if (auto splatOp = other.getDefiningOp<triton::SplatOp>())
      return splatOp.getSrc();

    DenseElementsAttr denseAttr;
    if (matchPattern(other, m_Constant(&denseAttr)) && denseAttr.isSplat()) {
      if (auto floatType = dyn_cast<FloatType>(elementType)) {
        return rewriter.create<arith::ConstantOp>(
            loc, rewriter.getFloatAttr(elementType,
                                       denseAttr.getSplatValue<APFloat>()));
      }
      if (isa<IntegerType>(elementType)) {
        return rewriter.create<arith::ConstantOp>(
            loc, rewriter.getIntegerAttr(elementType,
                                         denseAttr.getSplatValue<APInt>()));
      }
    }
    return Value();
  }

  if (op.getPadding().has_value() &&
      op.getPadding().value() == triton::PaddingOption::PAD_NAN) {
    auto floatTy = dyn_cast<FloatType>(elementType);
    if (!floatTy)
      return Value();
    return rewriter.create<arith::ConstantOp>(
        loc, rewriter.getFloatAttr(
                 elementType, APFloat::getNaN(floatTy.getFloatSemantics())));
  }
  return rewriter.create<arith::ConstantOp>(loc,
                                            rewriter.getZeroAttr(elementType));
}

static Value createStrideLoadOp(Location loc, RankedTensorType resultType,
                                Value src, Value offset, Value other,
                                ArrayRef<Value> strides, ArrayRef<Value> numels,
                                PatternRewriter &rewriter) {
  int64_t rank = resultType.getRank();
  if (static_cast<int64_t>(strides.size()) != rank ||
      static_cast<int64_t>(numels.size()) != rank) {
    return Value();
  }

  Type indexType = rewriter.getI32Type();
  auto chooseIndexType = [&](Value value) -> bool {
    Type type = value.getType();
    if (isa<IndexType>(type) || type.isInteger(64)) {
      indexType = rewriter.getI64Type();
      return true;
    }
    return type.isInteger(32);
  };
  auto castToIndexType = [&](Value value) -> Value {
    Type type = value.getType();
    if (type == indexType)
      return value;
    if (isa<IndexType>(type))
      return rewriter.create<arith::IndexCastOp>(loc, indexType, value);
    if (type.isInteger(32) && indexType.isInteger(64))
      return rewriter.create<arith::ExtSIOp>(loc, indexType, value);
    return Value();
  };
  auto chooseAll = [&](ArrayRef<Value> values) -> bool {
    for (Value value : values) {
      if (!chooseIndexType(value))
        return false;
    }
    return true;
  };

  if (!chooseIndexType(offset) || !chooseAll(strides) || !chooseAll(numels))
    return Value();
  offset = castToIndexType(offset);
  if (!offset)
    return Value();
  SmallVector<Value> castStrides;
  SmallVector<Value> castNumels;
  castStrides.reserve(rank);
  castNumels.reserve(rank);
  for (Value stride : strides) {
    stride = castToIndexType(stride);
    if (!stride)
      return Value();
    castStrides.push_back(stride);
  }
  for (Value numel : numels) {
    numel = castToIndexType(numel);
    if (!numel)
      return Value();
    castNumels.push_back(numel);
  }

  auto strideLoad = rewriter.create<triton::ascend::StrideLoadOp>(
      loc, resultType, src, offset, other, castStrides, castNumels);
  strideLoad->setAttr(RewrittenByStridedLoadStoreRewriteTAG,
                      UnitAttr::get(rewriter.getContext()));
  return strideLoad->getResult(0);
}

static Operation *createStrideStoreOp(Location loc, RankedTensorType valueType,
                                      Value dst, Value src, Value offset,
                                      ArrayRef<Value> strides,
                                      ArrayRef<Value> numels,
                                      PatternRewriter &rewriter) {
  int64_t rank = valueType.getRank();
  if (static_cast<int64_t>(strides.size()) != rank ||
      static_cast<int64_t>(numels.size()) != rank) {
    return nullptr;
  }

  Type indexType = rewriter.getI32Type();
  auto chooseIndexType = [&](Value value) -> bool {
    Type type = value.getType();
    if (isa<IndexType>(type) || type.isInteger(64)) {
      indexType = rewriter.getI64Type();
      return true;
    }
    return type.isInteger(32);
  };
  auto castToIndexType = [&](Value value) -> Value {
    Type type = value.getType();
    if (type == indexType)
      return value;
    if (isa<IndexType>(type))
      return rewriter.create<arith::IndexCastOp>(loc, indexType, value);
    if (type.isInteger(32) && indexType.isInteger(64))
      return rewriter.create<arith::ExtSIOp>(loc, indexType, value);
    return Value();
  };
  auto chooseAll = [&](ArrayRef<Value> values) -> bool {
    for (Value value : values) {
      if (!chooseIndexType(value))
        return false;
    }
    return true;
  };

  if (!chooseIndexType(offset) || !chooseAll(strides) || !chooseAll(numels))
    return nullptr;
  offset = castToIndexType(offset);
  if (!offset)
    return nullptr;
  SmallVector<Value> castStrides;
  SmallVector<Value> castNumels;
  castStrides.reserve(rank);
  castNumels.reserve(rank);
  for (Value stride : strides) {
    stride = castToIndexType(stride);
    if (!stride)
      return nullptr;
    castStrides.push_back(stride);
  }
  for (Value numel : numels) {
    numel = castToIndexType(numel);
    if (!numel)
      return nullptr;
    castNumels.push_back(numel);
  }

  auto strideStore = rewriter.create<triton::ascend::StrideStoreOp>(
      loc, dst, src, offset, castStrides, castNumels);
  strideStore->setAttr(RewrittenByStridedLoadStoreRewriteTAG,
                       UnitAttr::get(rewriter.getContext()));
  return strideStore.getOperation();
}

static Value materializeI64(OpFoldResult ofr, Location loc,
                            PatternRewriter &rewriter) {
  if (auto attr = ofr.dyn_cast<Attribute>()) {
    int64_t value = cast<IntegerAttr>(attr).getInt();
    return rewriter.create<arith::ConstantOp>(
        loc, rewriter.getI64IntegerAttr(value));
  }
  return ensureI64Scalar(llvm::cast<Value>(ofr), loc, rewriter);
}

static Value clampI64(Value value, Value lower, Value upper, Location loc,
                      PatternRewriter &rewriter) {
  value = rewriter.create<arith::MaxSIOp>(loc, value, lower);
  return rewriter.create<arith::MinSIOp>(loc, value, upper);
}

static Value getPrefixMaskNumel(Operation *op, Value mask,
                                RankedTensorType resultType,
                                PatternRewriter &rewriter) {
  if (!mask) {
    return rewriter.create<arith::ConstantOp>(
        op->getLoc(),
        rewriter.getI64IntegerAttr(resultType.getShape().front()));
  }

  auto maskState = triton::runMaskAnalysis(op, rewriter);
  if (!maskState || maskState->getRank() != 1)
    return Value();

  auto offset = getConstantIntValue(maskState->offsets.front());
  if (!offset.has_value() || offset.value() != 0)
    return Value();

  auto loc = op->getLoc();
  Value zero =
      rewriter.create<arith::ConstantOp>(loc, rewriter.getI64IntegerAttr(0));
  Value block = rewriter.create<arith::ConstantOp>(
      loc, rewriter.getI64IntegerAttr(resultType.getShape().front()));
  Value numel = materializeI64(maskState->dims.front(), loc, rewriter);
  if (!numel)
    return Value();
  return clampI64(numel, zero, block, loc, rewriter);
}

static FailureOr<SmallVector<Value>>
getAddPtrStrideNumels(Operation *op, Value mask, RankedTensorType resultType,
                      PatternRewriter &rewriter) {
  auto loc = op->getLoc();
  int64_t rank = resultType.getRank();
  ArrayRef<int64_t> shape = resultType.getShape();

  if (!mask) {
    SmallVector<Value> numels;
    for (int64_t d = 0; d < rank; ++d)
      numels.push_back(rewriter.create<arith::ConstantOp>(
          loc, rewriter.getI64IntegerAttr(shape[d])));
    return numels;
  }

  if (rank == 1) {
    Value numel = getPrefixMaskNumel(op, mask, resultType, rewriter);
    if (!numel)
      return failure();
    return SmallVector<Value>{numel};
  }

  auto maskState = triton::runMaskAnalysis(op, rewriter);
  if (!maskState || maskState->getRank() != rank)
    return failure();

  for (int64_t d = 0; d < rank; ++d) {
    auto offsetVal = getConstantIntValue(maskState->offsets[d]);
    if (!offsetVal.has_value() || offsetVal.value() != 0)
      return failure();
  }

  Value zero =
      rewriter.create<arith::ConstantOp>(loc, rewriter.getI64IntegerAttr(0));
  SmallVector<Value> numels;
  numels.reserve(rank);
  for (int64_t d = 0; d < rank; ++d) {
    Value block = rewriter.create<arith::ConstantOp>(
        loc, rewriter.getI64IntegerAttr(shape[d]));
    Value numel = materializeI64(maskState->dims[d], loc, rewriter);
    if (!numel)
      return failure();
    numel = clampI64(numel, zero, block, loc, rewriter);
    numels.push_back(numel);
  }
  return numels;
}

static FailureOr<SmallVector<Value>> getBlockPtrStrideNumels(
    Operation *op, Value mask, ArrayRef<int32_t> boundaryCheck,
    triton::MakeTensorPtrOp mtpt, RankedTensorType resultType,
    ArrayRef<Value> logicalOffsets, PatternRewriter &rewriter) {
  auto loc = op->getLoc();
  Value zero =
      rewriter.create<arith::ConstantOp>(loc, rewriter.getI64IntegerAttr(0));
  ArrayRef<int64_t> shape = resultType.getShape();
  int64_t rank = resultType.getRank();
  SmallVector<Value> numels;
  numels.reserve(rank);

  if (mask && rank != 1)
    return failure();
  Value prefixMaskNumel;
  if (mask) {
    prefixMaskNumel = getPrefixMaskNumel(op, mask, resultType, rewriter);
    if (!prefixMaskNumel)
      return failure();
  }

  for (int64_t d = 0; d < rank; ++d) {
    Value block = rewriter.create<arith::ConstantOp>(
        loc, rewriter.getI64IntegerAttr(shape[d]));
    Value numel = (d == 0 && prefixMaskNumel) ? prefixMaskNumel : block;

    bool hasBoundaryCheck = false;
    for (int32_t axis : boundaryCheck) {
      if (axis == static_cast<int32_t>(d)) {
        hasBoundaryCheck = true;
        break;
      }
    }
    if (hasBoundaryCheck) {
      Value parentSize = ensureI64Scalar(mtpt.getShape()[d], loc, rewriter);
      if (!parentSize)
        return failure();
      Value remaining =
          rewriter.create<arith::SubIOp>(loc, parentSize, logicalOffsets[d]);
      Value boundaryNumel = clampI64(remaining, zero, block, loc, rewriter);
      numel = rewriter.create<arith::MinSIOp>(loc, numel, boundaryNumel);
    }
    numels.push_back(numel);
  }

  return numels;
}

// Expand a 1D tensor `v` of length `targetShape[axis]` into a rank-N tensor
// with size `targetShape[axis]` at `axis` and size 1 elsewhere, then
// broadcast to `targetShape`. Used to materialise the per-axis contribution
// `arange(0, B_d) * stride_d` into the full N-D offset tensor.
static Value expandAndBroadcastForAxis(Value v, int axis,
                                       ArrayRef<int64_t> targetShape,
                                       Type elementType, Location loc,
                                       PatternRewriter &rewriter) {
  Value cur = v;
  int rank = static_cast<int>(targetShape.size());
  for (int d = 0; d < rank; ++d) {
    if (d == axis)
      continue;
    int curRank =
        static_cast<int>(cast<RankedTensorType>(cur.getType()).getRank());
    int expandAt = (d < axis) ? 0 : curRank;
    cur = rewriter.create<triton::ExpandDimsOp>(loc, cur, expandAt);
  }
  auto targetTy = RankedTensorType::get(targetShape, elementType);
  return rewriter.create<triton::BroadcastOp>(loc, targetTy, cur);
}

// Build the per-element OOB mask for block_ptr's boundary_check.
// For each axis d in `boundaryCheck`:
//   idx_d[i] = arange(0, B_d)[i] + effective_offset_d
//   in_bounds_d[i] = (idx_d[i] >= 0) AND (idx_d[i] < parentShape[d])
// Then broadcast each axis's mask to full `blockShape` and AND all together.
// Returns a tensor<...xi1> of shape `blockShape`, or null if boundaryCheck
// is empty.
static Value buildBoundaryMask(Location loc, PatternRewriter &rewriter,
                               ArrayRef<int64_t> blockShape,
                               ArrayRef<int32_t> boundaryCheck,
                               ArrayRef<Value> effectiveOffsetsI32,
                               ValueRange parentShape) {
  if (boundaryCheck.empty())
    return Value();

  auto i32Ty = rewriter.getIntegerType(32);
  auto i64Ty = rewriter.getIntegerType(64);
  auto i1Ty = rewriter.getIntegerType(1);

  Value combined;
  for (int32_t axisRaw : boundaryCheck) {
    int axis = static_cast<int>(axisRaw);
    int64_t blockD = blockShape[axis];

    auto rangeTy32 = RankedTensorType::get({blockD}, i32Ty);
    auto rangeTy64 = RankedTensorType::get({blockD}, i64Ty);

    Value arange = rewriter.create<triton::MakeRangeOp>(
        loc, rangeTy32, /*start=*/0, /*end=*/static_cast<int32_t>(blockD));
    Value offSplat = rewriter.create<triton::SplatOp>(
        loc, rangeTy32, effectiveOffsetsI32[axis]);
    Value idx32 = rewriter.create<arith::AddIOp>(loc, arange, offSplat);
    Value idx64 = rewriter.create<arith::ExtSIOp>(loc, rangeTy64, idx32);

    Value shapeSplat =
        rewriter.create<triton::SplatOp>(loc, rangeTy64, parentShape[axis]);
    Value zeroDense = rewriter.create<arith::ConstantOp>(
        loc, DenseElementsAttr::get(rangeTy64, llvm::APInt(64, 0)));
    Value cmpLower = rewriter.create<arith::CmpIOp>(
        loc, arith::CmpIPredicate::sge, idx64, zeroDense);
    Value cmpUpper = rewriter.create<arith::CmpIOp>(
        loc, arith::CmpIPredicate::slt, idx64, shapeSplat);
    Value axisMask = rewriter.create<arith::AndIOp>(loc, cmpLower, cmpUpper);

    Value bcast = expandAndBroadcastForAxis(axisMask, axis, blockShape, i1Ty,
                                            loc, rewriter);
    combined =
        combined
            ? rewriter.create<arith::AndIOp>(loc, combined, bcast).getResult()
            : bcast;
  }
  return combined;
}

// Build the padding "other" tensor for tt.indirect_load. Honours
// PaddingOption::PAD_NAN for float element types; otherwise (including
// PAD_ZERO or unspecified) returns a zero-splat tensor of `resultType`.
// Returns null if PAD_NAN is requested but the element type is not float
// (caller should treat as a hard failure).
static Value buildPaddingOther(Location loc, PatternRewriter &rewriter,
                               RankedTensorType resultType,
                               std::optional<triton::PaddingOption> padding) {
  Type elementType = resultType.getElementType();
  Value padScalar;
  if (padding.has_value() &&
      padding.value() == triton::PaddingOption::PAD_NAN) {
    auto floatTy = dyn_cast<FloatType>(elementType);
    if (!floatTy)
      return Value();
    auto nan = llvm::APFloat::getNaN(floatTy.getFloatSemantics());
    padScalar = rewriter.create<arith::ConstantOp>(
        loc, rewriter.getFloatAttr(elementType, nan));
  } else {
    padScalar = rewriter.create<arith::ConstantOp>(
        loc, rewriter.getZeroAttr(elementType));
  }
  return rewriter.create<triton::SplatOp>(loc, resultType, padScalar);
}

// AddPtr path: tt.load(tt.addptr(tt.splat(%scalar_ptr), %offsets)).
// Uses PtrAnalysis (which has IR side effects), so this function must
// stamp InspectedByStridedLoadStoreRewriteTAG on every "PtrAnalysis ran but
// don't rewrite" path -- see comment block inside.
static LogicalResult tryRewriteAddPtrLoad(triton::LoadOp op,
                                          triton::AddPtrOp addPtrOp,
                                          RankedTensorType resultType,
                                          PatternRewriter &rewriter) {
  auto loc = op.getLoc();

  // The base must resolve to a scalar pointer through shape-only wrappers.
  Value scalarBase = getScalarBasePtr(addPtrOp.getPtr());
  if (!scalarBase)
    return failure();

  // Pre-filter without mutating IR: if no per-element multiplication by a
  // constant > 1 exists in the offset chain, last stride must be 1.
  if (!offsetMayContainStrideGtOne(addPtrOp.getOffset())) {
    return failure();
  }

  // From here, PtrAnalysis may insert helper IR. Every early-out path
  // MUST stamp InspectedByStridedLoadStoreRewriteTAG and return success() so
  // the greedy driver does not re-walk the same op (which would re-run
  // PtrAnalysis and accumulate dead IR until maxIterations).
  TritonToStructured::PtrAnalysis ptrAnalysis;
  TritonToStructured::PtrState ptrState;
  auto markInspectedAndReturn = [&]() {
    op->setAttr(InspectedByStridedLoadStoreRewriteTAG,
                UnitAttr::get(rewriter.getContext()));
    return success();
  };
  if (ptrAnalysis.visitOperand(op.getPtr(), ptrState, loc, rewriter).failed())
    return markInspectedAndReturn();
  if (ptrState.stateInfo.empty())
    return markInspectedAndReturn();
  ptrState.analyzePermute();
  if (ptrState.isPermuted)
    return markInspectedAndReturn();

  // Use SIMT indirect only for static non-pow2 or masked single-tile pow2
  // strides; keep dynamic strides on the structured SIMD path.
  auto lastStrideOpt = getConstantIntValue(ptrState.stateInfo.back().stride);
  if (!lastStrideOpt.has_value())
    return markInspectedAndReturn();
  int64_t lastStride = std::abs(lastStrideOpt.value());
  if (lastStride <= 1)
    return markInspectedAndReturn();
  bool routeMaskedPow2ToIndirect =
      shouldRouteMaskedSingleTilePow2ToIndirect(op.getMask(), resultType);
  if (lastStride == 2 && !routeMaskedPow2ToIndirect)
    return markInspectedAndReturn(); // even -> deinterleave; odd -> strided DMA
  if ((lastStride & (lastStride - 1)) == 0 && !routeMaskedPow2ToIndirect)
    return markInspectedAndReturn(); // power-of-two >= 4 -> strided DMA

  bool useStrideLoad = !routeMaskedPow2ToIndirect;
  if (useStrideLoad && resultType.getRank() >= 1 && resultType.getRank() <= 3 &&
      resultType.hasStaticShape() &&
      static_cast<int64_t>(ptrState.stateInfo.size()) == resultType.getRank()) {
    Value src = ptrState.source;
    SmallVector<Value> strides;
    SmallVector<Value> numels;
    strides.reserve(resultType.getRank());

    Value baseOffset = materializeI64(ptrState.offset, loc, rewriter);
    bool operandsReady = src && baseOffset;
    for (int64_t d = 0; operandsReady && d < resultType.getRank(); ++d) {
      Value stride =
          materializeI64(ptrState.stateInfo[d].stride, loc, rewriter);
      if (!stride) {
        operandsReady = false;
        break;
      }
      strides.push_back(stride);
    }

    if (operandsReady) {
      auto numelsResult = getAddPtrStrideNumels(op.getOperation(), op.getMask(),
                                                resultType, rewriter);
      if (succeeded(numelsResult)) {
        numels = *numelsResult;
      } else {
        operandsReady = false;
      }
    }
    Value other = operandsReady
                      ? getStrideLoadOtherScalar(op, resultType, rewriter)
                      : Value();
    operandsReady = operandsReady && other;

    if (operandsReady) {
      Value strideLoadResult = createStrideLoadOp(
          loc, resultType, src, baseOffset, other, strides, numels, rewriter);
      if (!strideLoadResult)
        return markInspectedAndReturn();

      logMatch(loc, "AddPtr", "tt.load -> ttasc.stride_load", lastStride);
      rewriter.replaceOp(op, strideLoadResult);
      return success();
    }
  }

  if (useStrideLoad && resultType.getRank() <= 3)
    return markInspectedAndReturn();

  Value offsetTensor =
      ensureI64OffsetTensor(addPtrOp.getOffset(), loc, rewriter);
  if (!offsetTensor)
    return failure();

  Value src;
  Value scalarOffset;
  if (failed(unwrapScalarAddPtrChain(scalarBase, src, scalarOffset, loc,
                                     rewriter)))
    return markInspectedAndReturn();
  offsetTensor =
      addScalarOffsetToTensor(offsetTensor, scalarOffset, loc, rewriter);

  auto indirectLoad = rewriter.create<triton::ascend::IndirectLoadOp>(
      loc, resultType, src, offsetTensor, op.getMask(), op.getOther(),
      ConverterUtils::requiresVolatileIndirectLoad(op.getPtr(), op));
  indirectLoad->setAttr(RewrittenByStridedLoadStoreRewriteTAG,
                        UnitAttr::get(rewriter.getContext()));

  logMatch(loc, "AddPtr", "tt.load -> tt.indirect_load", lastStride);
  rewriter.replaceOp(op, indirectLoad.getResult());
  return success();
}

// Block-ptr path: tt.load(tt.make_tensor_ptr ...) or tt.load(tt.advance ...).
// Unlike the AddPtr path this does NOT use PtrAnalysis -- strides come
// directly from mtpt.getStrides() / order from mtpt.getOrder() -- so we
// can decide everything before touching IR and just return failure() if
// we don't want to rewrite.
static LogicalResult tryRewriteBlockPtrLoad(triton::LoadOp op,
                                            triton::MakeTensorPtrOp mtpt,
                                            triton::AdvanceOp advance,
                                            RankedTensorType resultType,
                                            PatternRewriter &rewriter) {
  auto loc = op.getLoc();
  auto i64Ty = rewriter.getIntegerType(64);
  ArrayRef<int64_t> shape = resultType.getShape();
  int64_t rank = static_cast<int64_t>(shape.size());

  // ---- order must match the "non-permuted" layout: order[i] == rank-1-i,
  //      i.e. innermost (fastest-changing) is the last dim of the tensor.
  //      ImplicitPermute handles permuted layouts via tt.trans, so we
  //      leave anything non-canonical alone.
  auto order = mtpt.getOrder();
  if (static_cast<int64_t>(order.size()) != rank)
    return failure();
  for (int64_t i = 0; i < rank; ++i) {
    if (order[i] != rank - 1 - i) {
      return failure();
    }
  }

  // ---- stride check ----
  auto strides = mtpt.getStrides();
  if (strides.empty() || static_cast<int64_t>(strides.size()) != rank)
    return failure();
  // Stride dispatch: strided DMA on the MTE engine only supports power-of-two
  // strides; a non-power-of-two stride would degrade to a slow scalar access.
  // Dynamic strides stay on the structured SIMD path because they may be
  // runtime stride 1 or power-of-two, where SIMT stride is slower. So we
  // only rewrite to SIMT stride for *static non-power-of-two* strides:
  //   stride 1 -> contiguous; stride 2 (even dim) -> deinterleave;
  //   stride >= 4 (power of two) -> (compact) strided DMA.
  APInt lastStrideC;
  if (!matchPattern(strides.back(), m_ConstantInt(&lastStrideC)))
    return failure();
  int64_t lastStride = std::abs(lastStrideC.getSExtValue());
  if (lastStride <= 1)
    return failure();
  if (lastStride == 2)
    return failure(); // even -> deinterleave; odd -> strided DMA
  if ((lastStride & (lastStride - 1)) == 0)
    return failure(); // power-of-two >= 4 -> strided DMA

  // ---- Compute per-axis effective base offsets: mtpt.offsets[d] +
  // (advance.offsets[d] if present)
  ValueRange mtptOffsets = mtpt.getOffsets();
  ValueRange advOffsets = advance ? advance.getOffsets() : ValueRange{};
  if (static_cast<int64_t>(mtptOffsets.size()) != rank)
    return failure();
  if (advance && static_cast<int64_t>(advOffsets.size()) != rank)
    return failure();

  // ---- All checks passed: from here on IR mutation is committed. ----

  // Unwrap any `tt.addptr` chain on the SCALAR base ptr (e.g. when the
  // kernel writes `tl.make_block_ptr(s + bos*H + i_h, ...)`). If we left
  // those scalar AddPtrs in place, the AddPtrConverter would lower each
  // into a `memref.reinterpret_cast ... sizes: [1]` single-element view. By
  // walking the scalar AddPtr chain here we (a) fold its scalar offsets into
  // `scalarBaseAdj` and (b) recover the original underlying `!tt.ptr<T>` to
  // use as our src.
  Value src = mtpt.getBase();
  Value scalarBaseAdj =
      rewriter.create<arith::ConstantOp>(loc, rewriter.getI64IntegerAttr(0));
  while (auto addptr = src.getDefiningOp<triton::AddPtrOp>()) {
    // Only unwrap scalar AddPtr chains -- tensor-of-ptrs AddPtrs are not
    // expected here (mtpt.getBase() is always a scalar ptr).
    if (isa<RankedTensorType>(addptr.getPtr().getType()))
      break;
    Value off = addptr.getOffset();
    Value offI64 = ensureI64Scalar(off, loc, rewriter);
    if (!offI64)
      return failure();
    scalarBaseAdj = rewriter.create<arith::AddIOp>(loc, scalarBaseAdj, offI64);
    src = addptr.getPtr();
  }

  // Build the scalar base offset:
  //   scalarBaseAdj + sum_d (mtpt.offsets[d] + adv.offsets[d]) * strides[d].
  // Boundary checks are expressed in logical make_tensor_ptr coordinates, so
  // numels must use the unstrided per-axis offsets rather than this physical
  // GM offset.
  Value scalarBase = scalarBaseAdj;
  SmallVector<Value> logicalOffsets;
  SmallVector<Value> strideOperands;
  logicalOffsets.reserve(rank);
  strideOperands.reserve(rank);
  for (int64_t d = 0; d < rank; ++d) {
    Value baseOff = ensureI64Scalar(mtptOffsets[d], loc, rewriter);
    if (!baseOff)
      return failure();
    if (advance) {
      Value advStep = ensureI64Scalar(advOffsets[d], loc, rewriter);
      if (!advStep)
        return failure();
      baseOff = rewriter.create<arith::AddIOp>(loc, baseOff, advStep);
    }
    logicalOffsets.push_back(baseOff);
    Value strI64 = ensureI64Scalar(strides[d], loc, rewriter);
    if (!strI64)
      return failure();
    strideOperands.push_back(strI64);
    Value prod = rewriter.create<arith::MulIOp>(loc, baseOff, strI64);
    scalarBase = rewriter.create<arith::AddIOp>(loc, scalarBase, prod);
  }

  if (resultType.getRank() >= 1 && resultType.getRank() <= 3 &&
      resultType.hasStaticShape()) {
    FailureOr<SmallVector<Value>> numels = getBlockPtrStrideNumels(
        op.getOperation(), op.getMask(), op.getBoundaryCheck(), mtpt,
        resultType, logicalOffsets, rewriter);
    if (succeeded(numels)) {
      Value other = getStrideLoadOtherScalar(op, resultType, rewriter);
      if (!other)
        return failure();
      Value strideLoadResult =
          createStrideLoadOp(loc, resultType, src, scalarBase, other,
                             strideOperands, *numels, rewriter);
      if (!strideLoadResult)
        return failure();

      logMatch(loc, llvm::Twine("BlockPtr") + (advance ? "+Advance" : ""),
               "tt.load -> ttasc.stride_load", lastStride);
      rewriter.replaceOp(op, strideLoadResult);
      return success();
    }
  }

  if (rank <= 3)
    return failure();

  auto i64TensorTy = RankedTensorType::get(shape, i64Ty);
  Value offsetTensor =
      rewriter.create<triton::SplatOp>(loc, i64TensorTy, scalarBase);
  for (int64_t d = 0; d < rank; ++d) {
    auto axisTy = RankedTensorType::get({shape[d]}, rewriter.getI32Type());
    Value arange = rewriter.create<triton::MakeRangeOp>(
        loc, axisTy, /*start=*/0, /*end=*/static_cast<int32_t>(shape[d]));
    Value arangeI64 = rewriter.create<arith::ExtSIOp>(
        loc, RankedTensorType::get({shape[d]}, i64Ty), arange);
    Value strI64 = ensureI64Scalar(strides[d], loc, rewriter);
    Value strSplat = rewriter.create<triton::SplatOp>(
        loc, RankedTensorType::get({shape[d]}, i64Ty), strI64);
    Value stridedArange =
        rewriter.create<arith::MulIOp>(loc, arangeI64, strSplat);
    Value broadcasted = expandAndBroadcastForAxis(
        stridedArange, static_cast<int>(d), shape, i64Ty, loc, rewriter);
    offsetTensor =
        rewriter.create<arith::AddIOp>(loc, offsetTensor, broadcasted);
  }

  // ---- boundary_check: build OOB mask + padding "other" ----
  Value mask = op.getMask();
  Value other = op.getOther();
  auto boundaryCheck = op.getBoundaryCheck();
  if (!boundaryCheck.empty()) {
    // Effective offset per axis = mtpt.offsets[d] + (advance.offsets[d] if any)
    SmallVector<Value> effOffsets;
    for (int64_t d = 0; d < rank; ++d) {
      Value off = mtptOffsets[d];
      if (advance) {
        off = rewriter.create<arith::AddIOp>(loc, off, advOffsets[d]);
      }
      effOffsets.push_back(off);
    }
    Value boundaryMask = buildBoundaryMask(loc, rewriter, shape, boundaryCheck,
                                           effOffsets, mtpt.getShape());
    mask = mask ? rewriter.create<arith::AndIOp>(loc, mask, boundaryMask)
                      .getResult()
                : boundaryMask;
    if (!other) {
      other = buildPaddingOther(loc, rewriter, resultType, op.getPadding());
      if (!other) {
        // PAD_NAN requested on non-float element type: bail to legacy
        // path (which would also assert there).
        return failure();
      }
    }
  }

  // ---- Emit tt.indirect_load ----
  auto indirectLoad = rewriter.create<triton::ascend::IndirectLoadOp>(
      loc, resultType, src, offsetTensor, mask, other,
      ConverterUtils::requiresVolatileIndirectLoad(op.getPtr(), op));
  indirectLoad->setAttr(RewrittenByStridedLoadStoreRewriteTAG,
                        UnitAttr::get(rewriter.getContext()));

  logMatch(loc,
           llvm::Twine("BlockPtr") + (advance ? "+Advance" : "") +
               (boundaryCheck.empty() ? "" : "+Boundary"),
           "tt.load -> tt.indirect_load", lastStride);
  rewriter.replaceOp(op, indirectLoad.getResult());
  return success();
}

// V2 (Store) helpers ----------------------------------------------------------

// AddPtr path for tt.store. Mirrors tryRewriteAddPtrLoad but emits
// triton::ascend::StrideStoreOp when the mask can be represented by per-axis
// numel bounds.
static LogicalResult tryRewriteAddPtrStore(triton::StoreOp op,
                                           triton::AddPtrOp addPtrOp,
                                           RankedTensorType valueType,
                                           PatternRewriter &rewriter) {
  auto loc = op.getLoc();

  Value scalarBase = getScalarBasePtr(addPtrOp.getPtr());
  if (!scalarBase)
    return failure();

  if (!offsetMayContainStrideGtOne(addPtrOp.getOffset()))
    return failure();

  TritonToStructured::PtrAnalysis ptrAnalysis;
  TritonToStructured::PtrState ptrState;
  auto markInspectedAndReturn = [&]() {
    op->setAttr(InspectedByStridedLoadStoreRewriteTAG,
                UnitAttr::get(rewriter.getContext()));
    return success();
  };
  if (ptrAnalysis.visitOperand(op.getPtr(), ptrState, loc, rewriter).failed())
    return markInspectedAndReturn();
  if (ptrState.stateInfo.empty())
    return markInspectedAndReturn();
  ptrState.analyzePermute();
  if (ptrState.isPermuted)
    return markInspectedAndReturn();

  // Use SIMT indirect only for static non-pow2 or masked single-tile pow2
  // strides; keep dynamic strides on the structured SIMD path.
  auto lastStrideOpt = getConstantIntValue(ptrState.stateInfo.back().stride);
  if (!lastStrideOpt.has_value())
    return markInspectedAndReturn();
  int64_t lastStride = std::abs(lastStrideOpt.value());
  if (lastStride <= 1)
    return markInspectedAndReturn();
  bool routeMaskedPow2ToIndirect =
      shouldRouteMaskedSingleTilePow2ToIndirect(op.getMask(), valueType);
  if (lastStride == 2 && !routeMaskedPow2ToIndirect)
    return markInspectedAndReturn(); // even -> deinterleave; odd -> strided DMA
  if ((lastStride & (lastStride - 1)) == 0 && !routeMaskedPow2ToIndirect)
    return markInspectedAndReturn(); // power-of-two >= 4 -> strided DMA

  bool useStrideStore = !routeMaskedPow2ToIndirect;
  if (useStrideStore && valueType.getRank() >= 1 && valueType.getRank() <= 3 &&
      valueType.hasStaticShape() &&
      static_cast<int64_t>(ptrState.stateInfo.size()) == valueType.getRank()) {
    Value dst = ptrState.source;
    SmallVector<Value> strides;
    SmallVector<Value> numels;
    strides.reserve(valueType.getRank());

    Value baseOffset = materializeI64(ptrState.offset, loc, rewriter);
    bool operandsReady = dst && baseOffset;
    for (int64_t d = 0; operandsReady && d < valueType.getRank(); ++d) {
      Value stride =
          materializeI64(ptrState.stateInfo[d].stride, loc, rewriter);
      if (!stride) {
        operandsReady = false;
        break;
      }
      strides.push_back(stride);
    }

    if (operandsReady) {
      auto numelsResult = getAddPtrStrideNumels(op.getOperation(), op.getMask(),
                                                valueType, rewriter);
      if (succeeded(numelsResult)) {
        numels = *numelsResult;
      } else {
        operandsReady = false;
      }
    }

    if (operandsReady) {
      Operation *strideStore =
          createStrideStoreOp(loc, valueType, dst, op.getValue(), baseOffset,
                              strides, numels, rewriter);
      if (!strideStore)
        return markInspectedAndReturn();

      logMatch(loc, "AddPtr", "tt.store -> ttasc.stride_store", lastStride);
      rewriter.eraseOp(op);
      return success();
    }
  }

  if (useStrideStore && valueType.getRank() <= 3)
    return markInspectedAndReturn();

  Value offsetTensor =
      ensureI64OffsetTensor(addPtrOp.getOffset(), loc, rewriter);
  if (!offsetTensor)
    return failure();

  Value src;
  Value scalarOffset;
  if (failed(unwrapScalarAddPtrChain(scalarBase, src, scalarOffset, loc,
                                     rewriter)))
    return markInspectedAndReturn();
  offsetTensor =
      addScalarOffsetToTensor(offsetTensor, scalarOffset, loc, rewriter);

  auto indirectStore = rewriter.create<triton::ascend::IndirectStoreOp>(
      loc, src, offsetTensor, op.getValue(), op.getMask());
  indirectStore->setAttr(RewrittenByStridedLoadStoreRewriteTAG,
                         UnitAttr::get(rewriter.getContext()));

  logMatch(loc, "AddPtr", "tt.store -> tt.indirect_store", lastStride);
  rewriter.eraseOp(op);
  return success();
}

// Block-ptr path for tt.store. Mirrors tryRewriteBlockPtrLoad.
static LogicalResult tryRewriteBlockPtrStore(triton::StoreOp op,
                                             triton::MakeTensorPtrOp mtpt,
                                             triton::AdvanceOp advance,
                                             RankedTensorType valueType,
                                             PatternRewriter &rewriter) {
  auto loc = op.getLoc();
  auto i64Ty = rewriter.getIntegerType(64);
  ArrayRef<int64_t> shape = valueType.getShape();
  int64_t rank = static_cast<int64_t>(shape.size());

  auto order = mtpt.getOrder();
  if (static_cast<int64_t>(order.size()) != rank)
    return failure();
  for (int64_t i = 0; i < rank; ++i) {
    if (order[i] != rank - 1 - i)
      return failure();
  }

  auto strides = mtpt.getStrides();
  if (strides.empty() || static_cast<int64_t>(strides.size()) != rank)
    return failure();
  // Stride dispatch (mirrors tryRewriteBlockPtrLoad): dynamic strides stay on
  // the structured SIMD path; only static non-power-of-two strides fall
  // through to SIMT stride_store.
  APInt lastStrideC;
  if (!matchPattern(strides.back(), m_ConstantInt(&lastStrideC)))
    return failure();
  int64_t lastStride = std::abs(lastStrideC.getSExtValue());
  if (lastStride <= 1)
    return failure();
  if (lastStride == 2)
    return failure(); // even -> deinterleave; odd -> strided DMA
  if ((lastStride & (lastStride - 1)) == 0)
    return failure(); // power-of-two >= 4 -> strided DMA

  ValueRange mtptOffsets = mtpt.getOffsets();
  ValueRange advOffsets = advance ? advance.getOffsets() : ValueRange{};
  if (static_cast<int64_t>(mtptOffsets.size()) != rank)
    return failure();
  if (advance && static_cast<int64_t>(advOffsets.size()) != rank)
    return failure();

  // ---- All checks passed: from here on IR mutation is committed. ----

  // Unwrap scalar AddPtr chain on the base (see comment in
  // tryRewriteBlockPtrLoad for why this is required to avoid lowering to a
  // size-1 reinterpret_cast view).
  Value src = mtpt.getBase();
  Value scalarBaseAdj =
      rewriter.create<arith::ConstantOp>(loc, rewriter.getI64IntegerAttr(0));
  while (auto addptr = src.getDefiningOp<triton::AddPtrOp>()) {
    if (isa<RankedTensorType>(addptr.getPtr().getType()))
      break;
    Value off = addptr.getOffset();
    Value offI64 = ensureI64Scalar(off, loc, rewriter);
    if (!offI64)
      return failure();
    scalarBaseAdj = rewriter.create<arith::AddIOp>(loc, scalarBaseAdj, offI64);
    src = addptr.getPtr();
  }

  Value scalarBase = scalarBaseAdj;
  SmallVector<Value> logicalOffsets;
  SmallVector<Value> strideOperands;
  logicalOffsets.reserve(rank);
  strideOperands.reserve(rank);
  for (int64_t d = 0; d < rank; ++d) {
    Value baseOff = ensureI64Scalar(mtptOffsets[d], loc, rewriter);
    if (!baseOff)
      return failure();
    if (advance) {
      Value advStep = ensureI64Scalar(advOffsets[d], loc, rewriter);
      if (!advStep)
        return failure();
      baseOff = rewriter.create<arith::AddIOp>(loc, baseOff, advStep);
    }
    logicalOffsets.push_back(baseOff);
    Value strI64 = ensureI64Scalar(strides[d], loc, rewriter);
    if (!strI64)
      return failure();
    strideOperands.push_back(strI64);
    Value prod = rewriter.create<arith::MulIOp>(loc, baseOff, strI64);
    scalarBase = rewriter.create<arith::AddIOp>(loc, scalarBase, prod);
  }

  auto boundaryCheck = op.getBoundaryCheck();
  if (valueType.getRank() >= 1 && valueType.getRank() <= 3 &&
      valueType.hasStaticShape()) {
    FailureOr<SmallVector<Value>> numels =
        getBlockPtrStrideNumels(op.getOperation(), op.getMask(), boundaryCheck,
                                mtpt, valueType, logicalOffsets, rewriter);
    if (succeeded(numels)) {
      Operation *strideStore =
          createStrideStoreOp(loc, valueType, src, op.getValue(), scalarBase,
                              strideOperands, *numels, rewriter);
      if (strideStore) {
        logMatch(loc,
                 llvm::Twine("BlockPtr") + (advance ? "+Advance" : "") +
                     (boundaryCheck.empty() ? "" : "+Boundary"),
                 "tt.store -> ttasc.stride_store", lastStride);
        rewriter.eraseOp(op);
        return success();
      }
    }
  }

  if (rank <= 3)
    return failure();

  auto i64TensorTy = RankedTensorType::get(shape, i64Ty);
  Value offsetTensor =
      rewriter.create<triton::SplatOp>(loc, i64TensorTy, scalarBase);
  for (int64_t d = 0; d < rank; ++d) {
    auto axisTy = RankedTensorType::get({shape[d]}, rewriter.getI32Type());
    Value arange = rewriter.create<triton::MakeRangeOp>(
        loc, axisTy, /*start=*/0, /*end=*/static_cast<int32_t>(shape[d]));
    Value arangeI64 = rewriter.create<arith::ExtSIOp>(
        loc, RankedTensorType::get({shape[d]}, i64Ty), arange);
    Value strI64 = ensureI64Scalar(strides[d], loc, rewriter);
    Value strSplat = rewriter.create<triton::SplatOp>(
        loc, RankedTensorType::get({shape[d]}, i64Ty), strI64);
    Value stridedArange =
        rewriter.create<arith::MulIOp>(loc, arangeI64, strSplat);
    Value broadcasted = expandAndBroadcastForAxis(
        stridedArange, static_cast<int>(d), shape, i64Ty, loc, rewriter);
    offsetTensor =
        rewriter.create<arith::AddIOp>(loc, offsetTensor, broadcasted);
  }

  // ---- boundary_check: build OOB mask (store has no "other") ----
  Value mask = op.getMask();
  if (!boundaryCheck.empty()) {
    SmallVector<Value> effOffsets;
    for (int64_t d = 0; d < rank; ++d) {
      Value off = mtptOffsets[d];
      if (advance) {
        off = rewriter.create<arith::AddIOp>(loc, off, advOffsets[d]);
      }
      effOffsets.push_back(off);
    }
    Value boundaryMask = buildBoundaryMask(loc, rewriter, shape, boundaryCheck,
                                           effOffsets, mtpt.getShape());
    mask = mask ? rewriter.create<arith::AndIOp>(loc, mask, boundaryMask)
                      .getResult()
                : boundaryMask;
  }

  auto indirectStore = rewriter.create<triton::ascend::IndirectStoreOp>(
      loc, src, offsetTensor, op.getValue(), mask);
  indirectStore->setAttr(RewrittenByStridedLoadStoreRewriteTAG,
                         UnitAttr::get(rewriter.getContext()));

  logMatch(loc,
           llvm::Twine("BlockPtr") + (advance ? "+Advance" : "") +
               (boundaryCheck.empty() ? "" : "+Boundary"),
           "tt.store -> tt.indirect_store", lastStride);
  rewriter.eraseOp(op);
  return success();
}

} // namespace

LogicalResult LoadConverter::matchAndRewrite(triton::LoadOp op,
                                             PatternRewriter &rewriter) const {
  auto loc = op.getLoc();
  (void)loc;

  // ---- Re-entry / cross-step guards ----
  if (op->hasAttr(InspectedByStridedLoadStoreRewriteTAG))
    return failure();
  if (op->hasAttr(RewrittenByStridedLoadStoreRewriteTAG))
    return failure();
  if (op->hasAttr(mlir::triton::memory_access::ImplicitPermuteHandledTAG))
    return failure();
  if (op->hasAttr(mlir::ConverterUtils::discreteAttrName))
    return failure();

  // ---- Common early checks (apply to both AddPtr and block-ptr paths) ----
  // boundary_check is legal only on make_tensor_ptr loads; block-ptr handler
  // builds an OOB mask + padding "other" for it. AddPtr loads should never
  // have boundary_check (defensive bail kept in tryRewriteAddPtrLoad below).
  auto resultType = dyn_cast<RankedTensorType>(op.getResult().getType());
  if (!resultType)
    return failure();
  if (resultType.getShape().size() > kFastPathRankLimit)
    return failure();

  // ---- Dispatch on source op ----
  Value ptr = op.getPtr();
  if (auto addPtrOp = ptr.getDefiningOp<triton::AddPtrOp>()) {
    if (!op.getBoundaryCheck().empty())
      return failure(); // defensive
    return tryRewriteAddPtrLoad(op, addPtrOp, resultType, rewriter);
  }
  if (auto mtptOp = ptr.getDefiningOp<triton::MakeTensorPtrOp>()) {
    return tryRewriteBlockPtrLoad(op, mtptOp, /*advance=*/nullptr, resultType,
                                  rewriter);
  }
  if (auto advOp = ptr.getDefiningOp<triton::AdvanceOp>()) {
    // V1: one-level advance only. Nested advance / scf.for iter-arg block
    // ptr falls through to the legacy strided memref.copy path.
    if (auto baseMtpt =
            advOp.getPtr().getDefiningOp<triton::MakeTensorPtrOp>()) {
      return tryRewriteBlockPtrLoad(op, baseMtpt, advOp, resultType, rewriter);
    }
    return failure();
  }
  return failure();
}

LogicalResult StoreConverter::matchAndRewrite(triton::StoreOp op,
                                              PatternRewriter &rewriter) const {
  // ---- Re-entry / cross-step guards (same convention as LoadConverter) ----
  if (op->hasAttr(InspectedByStridedLoadStoreRewriteTAG))
    return failure();
  if (op->hasAttr(RewrittenByStridedLoadStoreRewriteTAG))
    return failure();
  if (op->hasAttr(mlir::triton::memory_access::ImplicitPermuteHandledTAG))
    return failure();
  if (op->hasAttr(mlir::ConverterUtils::discreteAttrName))
    return failure();

  // boundary_check is legal only on make_tensor_ptr stores; block-ptr handler
  // builds an OOB mask for it. AddPtr stores should never have boundary_check
  // (defensive bail).
  auto valueType = dyn_cast<RankedTensorType>(op.getValue().getType());
  if (!valueType)
    return failure();
  if (valueType.getShape().size() > kFastPathRankLimit)
    return failure();

  Value ptr = op.getPtr();
  if (auto addPtrOp = ptr.getDefiningOp<triton::AddPtrOp>()) {
    if (!op.getBoundaryCheck().empty())
      return failure(); // defensive
    return tryRewriteAddPtrStore(op, addPtrOp, valueType, rewriter);
  }
  if (auto mtptOp = ptr.getDefiningOp<triton::MakeTensorPtrOp>()) {
    return tryRewriteBlockPtrStore(op, mtptOp, /*advance=*/nullptr, valueType,
                                   rewriter);
  }
  if (auto advOp = ptr.getDefiningOp<triton::AdvanceOp>()) {
    if (auto baseMtpt =
            advOp.getPtr().getDefiningOp<triton::MakeTensorPtrOp>()) {
      return tryRewriteBlockPtrStore(op, baseMtpt, advOp, valueType, rewriter);
    }
    return failure();
  }
  return failure();
}

} // namespace StridedLoadStoreRewrite
