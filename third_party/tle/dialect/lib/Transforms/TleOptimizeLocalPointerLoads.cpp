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

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/Triton/IR/Types.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"

#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include <limits>
#include <optional>

namespace mlir::triton::tle {

#define GEN_PASS_DEF_TRITONTLEOPTIMIZELOCALPOINTERLOADS
#include "tle/dialect/include/Transforms/Passes.h.inc"

namespace {

namespace ttg = mlir::triton::gpu;

constexpr int kSharedMemoryAddressSpace = 3;

struct RematerializedValue {
  Value value;
  bool usesLocalPointerLoad = false;
};

struct RematerializationCacheEntry {
  Value source;
  Type targetType;
  RematerializedValue rematerialized;
};

static Value stripConvertLayouts(Value value) {
  Value current = value;
  while (auto cvt = current.getDefiningOp<ttg::ConvertLayoutOp>())
    current = cvt.getSrc();
  return current;
}

static Value stripIndexValueWrappers(Value value) {
  Value current = value;
  while (true) {
    if (auto cvt = current.getDefiningOp<ttg::ConvertLayoutOp>()) {
      current = cvt.getSrc();
      continue;
    }
    if (auto ext = current.getDefiningOp<arith::ExtSIOp>()) {
      current = ext.getIn();
      continue;
    }
    if (auto ext = current.getDefiningOp<arith::ExtUIOp>()) {
      current = ext.getIn();
      continue;
    }
    if (auto trunc = current.getDefiningOp<arith::TruncIOp>()) {
      current = trunc.getIn();
      continue;
    }
    if (auto cast = current.getDefiningOp<arith::IndexCastOp>()) {
      current = cast.getIn();
      continue;
    }
    break;
  }
  return current;
}

static std::optional<int64_t> getConstantIntLike(Value value) {
  Value current = stripIndexValueWrappers(value);
  if (auto splat = current.getDefiningOp<triton::SplatOp>())
    return getConstantIntLike(splat.getSrc());
  if (auto cst = current.getDefiningOp<arith::ConstantOp>()) {
    if (auto dense = dyn_cast<DenseIntElementsAttr>(cst.getValue())) {
      if (dense.isSplat())
        return dense.getSplatValue<APInt>().getSExtValue();
    }
  }
  if (auto cst = current.getDefiningOp<arith::ConstantIntOp>())
    return cst.value();
  if (auto cst = current.getDefiningOp<arith::ConstantIndexOp>())
    return cst.value();
  return std::nullopt;
}

static bool matchRangeWithStaticOffset(Value value, int64_t extent,
                                       int64_t &offset) {
  Value current = stripIndexValueWrappers(value);
  if (auto range = current.getDefiningOp<triton::MakeRangeOp>()) {
    offset = range.getStart();
    return range.getEnd() - range.getStart() == extent;
  }

  auto add = current.getDefiningOp<arith::AddIOp>();
  if (!add)
    return false;

  auto tryMatch = [&](Value lhs, Value rhs) -> bool {
    Value lhsStripped = stripIndexValueWrappers(lhs);
    auto range = lhsStripped.getDefiningOp<triton::MakeRangeOp>();
    if (!range)
      return false;
    std::optional<int64_t> cst = getConstantIntLike(rhs);
    if (!cst)
      return false;
    offset = range.getStart() + *cst;
    return range.getEnd() - range.getStart() == extent;
  };

  return tryMatch(add.getLhs(), add.getRhs()) ||
         tryMatch(add.getRhs(), add.getLhs());
}

static bool matchFullIndexTensorForAxis(Value index, size_t axis,
                                        ArrayRef<int64_t> shape,
                                        int64_t &offset) {
  auto indexTy = dyn_cast<RankedTensorType>(index.getType());
  if (!indexTy || !indexTy.getElementType().isInteger())
    return false;
  if (indexTy.getShape() != shape)
    return false;

  Value current = stripIndexValueWrappers(index);
  if (shape.size() == 1)
    return matchRangeWithStaticOffset(current, shape.front(), offset);

  auto bcast = current.getDefiningOp<triton::BroadcastOp>();
  if (!bcast)
    return false;

  auto bcastSrcTy = dyn_cast<RankedTensorType>(bcast.getSrc().getType());
  if (!bcastSrcTy || bcastSrcTy.getRank() != static_cast<int64_t>(shape.size()))
    return false;
  for (auto [dim, dimSize] : llvm::enumerate(shape)) {
    const int64_t expected = dim == axis ? dimSize : 1;
    if (bcastSrcTy.getShape()[dim] != expected)
      return false;
  }

  current = stripIndexValueWrappers(bcast.getSrc());
  while (auto expand = current.getDefiningOp<triton::ExpandDimsOp>())
    current = stripIndexValueWrappers(expand.getSrc());

  auto rangeTy = dyn_cast<RankedTensorType>(current.getType());
  if (!rangeTy || rangeTy.getRank() != 1)
    return false;
  if (rangeTy.getShape()[0] != shape[axis])
    return false;

  return matchRangeWithStaticOffset(current, shape[axis], offset);
}

struct StaticSubviewMatch {
  Value baseMemDesc;
  SmallVector<int32_t> offsets;
  SmallVector<int64_t> viewShape;
  SmallVector<int64_t> logicalToMemAxes;
  RankedTensorType valueType;
};

static std::optional<StaticSubviewMatch>
matchStaticSubviewMemDesc(triton::LoadOp load) {
  if (load.getMask() || load.getOther())
    return std::nullopt;
  if (load.getIsVolatile())
    return std::nullopt;
  if (load.getCache() != triton::CacheModifier::NONE ||
      load.getEvict() != triton::EvictionPolicy::NORMAL)
    return std::nullopt;

  auto loadTy = dyn_cast<RankedTensorType>(load.getType());
  if (!loadTy)
    return std::nullopt;

  Value ptr = stripConvertLayouts(load.getPtr());
  auto localPointers = ptr.getDefiningOp<tle::LocalPointersOp>();
  if (!localPointers)
    return std::nullopt;

  auto ptrTy = dyn_cast<RankedTensorType>(localPointers.getResult().getType());
  if (!ptrTy)
    return std::nullopt;

  auto memDescTy = dyn_cast<ttg::MemDescType>(localPointers.getSrc().getType());
  if (!memDescTy)
    return std::nullopt;

  auto memDescShape = memDescTy.getShape();
  if (loadTy.getShape() != ptrTy.getShape())
    return std::nullopt;
  if (loadTy.getElementType() != memDescTy.getElementType())
    return std::nullopt;

  SmallVector<int32_t> offsets(memDescTy.getRank(), 0);
  SmallVector<int64_t> viewShape(memDescShape.begin(), memDescShape.end());
  SmallVector<int64_t> logicalToMemAxes(loadTy.getRank(), -1);
  auto indices = localPointers.getIndices();
  if (indices.empty()) {
    if (loadTy.getNumElements() != memDescTy.getNumElements())
      return std::nullopt;
    int64_t logicalAxis = 0;
    for (auto [memAxis, memExtent] : llvm::enumerate(memDescShape)) {
      if (logicalAxis < loadTy.getRank() &&
          memExtent == loadTy.getShape()[logicalAxis]) {
        logicalToMemAxes[logicalAxis++] = memAxis;
      } else if (memExtent != 1) {
        return std::nullopt;
      }
    }
    if (logicalAxis != loadTy.getRank())
      return std::nullopt;
  } else {
    if (indices.size() != memDescShape.size())
      return std::nullopt;

    SmallVector<bool> mappedLogicalAxes(loadTy.getRank(), false);
    int64_t previousLogicalAxis = -1;
    for (auto [memAxis, index] : llvm::enumerate(indices)) {
      auto indexTy = dyn_cast<RankedTensorType>(index.getType());
      if (!indexTy || indexTy.getShape() != loadTy.getShape() ||
          !indexTy.getElementType().isInteger())
        return std::nullopt;

      if (std::optional<int64_t> constant = getConstantIntLike(index)) {
        if (*constant < 0 || *constant >= memDescShape[memAxis] ||
            *constant > std::numeric_limits<int32_t>::max())
          return std::nullopt;
        offsets[memAxis] = static_cast<int32_t>(*constant);
        viewShape[memAxis] = 1;
        continue;
      }

      bool matched = false;
      for (auto logicalAxis : llvm::seq<int64_t>(0, loadTy.getRank())) {
        if (mappedLogicalAxes[logicalAxis])
          continue;
        int64_t offset = 0;
        if (!matchFullIndexTensorForAxis(index, logicalAxis,
                                         loadTy.getShape(), offset))
          continue;
        if (logicalAxis <= previousLogicalAxis || offset < 0 ||
            offset > std::numeric_limits<int32_t>::max() ||
            offset + loadTy.getShape()[logicalAxis] > memDescShape[memAxis])
          return std::nullopt;
        offsets[memAxis] = static_cast<int32_t>(offset);
        viewShape[memAxis] = loadTy.getShape()[logicalAxis];
        mappedLogicalAxes[logicalAxis] = true;
        logicalToMemAxes[logicalAxis] = memAxis;
        previousLogicalAxis = logicalAxis;
        matched = true;
        break;
      }
      if (!matched)
        return std::nullopt;
    }
    if (!llvm::all_of(mappedLogicalAxes, [](bool mapped) { return mapped; }))
      return std::nullopt;
  }

  Value baseMemDesc = localPointers.getSrc();
  if (memDescTy.getRank() != loadTy.getRank()) {
    if (!isa<ttg::DotOperandEncodingAttr>(loadTy.getEncoding()))
      return std::nullopt;

    while (auto subslice =
               baseMemDesc.getDefiningOp<ttg::MemDescSubsliceOp>()) {
      for (auto [axis, subOffset] : llvm::enumerate(subslice.getOffsets())) {
        int64_t combined = static_cast<int64_t>(offsets[axis]) + subOffset;
        if (combined < 0 ||
            combined > std::numeric_limits<int32_t>::max())
          return std::nullopt;
        offsets[axis] = static_cast<int32_t>(combined);
      }
      baseMemDesc = subslice.getSrc();
    }

    auto baseTy = cast<ttg::MemDescType>(baseMemDesc.getType());
    SmallVector<bool> mappedMemAxes(baseTy.getRank(), false);
    for (int64_t memAxis : logicalToMemAxes)
      mappedMemAxes[memAxis] = true;
    for (auto [memAxis, mapped] : llvm::enumerate(mappedMemAxes)) {
      if (!mapped && (baseTy.getShape()[memAxis] != 1 ||
                      viewShape[memAxis] != 1 || offsets[memAxis] != 0))
        return std::nullopt;
      if (offsets[memAxis] < 0 ||
          offsets[memAxis] + viewShape[memAxis] >
              baseTy.getShape()[memAxis])
        return std::nullopt;
    }
  }

  return StaticSubviewMatch{baseMemDesc, std::move(offsets),
                            std::move(viewShape),
                            std::move(logicalToMemAxes), loadTy};
}

static Value createSubviewForLoad(OpBuilder &builder, Location loc,
                                  StaticSubviewMatch match) {
  auto memDescTy = cast<ttg::MemDescType>(match.baseMemDesc.getType());
  if (memDescTy.getRank() != match.valueType.getRank()) {
    SmallVector<int64_t> baseLogicalShape(match.valueType.getRank());
    SmallVector<int32_t> logicalOffsets(match.valueType.getRank());
    for (auto logicalAxis : llvm::seq<int64_t>(0,
                                               match.valueType.getRank())) {
      int64_t memAxis = match.logicalToMemAxes[logicalAxis];
      baseLogicalShape[logicalAxis] = memDescTy.getShape()[memAxis];
      logicalOffsets[logicalAxis] = match.offsets[memAxis];
    }

    Value collapsed = ttg::MemDescReshapeOp::create(
        builder, loc, match.baseMemDesc, baseLogicalShape);
    bool isFullView =
        llvm::equal(baseLogicalShape, match.valueType.getShape()) &&
        llvm::all_of(logicalOffsets,
                     [](int32_t offset) { return offset == 0; });
    if (isFullView)
      return collapsed;

    auto collapsedTy = cast<ttg::MemDescType>(collapsed.getType());
    auto subTy = ttg::MemDescType::get(
        match.valueType.getShape(), match.valueType.getElementType(),
        collapsedTy.getEncoding(), collapsedTy.getMemorySpace(),
        collapsedTy.getMutableMemory(), collapsedTy.getAllocShape());
    return ttg::MemDescSubsliceOp::create(builder, loc, subTy, collapsed,
                                          logicalOffsets);
  }

  bool isFullView =
      llvm::equal(match.viewShape, memDescTy.getShape()) &&
      llvm::all_of(match.offsets, [](int32_t offset) { return offset == 0; });
  Value view = match.baseMemDesc;
  if (!isFullView) {
    auto subTy = ttg::MemDescType::get(
        match.viewShape, match.valueType.getElementType(),
        memDescTy.getEncoding(), memDescTy.getMemorySpace(),
        memDescTy.getMutableMemory(), memDescTy.getAllocShape());
    view = ttg::MemDescSubsliceOp::create(builder, loc, subTy,
                                          match.baseMemDesc, match.offsets);
  }

  return view;
}

static RankedTensorType cloneWithElementAndEncoding(RankedTensorType type,
                                                    Type elementType,
                                                    Attribute encoding) {
  return RankedTensorType::get(type.getShape(), elementType, encoding);
}

static std::optional<RematerializedValue>
findCachedRematerialization(Value source, Type targetType,
                            ArrayRef<RematerializationCacheEntry> cache) {
  for (const RematerializationCacheEntry &entry : llvm::reverse(cache)) {
    if (entry.source == source && entry.targetType == targetType)
      return entry.rematerialized;
  }
  return std::nullopt;
}

static void cacheRematerialization(
    Value source, Type targetType, RematerializedValue rematerialized,
    llvm::SmallVectorImpl<RematerializationCacheEntry> &cache) {
  cache.push_back({source, targetType, rematerialized});
}

static std::optional<RematerializedValue> rematerializeForLayout(
    Value value, RankedTensorType targetTy, OpBuilder &builder,
    llvm::SmallVectorImpl<RematerializationCacheEntry> &cache,
    unsigned depth = 0);

static std::optional<RematerializedValue>
rematerializeConstant(arith::ConstantOp constant, RankedTensorType targetTy,
                      OpBuilder &builder) {
  auto sourceTy = dyn_cast<RankedTensorType>(constant.getType());
  if (!sourceTy || sourceTy.getShape() != targetTy.getShape() ||
      sourceTy.getElementType() != targetTy.getElementType())
    return std::nullopt;

  auto splat = dyn_cast<SplatElementsAttr>(constant.getValue());
  if (!splat)
    return std::nullopt;

  auto newAttr =
      SplatElementsAttr::get(targetTy, splat.getSplatValue<Attribute>());
  Value newConstant =
      builder.create<arith::ConstantOp>(constant.getLoc(), targetTy, newAttr);
  return RematerializedValue{newConstant, false};
}

static std::optional<RematerializedValue>
rematerializeMakeRange(triton::MakeRangeOp range, RankedTensorType targetTy,
                       OpBuilder &builder) {
  auto sourceTy = dyn_cast<RankedTensorType>(range.getType());
  if (!sourceTy || sourceTy.getShape() != targetTy.getShape() ||
      sourceTy.getElementType() != targetTy.getElementType())
    return std::nullopt;
  if (!targetTy.getElementType().isInteger(32) || targetTy.getRank() != 1)
    return std::nullopt;

  Value newRange = builder
                       .create<triton::MakeRangeOp>(
                           range.getLoc(), targetTy,
                           static_cast<uint32_t>(range.getStartAttr().getInt()),
                           static_cast<uint32_t>(range.getEndAttr().getInt()))
                       .getResult();
  return RematerializedValue{newRange, false};
}

static std::optional<RematerializedValue>
rematerializeSplat(triton::SplatOp splat, RankedTensorType targetTy,
                   OpBuilder &builder) {
  auto sourceTy = dyn_cast<RankedTensorType>(splat.getType());
  if (!sourceTy || sourceTy.getShape() != targetTy.getShape() ||
      sourceTy.getElementType() != targetTy.getElementType())
    return std::nullopt;

  Value newSplat =
      builder.create<triton::SplatOp>(splat.getLoc(), targetTy, splat.getSrc());
  return RematerializedValue{newSplat, false};
}

static std::optional<RematerializedValue> rematerializeBroadcast(
    triton::BroadcastOp broadcast, RankedTensorType targetTy,
    OpBuilder &builder,
    llvm::SmallVectorImpl<RematerializationCacheEntry> &cache, unsigned depth) {
  auto sourceResultTy = dyn_cast<RankedTensorType>(broadcast.getType());
  auto sourceInputTy = dyn_cast<RankedTensorType>(broadcast.getSrc().getType());
  if (!sourceResultTy || !sourceInputTy ||
      sourceResultTy.getShape() != targetTy.getShape() ||
      sourceResultTy.getElementType() != targetTy.getElementType())
    return std::nullopt;

  auto targetInputTy = RankedTensorType::get(sourceInputTy.getShape(),
                                             sourceInputTy.getElementType(),
                                             targetTy.getEncoding());
  auto input = rematerializeForLayout(broadcast.getSrc(), targetInputTy,
                                      builder, cache, depth + 1);
  if (!input)
    return std::nullopt;

  Value newBroadcast = builder
                           .create<triton::BroadcastOp>(broadcast.getLoc(),
                                                        targetTy, input->value)
                           .getResult();
  return RematerializedValue{newBroadcast, input->usesLocalPointerLoad};
}

static std::optional<RematerializedValue> rematerializeExpandDims(
    triton::ExpandDimsOp expand, RankedTensorType targetTy, OpBuilder &builder,
    llvm::SmallVectorImpl<RematerializationCacheEntry> &cache, unsigned depth) {
  auto sourceResultTy = dyn_cast<RankedTensorType>(expand.getType());
  auto sourceInputTy = dyn_cast<RankedTensorType>(expand.getSrc().getType());
  if (!sourceResultTy || !sourceInputTy ||
      sourceResultTy.getShape() != targetTy.getShape() ||
      sourceResultTy.getElementType() != targetTy.getElementType())
    return std::nullopt;

  unsigned axis = expand.getAxis();
  if (axis >= static_cast<unsigned>(targetTy.getRank()))
    return std::nullopt;
  auto targetEncoding =
      dyn_cast_or_null<ttg::DistributedEncodingTrait>(targetTy.getEncoding());
  if (!targetEncoding)
    return std::nullopt;
  Attribute inputEncoding =
      ttg::SliceEncodingAttr::get(builder.getContext(), axis, targetEncoding);
  auto targetInputTy = RankedTensorType::get(
      sourceInputTy.getShape(), sourceInputTy.getElementType(), inputEncoding);
  auto input = rematerializeForLayout(expand.getSrc(), targetInputTy, builder,
                                      cache, depth + 1);
  if (!input)
    return std::nullopt;

  Value newExpand =
      builder
          .create<triton::ExpandDimsOp>(expand.getLoc(), targetTy, input->value,
                                        expand.getAxisAttr())
          .getResult();
  return RematerializedValue{newExpand, input->usesLocalPointerLoad};
}

static std::optional<RematerializedValue> rematerializeLocalPointerLoad(
    triton::LoadOp load, RankedTensorType targetTy, OpBuilder &builder,
    llvm::SmallVectorImpl<RematerializationCacheEntry> &cache, unsigned depth) {
  if (load.getMask() || load.getOther())
    return std::nullopt;
  if (!load.getBoundaryCheck().empty() || load.getPadding())
    return std::nullopt;
  if (load.getIsVolatile())
    return std::nullopt;
  if (load.getCache() != triton::CacheModifier::NONE ||
      load.getEvict() != triton::EvictionPolicy::NORMAL)
    return std::nullopt;

  auto sourceTy = dyn_cast<RankedTensorType>(load.getType());
  if (!sourceTy || sourceTy.getShape() != targetTy.getShape() ||
      sourceTy.getElementType() != targetTy.getElementType())
    return std::nullopt;

  Value ptr = stripConvertLayouts(load.getPtr());
  auto localPointers = ptr.getDefiningOp<tle::LocalPointersOp>();
  if (!localPointers)
    return std::nullopt;

  auto memDescTy = dyn_cast<ttg::MemDescType>(localPointers.getSrc().getType());
  if (!memDescTy || memDescTy.getElementType() != targetTy.getElementType())
    return std::nullopt;

  SmallVector<Value> indices;
  indices.reserve(localPointers.getIndices().size());
  for (Value index : localPointers.getIndices()) {
    auto indexTy = dyn_cast<RankedTensorType>(index.getType());
    if (!indexTy || indexTy.getShape() != sourceTy.getShape() ||
        !indexTy.getElementType().isInteger())
      return std::nullopt;

    auto targetIndexTy = RankedTensorType::get(
        targetTy.getShape(), indexTy.getElementType(), targetTy.getEncoding());
    auto rematerializedIndex =
        rematerializeForLayout(index, targetIndexTy, builder, cache, depth + 1);
    if (!rematerializedIndex || rematerializedIndex->usesLocalPointerLoad)
      return std::nullopt;
    indices.push_back(rematerializedIndex->value);
  }

  Type ptrElementTy = triton::PointerType::get(targetTy.getElementType(),
                                               kSharedMemoryAddressSpace);
  auto targetPtrTy = RankedTensorType::get(targetTy.getShape(), ptrElementTy,
                                           targetTy.getEncoding());
  auto newLocalPointers = builder.create<tle::LocalPointersOp>(
      localPointers.getLoc(), targetPtrTy, localPointers.getSrc(), indices);
  for (NamedAttribute attr : localPointers->getAttrs())
    newLocalPointers->setAttr(attr.getName(), attr.getValue());

  Value newLoad = builder
                      .create<triton::LoadOp>(
                          load.getLoc(), newLocalPointers.getResult(),
                          load.getCache(), load.getEvict(),
                          load.getIsVolatile(), load.getFlagtreeHintsAttr())
                      .getResult();
  return RematerializedValue{newLoad, true};
}

template <typename OpTy>
static std::optional<RematerializedValue> rematerializeSameTypeBinary(
    OpTy op, RankedTensorType targetTy, OpBuilder &builder,
    llvm::SmallVectorImpl<RematerializationCacheEntry> &cache, unsigned depth) {
  auto sourceTy = dyn_cast<RankedTensorType>(op.getType());
  if (!sourceTy || sourceTy.getShape() != targetTy.getShape() ||
      sourceTy.getElementType() != targetTy.getElementType())
    return std::nullopt;

  auto lhs =
      rematerializeForLayout(op.getLhs(), targetTy, builder, cache, depth + 1);
  auto rhs =
      rematerializeForLayout(op.getRhs(), targetTy, builder, cache, depth + 1);
  if (!lhs || !rhs)
    return std::nullopt;

  Value result =
      builder.create<OpTy>(op.getLoc(), targetTy, lhs->value, rhs->value)
          .getResult();
  return RematerializedValue{result, lhs->usesLocalPointerLoad ||
                                         rhs->usesLocalPointerLoad};
}

static std::optional<RematerializedValue> rematerializeForLayout(
    Value value, RankedTensorType targetTy, OpBuilder &builder,
    llvm::SmallVectorImpl<RematerializationCacheEntry> &cache, unsigned depth) {
  if (depth > 32)
    return std::nullopt;

  if (value.getType() == targetTy)
    return RematerializedValue{value, false};

  if (auto cached = findCachedRematerialization(value, targetTy, cache))
    return cached;

  auto sourceTy = dyn_cast<RankedTensorType>(value.getType());
  if (!sourceTy || sourceTy.getShape() != targetTy.getShape())
    return std::nullopt;

  Operation *def = value.getDefiningOp();
  if (!def)
    return std::nullopt;

  std::optional<RematerializedValue> rematerialized;
  if (auto constant = dyn_cast<arith::ConstantOp>(def)) {
    rematerialized = rematerializeConstant(constant, targetTy, builder);
  } else if (auto convert = dyn_cast<ttg::ConvertLayoutOp>(def)) {
    rematerialized = rematerializeForLayout(convert.getSrc(), targetTy, builder,
                                            cache, depth + 1);
  } else if (auto range = dyn_cast<triton::MakeRangeOp>(def)) {
    rematerialized = rematerializeMakeRange(range, targetTy, builder);
  } else if (auto splat = dyn_cast<triton::SplatOp>(def)) {
    rematerialized = rematerializeSplat(splat, targetTy, builder);
  } else if (auto broadcast = dyn_cast<triton::BroadcastOp>(def)) {
    rematerialized =
        rematerializeBroadcast(broadcast, targetTy, builder, cache, depth);
  } else if (auto expand = dyn_cast<triton::ExpandDimsOp>(def)) {
    rematerialized =
        rematerializeExpandDims(expand, targetTy, builder, cache, depth);
  } else if (auto load = dyn_cast<triton::LoadOp>(def)) {
    rematerialized =
        rematerializeLocalPointerLoad(load, targetTy, builder, cache, depth);
  } else if (auto addPtr = dyn_cast<triton::AddPtrOp>(def)) {
    auto offsetTy = dyn_cast<RankedTensorType>(addPtr.getOffset().getType());
    auto sourceResultTy = dyn_cast<RankedTensorType>(addPtr.getType());
    if (offsetTy && sourceResultTy &&
        sourceResultTy.getShape() == targetTy.getShape() &&
        sourceResultTy.getElementType() == targetTy.getElementType()) {
      auto targetOffsetTy =
          RankedTensorType::get(targetTy.getShape(), offsetTy.getElementType(),
                                targetTy.getEncoding());
      auto ptr = rematerializeForLayout(addPtr.getPtr(), targetTy, builder,
                                        cache, depth + 1);
      auto offset = rematerializeForLayout(addPtr.getOffset(), targetOffsetTy,
                                           builder, cache, depth + 1);
      if (ptr && offset) {
        Value result = builder
                           .create<triton::AddPtrOp>(addPtr.getLoc(), targetTy,
                                                     ptr->value, offset->value)
                           .getResult();
        rematerialized = RematerializedValue{
            result, ptr->usesLocalPointerLoad || offset->usesLocalPointerLoad};
      }
    }
  } else if (auto cmp = dyn_cast<arith::CmpIOp>(def)) {
    if (targetTy.getElementType().isInteger(1)) {
      auto lhsTy = cloneWithElementAndEncoding(
          targetTy,
          cast<RankedTensorType>(cmp.getLhs().getType()).getElementType(),
          targetTy.getEncoding());
      auto rhsTy = cloneWithElementAndEncoding(
          targetTy,
          cast<RankedTensorType>(cmp.getRhs().getType()).getElementType(),
          targetTy.getEncoding());
      auto lhs = rematerializeForLayout(cmp.getLhs(), lhsTy, builder, cache,
                                        depth + 1);
      auto rhs = rematerializeForLayout(cmp.getRhs(), rhsTy, builder, cache,
                                        depth + 1);
      if (lhs && rhs) {
        Value result =
            builder
                .create<arith::CmpIOp>(cmp.getLoc(), cmp.getPredicate(),
                                       lhs->value, rhs->value)
                .getResult();
        rematerialized = RematerializedValue{
            result, lhs->usesLocalPointerLoad || rhs->usesLocalPointerLoad};
      }
    }
  } else if (auto select = dyn_cast<arith::SelectOp>(def)) {
    if (sourceTy.getElementType() == targetTy.getElementType()) {
      auto condTy = cloneWithElementAndEncoding(targetTy, builder.getI1Type(),
                                                targetTy.getEncoding());
      auto cond = rematerializeForLayout(select.getCondition(), condTy, builder,
                                         cache, depth + 1);
      auto trueValue = rematerializeForLayout(select.getTrueValue(), targetTy,
                                              builder, cache, depth + 1);
      auto falseValue = rematerializeForLayout(select.getFalseValue(), targetTy,
                                               builder, cache, depth + 1);
      if (cond && trueValue && falseValue) {
        Value result =
            builder
                .create<arith::SelectOp>(select.getLoc(), targetTy, cond->value,
                                         trueValue->value, falseValue->value)
                .getResult();
        rematerialized =
            RematerializedValue{result, cond->usesLocalPointerLoad ||
                                            trueValue->usesLocalPointerLoad ||
                                            falseValue->usesLocalPointerLoad};
      }
    }
  } else if (auto ext = dyn_cast<arith::ExtSIOp>(def)) {
    auto inTy = cloneWithElementAndEncoding(
        targetTy,
        cast<RankedTensorType>(ext.getIn().getType()).getElementType(),
        targetTy.getEncoding());
    auto in =
        rematerializeForLayout(ext.getIn(), inTy, builder, cache, depth + 1);
    if (in) {
      Value result =
          builder.create<arith::ExtSIOp>(ext.getLoc(), targetTy, in->value)
              .getResult();
      rematerialized = RematerializedValue{result, in->usesLocalPointerLoad};
    }
  } else if (auto ext = dyn_cast<arith::ExtUIOp>(def)) {
    auto inTy = cloneWithElementAndEncoding(
        targetTy,
        cast<RankedTensorType>(ext.getIn().getType()).getElementType(),
        targetTy.getEncoding());
    auto in =
        rematerializeForLayout(ext.getIn(), inTy, builder, cache, depth + 1);
    if (in) {
      Value result =
          builder.create<arith::ExtUIOp>(ext.getLoc(), targetTy, in->value)
              .getResult();
      rematerialized = RematerializedValue{result, in->usesLocalPointerLoad};
    }
  } else if (auto trunc = dyn_cast<arith::TruncIOp>(def)) {
    auto inTy = cloneWithElementAndEncoding(
        targetTy,
        cast<RankedTensorType>(trunc.getIn().getType()).getElementType(),
        targetTy.getEncoding());
    auto in =
        rematerializeForLayout(trunc.getIn(), inTy, builder, cache, depth + 1);
    if (in) {
      Value result =
          builder.create<arith::TruncIOp>(trunc.getLoc(), targetTy, in->value)
              .getResult();
      rematerialized = RematerializedValue{result, in->usesLocalPointerLoad};
    }
  } else if (auto add = dyn_cast<arith::AddIOp>(def)) {
    rematerialized =
        rematerializeSameTypeBinary(add, targetTy, builder, cache, depth);
  } else if (auto sub = dyn_cast<arith::SubIOp>(def)) {
    rematerialized =
        rematerializeSameTypeBinary(sub, targetTy, builder, cache, depth);
  } else if (auto mul = dyn_cast<arith::MulIOp>(def)) {
    rematerialized =
        rematerializeSameTypeBinary(mul, targetTy, builder, cache, depth);
  } else if (auto andOp = dyn_cast<arith::AndIOp>(def)) {
    rematerialized =
        rematerializeSameTypeBinary(andOp, targetTy, builder, cache, depth);
  } else if (auto orOp = dyn_cast<arith::OrIOp>(def)) {
    rematerialized =
        rematerializeSameTypeBinary(orOp, targetTy, builder, cache, depth);
  } else if (auto xorOp = dyn_cast<arith::XOrIOp>(def)) {
    rematerialized =
        rematerializeSameTypeBinary(xorOp, targetTy, builder, cache, depth);
  }

  if (!rematerialized)
    return std::nullopt;
  cacheRematerialization(value, targetTy, *rematerialized, cache);
  return rematerialized;
}

static bool isLocalPointerLoad(triton::LoadOp load) {
  if (!load || load.getIsVolatile() || load.getMask() || load.getOther())
    return false;
  return stripConvertLayouts(load.getPtr())
             .getDefiningOp<tle::LocalPointersOp>() != nullptr;
}

static bool isDeadRematerializableOp(Operation *op) {
  if (!op || !op->use_empty())
    return false;
  if (auto load = dyn_cast<triton::LoadOp>(op))
    return isLocalPointerLoad(load);
  return isa<tle::LocalPointersOp, ttg::ConvertLayoutOp, triton::MakeRangeOp,
             triton::SplatOp, triton::BroadcastOp, triton::ExpandDimsOp,
             triton::AddPtrOp, arith::ConstantOp, arith::CmpIOp,
             arith::SelectOp, arith::ExtSIOp, arith::ExtUIOp, arith::TruncIOp,
             arith::AddIOp, arith::SubIOp, arith::MulIOp, arith::AndIOp,
             arith::OrIOp, arith::XOrIOp>(op);
}

static void eraseDeadRematerializableOps(ModuleOp module) {
  while (true) {
    SmallVector<Operation *> deadOps;
    module.walk([&](Operation *op) {
      if (isDeadRematerializableOp(op))
        deadOps.push_back(op);
    });
    if (deadOps.empty())
      return;
    for (Operation *op : deadOps)
      op->erase();
  }
}

class OptimizeLocalPointerLoadsPass
    : public impl::TritonTleOptimizeLocalPointerLoadsBase<
          OptimizeLocalPointerLoadsPass> {
  void runOnOperation() override {
    ModuleOp module = getOperation();

    struct ConvertRewriteItem {
      ttg::ConvertLayoutOp convert;
      Value source;
      RematerializedValue replacement;
    };
    struct RewriteItem {
      triton::LoadOp load;
      StaticSubviewMatch match;
    };
    SmallVector<ConvertRewriteItem> convertRewrites;
    SmallVector<RewriteItem> rewrites;

    module.walk([&](ttg::ConvertLayoutOp convert) {
      auto targetTy = dyn_cast<RankedTensorType>(convert.getType());
      auto sourceTy = dyn_cast<RankedTensorType>(convert.getSrc().getType());
      if (!targetTy || !sourceTy || targetTy.getShape() != sourceTy.getShape())
        return;

      OpBuilder builder(convert);
      SmallVector<RematerializationCacheEntry> rematerializationCache;
      auto rematerialized = rematerializeForLayout(
          convert.getSrc(), targetTy, builder, rematerializationCache);
      if (!rematerialized || !rematerialized->usesLocalPointerLoad)
        return;
      convertRewrites.push_back(
          {convert, convert.getSrc(), std::move(*rematerialized)});
    });

    for (ConvertRewriteItem &item : convertRewrites) {
      if (!item.convert || !item.replacement.value)
        continue;
      item.convert.replaceAllUsesWith(item.replacement.value);
      item.convert.erase();
    }
    eraseDeadRematerializableOps(module);

    module.walk([&](triton::LoadOp load) {
      if (auto match = matchStaticSubviewMemDesc(load))
        rewrites.push_back({load, std::move(*match)});
    });

    for (RewriteItem &item : rewrites) {
      if (!item.load || !item.match.baseMemDesc)
        continue;
      OpBuilder builder(item.load);
      Value memDesc = createSubviewForLoad(builder, item.load.getLoc(),
                                           std::move(item.match));
      auto localLoad = builder.create<ttg::LocalLoadOp>(
          item.load.getLoc(), item.load.getType(), memDesc);
      item.load.replaceAllUsesWith(localLoad.getResult());
      item.load.erase();
    }
  }
};

} // namespace
} // namespace mlir::triton::tle
