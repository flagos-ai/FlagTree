// MIT License
//
// Copyright (c) 2025 The FlagOS Contributors

#include "tle/dialect/include/IR/Dialect.h"
#include "tle/dialect/include/Transforms/Passes.h"
#include "tle/dialect/include/Transforms/TransformAttrs.h"

#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/Triton/IR/Types.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"

#include <optional>

namespace mlir::triton::tle {

#define GEN_PASS_DEF_TRITONTLEOPTIMIZELOCALPOINTERASYNCSTORES
#include "tle/dialect/include/Transforms/Passes.h.inc"

namespace {

namespace tt = mlir::triton;
namespace ttg = mlir::triton::gpu;

static Value stripConvertLayouts(Value value) {
  Value current = value;
  while (auto cvt = current.getDefiningOp<ttg::ConvertLayoutOp>())
    current = cvt.getSrc();
  return current;
}

static Value stripStoreValueWrappers(Value value) {
  Value current = value;
  while (true) {
    if (auto cvt = current.getDefiningOp<ttg::ConvertLayoutOp>()) {
      current = cvt.getSrc();
      continue;
    }
    break;
  }
  return current;
}

static bool isGlobalPointerTensor(Value value) {
  auto tensorTy = dyn_cast<RankedTensorType>(value.getType());
  if (!tensorTy)
    return false;
  auto ptrTy = dyn_cast<tt::PointerType>(tensorTy.getElementType());
  if (!ptrTy)
    return false;
  return ptrTy.getAddressSpace() == 1;
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
  value = stripIndexValueWrappers(value);
  if (auto splat = value.getDefiningOp<tt::SplatOp>())
    return getConstantIntLike(splat.getSrc());
  if (auto cst = value.getDefiningOp<arith::ConstantOp>()) {
    if (auto dense = dyn_cast<DenseIntElementsAttr>(cst.getValue())) {
      if (dense.isSplat())
        return dense.getSplatValue<APInt>().getSExtValue();
    }
  }
  if (auto cst = value.getDefiningOp<arith::ConstantIntOp>())
    return cst.value();
  if (auto cst = value.getDefiningOp<arith::ConstantIndexOp>())
    return cst.value();
  return std::nullopt;
}

static bool matchRangeWithStaticOffset(Value value, int64_t extent,
                                       int64_t &offset) {
  Value current = stripIndexValueWrappers(value);
  if (auto range = current.getDefiningOp<tt::MakeRangeOp>()) {
    offset = range.getStart();
    return range.getEnd() - range.getStart() == extent;
  }

  auto add = current.getDefiningOp<arith::AddIOp>();
  if (!add)
    return false;

  auto tryMatch = [&](Value lhs, Value rhs) -> bool {
    Value lhsStripped = stripIndexValueWrappers(lhs);
    auto range = lhsStripped.getDefiningOp<tt::MakeRangeOp>();
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

  auto bcast = current.getDefiningOp<tt::BroadcastOp>();
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
  while (auto expand = current.getDefiningOp<tt::ExpandDimsOp>())
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
  RankedTensorType valueType;
};

static std::optional<StaticSubviewMatch>
matchStaticSubviewMemDesc(triton::StoreOp store) {
  Value ptr = stripConvertLayouts(store.getPtr());
  auto localPointers = ptr.getDefiningOp<tle::LocalPointersOp>();
  if (!localPointers)
    return std::nullopt;

  auto valueTy = dyn_cast<RankedTensorType>(store.getValue().getType());
  auto ptrTy = dyn_cast<RankedTensorType>(localPointers.getResult().getType());
  auto memDescTy = dyn_cast<ttg::MemDescType>(localPointers.getSrc().getType());
  if (!valueTy || !ptrTy || !memDescTy)
    return std::nullopt;
  if (valueTy.getShape() != ptrTy.getShape())
    return std::nullopt;
  if (valueTy.getElementType() != memDescTy.getElementType())
    return std::nullopt;

  auto indices = localPointers.getIndices();
  SmallVector<int32_t> offsets(memDescTy.getRank(), 0);
  if (indices.empty()) {
    if (llvm::equal(valueTy.getShape(), memDescTy.getShape()))
      return StaticSubviewMatch{localPointers.getSrc(), std::move(offsets),
                                valueTy};
    return std::nullopt;
  }
  if (indices.size() != static_cast<size_t>(memDescTy.getRank()))
    return std::nullopt;

  for (auto [axis, index] : llvm::enumerate(indices)) {
    int64_t offset = 0;
    if (!matchFullIndexTensorForAxis(index, axis, valueTy.getShape(), offset))
      return std::nullopt;
    offsets[axis] = static_cast<int32_t>(offset);
  }

  return StaticSubviewMatch{localPointers.getSrc(), std::move(offsets),
                            valueTy};
}

static Value createSubviewForStore(OpBuilder &builder, Location loc,
                                   StaticSubviewMatch match) {
  auto memDescTy = cast<ttg::MemDescType>(match.baseMemDesc.getType());
  bool isFullView =
      llvm::equal(match.valueType.getShape(), memDescTy.getShape()) &&
      llvm::all_of(match.offsets, [](int32_t offset) { return offset == 0; });
  if (isFullView)
    return match.baseMemDesc;

  auto subTy = ttg::MemDescType::get(
      match.valueType.getShape(), match.valueType.getElementType(),
      memDescTy.getEncoding(), memDescTy.getMemorySpace(),
      memDescTy.getMutableMemory(), memDescTy.getAllocShape());
  return ttg::MemDescSubsliceOp::create(builder, loc, subTy, match.baseMemDesc,
                                        match.offsets);
}

class OptimizeLocalPointerAsyncStoresPass
    : public impl::TritonTleOptimizeLocalPointerAsyncStoresBase<
          OptimizeLocalPointerAsyncStoresPass> {
  void runOnOperation() override {
    ModuleOp module = getOperation();

    SmallVector<triton::StoreOp> stores;
    module.walk([&](triton::StoreOp store) { stores.push_back(store); });

    for (triton::StoreOp store : stores) {
      if (!store)
        continue;
      Value strippedStoreValue = stripStoreValueWrappers(store.getValue());
      auto load = strippedStoreValue.getDefiningOp<tt::LoadOp>();
      if (!load || !load->hasOneUse())
        continue;
      if (load.getIsVolatile())
        continue;
      if (!isa<RankedTensorType>(load.getType()))
        continue;
      if (!isGlobalPointerTensor(load.getPtr()))
        continue;
      auto match = matchStaticSubviewMemDesc(store);
      if (!match)
        continue;
      if (cast<RankedTensorType>(load.getType()).getShape() !=
          match->valueType.getShape())
        continue;
      if (cast<RankedTensorType>(load.getType()).getElementType() !=
          match->valueType.getElementType())
        continue;

      OpBuilder builder(store);
      Value dst = createSubviewForStore(builder, store.getLoc(), *match);
      auto asyncCopy = builder.create<ttg::AsyncCopyGlobalToLocalOp>(
          store.getLoc(), load.getPtr(), dst, load.getMask(), load.getOther(),
          load.getCache(), load.getEvict(), load.getIsVolatile());
      asyncCopy->setAttr(kTleLocalPointerAsyncStoreAttr, builder.getUnitAttr());
      auto commit = builder.create<ttg::AsyncCommitGroupOp>(
          store.getLoc(), asyncCopy.getToken());
      builder.create<ttg::AsyncWaitOp>(store.getLoc(), commit.getResult(), 0);
      Value originalStoreValue = store.getValue();
      store.erase();
      for (Value current = originalStoreValue; current != load.getResult();) {
        Operation *def = current.getDefiningOp();
        auto cvt = dyn_cast_or_null<ttg::ConvertLayoutOp>(def);
        if (!cvt || !cvt->use_empty())
          break;
        current = cvt.getSrc();
        cvt.erase();
      }
      load.erase();
    }
  }
};

} // namespace
} // namespace mlir::triton::tle
