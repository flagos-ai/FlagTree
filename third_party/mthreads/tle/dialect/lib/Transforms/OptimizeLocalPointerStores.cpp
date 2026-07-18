#ifdef __TLE__

#include "Dialect/MUSATLE/IR/Dialect.h"
#include "TritonMUSAGPUTransforms/Passes.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"

#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include <optional>

namespace mlir {

#define GEN_PASS_DEF_TRITONMUSAGPUTLEOPTIMIZELOCALPOINTERSTORES
#include "TritonMUSAGPUTransforms/Passes.h.inc"

namespace {

namespace ttg = mlir::triton::gpu;

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

static bool matchZeroStartMakeRange(Value value, int64_t extent) {
  Value current = stripIndexValueWrappers(value);
  auto range = current.getDefiningOp<triton::MakeRangeOp>();
  return range && range.getStart() == 0 && range.getEnd() == extent;
}

static bool matchFullIndexTensorForAxis(Value index, size_t axis,
                                        ArrayRef<int64_t> shape) {
  auto indexTy = dyn_cast<RankedTensorType>(index.getType());
  if (!indexTy || !indexTy.getElementType().isInteger())
    return false;
  if (indexTy.getShape() != shape)
    return false;

  Value current = stripIndexValueWrappers(index);
  if (shape.size() == 1)
    return matchZeroStartMakeRange(current, shape.front());

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

  return matchZeroStartMakeRange(current, shape[axis]);
}

static std::optional<Value> matchFullViewMemDesc(triton::StoreOp store) {
  if (!store.getBoundaryCheck().empty())
    return std::nullopt;

  auto valueTy = dyn_cast<RankedTensorType>(store.getValue().getType());
  if (!valueTy)
    return std::nullopt;

  Value ptr = stripConvertLayouts(store.getPtr());
  auto localPointers = ptr.getDefiningOp<triton::musa_tle::LocalPointersOp>();
  if (!localPointers)
    return std::nullopt;

  auto ptrTy = dyn_cast<RankedTensorType>(localPointers.getResult().getType());
  auto memDescTy = dyn_cast<ttg::MemDescType>(localPointers.getSrc().getType());
  if (!ptrTy || !memDescTy)
    return std::nullopt;

  auto memDescShape = memDescTy.getShape();
  if (valueTy.getShape() != memDescShape || ptrTy.getShape() != memDescShape)
    return std::nullopt;
  if (valueTy.getElementType() != memDescTy.getElementType())
    return std::nullopt;

  auto indices = localPointers.getIndices();
  if (indices.empty())
    return localPointers.getSrc();
  if (indices.size() != memDescShape.size())
    return std::nullopt;

  for (auto [axis, index] : llvm::enumerate(indices))
    if (!matchFullIndexTensorForAxis(index, axis, memDescShape))
      return std::nullopt;

  return localPointers.getSrc();
}

class OptimizeLocalPointerStoresPass
    : public impl::TritonMUSAGPUTLEOptimizeLocalPointerStoresBase<
          OptimizeLocalPointerStoresPass> {
  void runOnOperation() override {
    ModuleOp module = getOperation();

    struct RewriteItem {
      triton::StoreOp store;
      Value memDesc;
    };
    SmallVector<RewriteItem> rewrites;

    module.walk([&](triton::StoreOp store) {
      if (auto memDesc = matchFullViewMemDesc(store))
        rewrites.push_back({store, *memDesc});
    });

    for (RewriteItem &item : rewrites) {
      if (!item.store || !item.memDesc)
        continue;

      OpBuilder builder(item.store);
      Value valueToStore = item.store.getValue();
      auto valueTy = cast<RankedTensorType>(valueToStore.getType());

      if (Value mask = item.store.getMask()) {
        auto maskTy = dyn_cast<RankedTensorType>(mask.getType());
        if (!maskTy || maskTy.getShape() != valueTy.getShape())
          continue;
        if (maskTy.getEncoding() != valueTy.getEncoding()) {
          auto targetMaskTy =
              RankedTensorType::get(maskTy.getShape(), maskTy.getElementType(),
                                    valueTy.getEncoding());
          mask = ttg::ConvertLayoutOp::create(builder, item.store.getLoc(),
                                              targetMaskTy, mask)
                     .getResult();
        }
        Value oldValue = ttg::LocalLoadOp::create(builder, item.store.getLoc(),
                                                  valueTy, item.memDesc)
                             .getResult();
        valueToStore = arith::SelectOp::create(builder, item.store.getLoc(),
                                               mask, valueToStore, oldValue)
                           .getResult();
      }

      ttg::LocalStoreOp::create(builder, item.store.getLoc(), valueToStore,
                                item.memDesc);
      item.store.erase();
    }
  }
};

} // namespace
} // namespace mlir

#endif // __TLE__
