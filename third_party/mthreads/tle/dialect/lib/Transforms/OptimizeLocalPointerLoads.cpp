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

#define GEN_PASS_DEF_TRITONMUSAGPUTLEOPTIMIZELOCALPOINTERLOADS
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
  if (!range)
    return false;
  return range.getStart() == 0 && range.getEnd() == extent;
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

static std::optional<Value> matchFullViewMemDesc(triton::LoadOp load) {
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
  auto localPointers = ptr.getDefiningOp<triton::musa_tle::LocalPointersOp>();
  if (!localPointers)
    return std::nullopt;

  auto ptrTy = dyn_cast<RankedTensorType>(localPointers.getResult().getType());
  if (!ptrTy)
    return std::nullopt;

  auto memDescTy = dyn_cast<ttg::MemDescType>(localPointers.getSrc().getType());
  if (!memDescTy)
    return std::nullopt;

  auto memDescShape = memDescTy.getShape();
  if (loadTy.getShape() != memDescShape || ptrTy.getShape() != memDescShape)
    return std::nullopt;
  if (loadTy.getElementType() != memDescTy.getElementType())
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

class OptimizeLocalPointerLoadsPass
    : public impl::TritonMUSAGPUTLEOptimizeLocalPointerLoadsBase<
          OptimizeLocalPointerLoadsPass> {
  void runOnOperation() override {
    ModuleOp module = getOperation();

    struct RewriteItem {
      triton::LoadOp load;
      Value memDesc;
    };
    SmallVector<RewriteItem> rewrites;

    module.walk([&](triton::LoadOp load) {
      if (auto memDesc = matchFullViewMemDesc(load))
        rewrites.push_back({load, *memDesc});
    });

    for (RewriteItem &item : rewrites) {
      if (!item.load || !item.memDesc)
        continue;
      OpBuilder builder(item.load);
      auto localLoad = ttg::LocalLoadOp::create(
          builder, item.load.getLoc(), item.load.getType(), item.memDesc);
      item.load.replaceAllUsesWith(localLoad.getResult());
      item.load.erase();
    }
  }
};

} // namespace
} // namespace mlir

#endif // __TLE__
