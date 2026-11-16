#pragma once

#include "mlir/IR/Operation.h"
#include "llvm/Support/MathExtras.h"
#include "tle/dialect/include/Transforms/FlagsGridBarrier.h"
#include <limits>

namespace mlir::triton::tle {

constexpr uint32_t kHierarchicalGridBuckets = 8;
constexpr uint32_t kHierarchicalGridSlotBytes = 128;
constexpr uint32_t kHierarchicalGridRootOffset =
    kHierarchicalGridBuckets * kHierarchicalGridSlotBytes;
constexpr uint32_t kHierarchicalGridGroupBytes =
    (kHierarchicalGridBuckets + 1) * kHierarchicalGridSlotBytes;

inline bool isHierarchicalGridBarrier(Operation *op) {
  auto protocol = op->getAttrOfType<StringAttr>("tle.grid_barrier_protocol");
  return protocol && protocol.getValue() == "hierarchical8";
}

inline uint32_t getGridBarrierGroupBytes(Operation *op) {
  if (isFlagsGridBarrier(op)) {
    auto layout = getFlagsGridLayout(op);
    assert(succeeded(layout) && "flags32 layout must be verified before allocation");
    return layout->groupBytes;
  }
  return isHierarchicalGridBarrier(op) ? kHierarchicalGridGroupBytes : 4;
}

inline uint32_t getGridBarrierScratchAlignment(Operation *op) {
  if (isFlagsGridBarrier(op))
    return kFlagsGridAlignment;
  return isHierarchicalGridBarrier(op) ? kHierarchicalGridSlotBytes : 4;
}

// Grid barriers use a frame-relative counter for each disjoint CTA group.
// Selected group extents divide their domain, as checked by the op verifier.
inline uint32_t getGridBarrierScratchBytes(Operation *op) {
  if (op->getName().getStringRef() != "tle.distributed_barrier")
    return 0;
  auto kind = op->getAttrOfType<StringAttr>("group_kind");
  if (!kind)
    return 0;
  uint32_t groupBytes = getGridBarrierGroupBytes(op);
  if (kind.getValue() == "grid")
    return groupBytes;
  if (kind.getValue() != "grid_axis")
    return 0;
  auto domain = op->getAttrOfType<DenseI32ArrayAttr>("group_domain_shape");
  auto axes = op->getAttrOfType<DenseI32ArrayAttr>("group_axes");
  auto shape = op->getAttrOfType<DenseI32ArrayAttr>("group_shape");
  assert(domain && axes && shape && axes.size() == shape.size() &&
         "grid scratch requires verified group metadata");
  SmallVector<int32_t> extents(domain.size(), 1);
  for (auto [axis, extent] : llvm::zip(axes.asArrayRef(), shape.asArrayRef()))
    extents[axis] = extent;
  uint64_t groups = 1;
  for (auto [dim, extent] : llvm::zip(domain.asArrayRef(), extents))
    groups *= dim / extent;
  assert(groups <= std::numeric_limits<int32_t>::max() / groupBytes &&
         "grid scratch must fit the allocator's signed byte offsets");
  return static_cast<uint32_t>(groups * groupBytes);
}

} // namespace mlir::triton::tle
