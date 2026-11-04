#pragma once

#include "mlir/IR/Operation.h"
#include "llvm/Support/MathExtras.h"
#include <limits>

namespace mlir::triton::tle {

// Grid barriers use a frame-relative counter for each disjoint CTA group.
// Selected group extents divide their domain, as checked by the op verifier.
inline uint32_t getGridBarrierScratchBytes(Operation *op) {
  if (op->getName().getStringRef() != "tle.distributed_barrier")
    return 0;
  auto kind = op->getAttrOfType<StringAttr>("group_kind");
  if (!kind)
    return 0;
  if (kind.getValue() == "grid")
    return 4;
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
  assert(groups <= std::numeric_limits<int32_t>::max() / 4 &&
         "grid scratch must fit the allocator's signed byte offsets");
  return static_cast<uint32_t>(groups * 4);
}

} // namespace mlir::triton::tle
