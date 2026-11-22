// Copyright 2026 FlagOS Contributors
// SPDX-License-Identifier: MIT
#ifndef TLE_IR_PHYSICAL_SHARED_H
#define TLE_IR_PHYSICAL_SHARED_H

#include "triton/Dialect/TritonGPU/IR/Dialect.h"

namespace mlir::triton::tle {
// A one-dimensional, unswizzled, single-CTA arena has an affine element
// address, independent of its extent. It is not a distributed bit layout.
inline bool isAffineFlatShared(ArrayRef<int64_t> shape, Attribute encoding) {
  auto enc = dyn_cast<gpu::SwizzledSharedEncodingAttr>(encoding);
  return shape.size() == 1 && enc && enc.getVec() == 1 &&
         enc.getPerPhase() == 1 && enc.getMaxPhase() == 1 &&
         llvm::all_of(gpu::getCTAsPerCGA(enc), [](unsigned n) { return n == 1; }) &&
         llvm::all_of(gpu::getCTASplitNum(enc), [](unsigned n) { return n == 1; });
}
inline bool isAffineFlatShared(gpu::MemDescType type) {
  return isAffineFlatShared(type.getShape(), type.getEncoding());
}
} // namespace mlir::triton::tle
#endif
