#ifndef TRITON_TLE_IR_COPYFILL_H
#define TRITON_TLE_IR_COPYFILL_H

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"

namespace mlir::triton::tle {

inline bool hasZeroFillBits(Value value) {
  // Layout conversion preserves bits; floating negative zero is not zero fill.
  while (auto convert = value.getDefiningOp<gpu::ConvertLayoutOp>())
    value = convert.getSrc();
  auto constant = value.getDefiningOp<arith::ConstantOp>();
  if (!constant)
    return false;
  if (auto dense = dyn_cast<DenseFPElementsAttr>(constant.getValue()))
    return dense.isSplat() &&
           dense.getSplatValue<APFloat>().bitcastToAPInt().isZero();
  if (auto dense = dyn_cast<DenseIntElementsAttr>(constant.getValue()))
    return dense.isSplat() && dense.getSplatValue<APInt>().isZero();
  return false;
}

} // namespace mlir::triton::tle

#endif
