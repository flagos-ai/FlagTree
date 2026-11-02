#ifndef TRITON_TLE_LLVM_COPYFILL_H
#define TRITON_TLE_LLVM_COPYFILL_H

#include "mlir/Dialect/LLVMIR/LLVMDialect.h"

namespace mlir::triton::tle {

inline Attribute getLLVMCopyFillConstant(Value value) {
  if (auto constant = value.getDefiningOp<LLVM::ConstantOp>())
    return constant.getValue();
  if (auto cast = value.getDefiningOp<LLVM::BitcastOp>())
    return getLLVMCopyFillConstant(cast.getArg());
  // These casts preserve positive zero. Other values are left conservative
  // by the zero-bit check; mask checks only accept actual i1 scalar values.
  if (auto cast = value.getDefiningOp<LLVM::FPTruncOp>())
    return getLLVMCopyFillConstant(cast.getArg());
  if (auto cast = value.getDefiningOp<LLVM::FPExtOp>())
    return getLLVMCopyFillConstant(cast.getArg());
  if (auto extract = value.getDefiningOp<LLVM::ExtractValueOp>()) {
    if (extract.getPosition().size() != 1)
      return {};
    auto container = extract.getContainer();
    while (auto insert = container.getDefiningOp<LLVM::InsertValueOp>()) {
      if (insert.getPosition() == extract.getPosition())
        return getLLVMCopyFillConstant(insert.getValue());
      container = insert.getContainer();
    }
  }
  return {};
}

inline bool hasLLVMZeroFillBits(Value value) {
  auto attr = getLLVMCopyFillConstant(value);
  if (auto integer = dyn_cast_or_null<IntegerAttr>(attr))
    return integer.getValue().isZero();
  if (auto floating = dyn_cast_or_null<FloatAttr>(attr))
    return floating.getValue().bitcastToAPInt().isZero();
  return false;
}

inline bool hasLLVMTrueMask(Value value) {
  if (!value.getType().isInteger(1))
    return false;
  auto attr = dyn_cast_or_null<IntegerAttr>(getLLVMCopyFillConstant(value));
  return attr && attr.getValue().isOne();
}

} // namespace mlir::triton::tle

#endif
