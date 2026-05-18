#ifndef TRITONMUSA_COMMON_SQMMA_ATTR_UTILS_H
#define TRITONMUSA_COMMON_SQMMA_ATTR_UTILS_H

#include "mlir/IR/Attributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Operation.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/StringRef.h"
#include <array>
#include <optional>

namespace mlir::triton::musa {

inline constexpr llvm::StringLiteral kSqmmaOpIdxAttr = "sqmma.op_idx";
inline constexpr llvm::StringLiteral kSqmmaElemBytesAttr = "sqmma.elem_bytes";
inline constexpr llvm::StringLiteral kSqmmaRowMajorAttr = "sqmma.row_major";

inline constexpr std::array<llvm::StringLiteral, 3> kSqmmaAttrNames = {
    kSqmmaOpIdxAttr, kSqmmaElemBytesAttr, kSqmmaRowMajorAttr};

inline std::optional<int64_t> getIntAttr(Operation *op, StringRef name) {
  if (auto intAttr = op->getAttrOfType<IntegerAttr>(name))
    return intAttr.getInt();
  return std::nullopt;
}

inline std::optional<int64_t> getSqmmaOpIdx(Operation *op) {
  return getIntAttr(op, kSqmmaOpIdxAttr);
}

inline std::optional<int64_t> getSqmmaElemBytes(Operation *op) {
  return getIntAttr(op, kSqmmaElemBytesAttr);
}

inline bool getSqmmaRowMajor(Operation *op, bool defaultValue) {
  if (auto boolAttr = op->getAttrOfType<BoolAttr>(kSqmmaRowMajorAttr))
    return boolAttr.getValue();
  if (auto intAttr = op->getAttrOfType<IntegerAttr>(kSqmmaRowMajorAttr))
    return intAttr.getInt() != 0;
  return defaultValue;
}

inline bool hasSqmmaOpIdxAttr(Operation *op) {
  return getSqmmaOpIdx(op).has_value();
}

inline void copySqmmaAttrs(Operation *src, Operation *dst) {
  for (auto name : kSqmmaAttrNames) {
    if (Attribute attr = src->getAttr(name))
      dst->setAttr(name, attr);
  }
}

inline void setSqmmaAttrs(Operation *op, int64_t opIdx, int64_t elemBytes,
                          bool rowMajor) {
  auto *ctx = op->getContext();
  auto i32Ty = IntegerType::get(ctx, 32);
  auto opIdxAttr = IntegerAttr::get(i32Ty, opIdx);
  auto elemBytesAttr = IntegerAttr::get(i32Ty, elemBytes);
  op->setAttr(kSqmmaOpIdxAttr, opIdxAttr);
  op->setAttr(kSqmmaElemBytesAttr, elemBytesAttr);
  op->setAttr(kSqmmaRowMajorAttr, BoolAttr::get(ctx, rowMajor));
}

} // namespace mlir::triton::musa

#endif // TRITONMUSA_COMMON_SQMMA_ATTR_UTILS_H
