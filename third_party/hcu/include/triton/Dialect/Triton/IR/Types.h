#ifndef TRITON_IR_TYPES_H_
#define TRITON_IR_TYPES_H_

#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/TypeSupport.h"
#include "mlir/IR/Types.h"
#include <optional>
#include <utility>

#define GET_TYPEDEF_CLASSES
#include "triton/Dialect/Triton/IR/Types.h.inc"

namespace mlir {

namespace triton {

bool isTensorPointerType(Type type);

bool isTensorOrTensorPointerType(Type type);

unsigned getPointeeBitWidth(Type type);

Type getPointeeType(Type type);

Type getPointerType(Type type, int addressSpace = 1);

int getAddressSpace(Type type);

Type getElementTypeOfTensorPointerType(Type type);

Type getI1SameShape(Type type);

Type getI32SameShape(Type type);

Type getPointerTypeSameShape(Type type);

Type getPointerTypeToElement(Type type);

// keep-in-sync: include/triton/Dialect/Triton/IR/Types.h
// Returns the (size, align) in bytes of a custom-type field, or std::nullopt
// if the type is not a supported struct/array field type.
std::optional<std::pair<int64_t, int64_t>> getTypeSizeAlign(Type type);

} // namespace triton

} // namespace mlir

#endif // TRITON_IR_TYPES_H_
