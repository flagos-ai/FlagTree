/*
 * Copyright 2018-2020 Philippe Tillet
 * Copyright 2020-2022 OpenAI
 * Copyright 2025-     FlagOS Contributors
 *
 * Permission is hereby granted, free of charge, to any person obtaining
 * a copy of this software and associated documentation files
 * (the "Software"), to deal in the Software without restriction,
 * including without limitation the rights to use, copy, modify, merge,
 * publish, distribute, sublicense, and/or sell copies of the Software,
 * and to permit persons to whom the Software is furnished to do so,
 * subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be
 * included in all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
 * EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
 * MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
 * IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY
 * CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,
 * TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE
 * SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
 */

#include "triton/Dialect/Triton/IR/Types.h"

#include "mlir/IR/DialectImplementation.h" // required by `Types.cpp.inc`
#include "mlir/IR/TypeUtilities.h"
#include "mlir/Support/LLVM.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "llvm/ADT/TypeSwitch.h" // required by `Types.cpp.inc`

using namespace mlir;
using namespace mlir::triton;

#define GET_TYPEDEF_CLASSES
#include "triton/Dialect/Triton/IR/Types.cpp.inc"

//===----------------------------------------------------------------------===//
// Triton Dialect
//===----------------------------------------------------------------------===//
void TritonDialect::registerTypes() {
  addTypes<
#define GET_TYPEDEF_LIST
#include "triton/Dialect/Triton/IR/Types.cpp.inc"
      >();
}

Type PointerType::parse(AsmParser &parser) {
  if (parser.parseLess())
    return Type();

  Type pointeeType;
  if (parser.parseType(pointeeType))
    return Type();

  int addressSpace = 1;
  if (succeeded(parser.parseOptionalComma())) {
    if (parser.parseInteger(addressSpace))
      return Type();
  }

  if (parser.parseGreater())
    return Type();

  return PointerType::get(pointeeType, addressSpace);
}

void PointerType::print(AsmPrinter &printer) const {
  if (getAddressSpace() == 1) {
    printer << "<" << getPointeeType() << ">";
  } else {
    printer << "<" << getPointeeType() << ", " << getAddressSpace() << ">";
  }
}

namespace mlir {

namespace triton {

unsigned getPointeeBitWidth(Type type) {
  auto pointeeType = getPointeeType(type);
  if (auto tensorTy = dyn_cast<RankedTensorType>(pointeeType))
    return tensorTy.getElementType().getIntOrFloatBitWidth();
  return pointeeType.getIntOrFloatBitWidth();
}

Type getI1SameShape(Type type) {
  auto i1Type = IntegerType::get(type.getContext(), 1);
  if (auto tensorTy = dyn_cast<RankedTensorType>(type))
    return tensorTy.clone(i1Type);
  return i1Type;
}

Type getPointeeType(Type type) {
  if (auto tensorTy = dyn_cast<RankedTensorType>(type)) {
    // Tensor of pointers
    auto ptrType = dyn_cast<PointerType>(tensorTy.getElementType());
    Type pointeeType = ptrType.getPointeeType();
    return tensorTy.clone(pointeeType);
  } else if (auto ptrType = dyn_cast<PointerType>(type)) {
    // scalar pointer
    Type pointeeType = ptrType.getPointeeType();
    return pointeeType;
  }
  return type;
}

Type getI32SameShape(Type type) {
  auto i32Type = IntegerType::get(type.getContext(), 32);
  if (auto tensorTy = dyn_cast<RankedTensorType>(type))
    return tensorTy.clone(i32Type);
  return i32Type;
}

Type getPointerTypeSameShape(Type type) {
  if (auto tensorTy = dyn_cast<RankedTensorType>(type)) {
    Type elementType = tensorTy.getElementType();
    PointerType ptrType = PointerType::get(elementType, 1);
    return tensorTy.clone(ptrType);
  } else {
    return PointerType::get(type, 1);
  }
}

Type getPointerTypeToElement(Type type) {
  Type elementType = getElementTypeOrSelf(type);
  PointerType ptrType = PointerType::get(elementType, 1);
  return ptrType;
}

// upstream Triton only uses address space 1 for Pointer Type
Type getPointerType(Type type, int addressSpace) {
  return PointerType::get(type, addressSpace);
}

int getAddressSpace(Type type) {
  if (auto ptrType = dyn_cast<PointerType>(type))
    return ptrType.getAddressSpace();
  return 1;
}

bool isTensorPointerType(Type type) {
  if (auto ptrType = dyn_cast<PointerType>(type))
    return isa<RankedTensorType>(ptrType.getPointeeType());
  return false;
}

bool isTensorOrTensorPointerType(Type type) {
  return isa<RankedTensorType>(type) || isTensorPointerType(type);
}

Type getElementTypeOfTensorPointerType(Type type) {
  if (auto ptrType = dyn_cast<PointerType>(type))
    if (auto tensorTy = dyn_cast<RankedTensorType>(ptrType.getPointeeType()))
      return tensorTy.getElementType();
  return {};
}

} // namespace triton

} // namespace mlir
