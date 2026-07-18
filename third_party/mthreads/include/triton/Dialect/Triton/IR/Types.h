// Copyright 2026 FlagOS Contributors
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.

#ifndef TRITON_IR_TYPES_H_
#define TRITON_IR_TYPES_H_

#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/TypeSupport.h"
#include "mlir/IR/Types.h"

#define GET_TYPEDEF_CLASSES
#include "triton/Dialect/Triton/IR/TypeInterfaces.h.inc"

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

} // namespace triton

} // namespace mlir

#endif // TRITON_IR_TYPES_H_
