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

#ifndef TRITON_AMDGPU_OP_INTERFACES
#define TRITON_AMDGPU_OP_INTERFACES

include "mlir/IR/OpBase.td"

def BufferOpInterface : OpInterface<"BufferOpInterface"> {
  let description = [{
    This interface is implemented by buffer load/store operations.
    It provides methods to access common properties such base pointer, offset, mask and others.
  }];

  let cppNamespace = "::mlir::triton::amdgpu";

  let methods = [
    InterfaceMethod<
      /*desc=*/"Get operation base ptr.",
      /*retType=*/"::mlir::TypedValue<::mlir::triton::PointerType>",
      /*methodName=*/"getPtr">,
    InterfaceMethod<
      /*desc=*/"Get mutable operation base ptr.",
      /*retType=*/"::mlir::OpOperand &",
      /*methodName=*/"getPtrMutable">,
    InterfaceMethod<
      /*desc=*/"Get operation offset tensor.",
      /*retType=*/"::mlir::TypedValue<::mlir::TensorType>",
      /*methodName=*/"getOffsets">,
    InterfaceMethod<
      /*desc=*/"Get mutable operation offset tensor.",
      /*retType=*/"::mlir::OpOperand &",
      /*methodName=*/"getOffsetsMutable">,
    InterfaceMethod<
      /*desc=*/"Get operation stride.",
      /*retType=*/"::mlir::TypedValue<::mlir::IntegerType>",
      /*methodName=*/"getStride">,
    InterfaceMethod<
      /*desc=*/"Get mutable operation stride.",
      /*retType=*/"::mlir::MutableOperandRange ",
      /*methodName=*/"getStrideMutable">
  ];
}

#endif // TRITON_AMDGPU_OP_INTERFACES
