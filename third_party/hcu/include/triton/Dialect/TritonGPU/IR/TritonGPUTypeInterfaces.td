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

#ifndef TRITON_GPU_TYPE_INTERFACES
#define TRITON_GPU_TYPE_INTERFACES

include "mlir/IR/OpBase.td"

// Interface dynamically attached to RankedTensorType and MemDescType.
def TTG_TensorOrMemDesc : TypeInterface<"TensorOrMemDesc"> {
  let cppNamespace = "::mlir::triton::gpu";
  let methods = [
    InterfaceMethod<"Returns the encoding of the tensor or memory descriptor",
      "mlir::Attribute", "getEncoding", (ins)>,
    InterfaceMethod<"Returns element type",
      "mlir::Type", "getElementType", (ins)>,
    InterfaceMethod<"Returns the type shape",
      "llvm::ArrayRef<int64_t>", "getShape", (ins)>,
    InterfaceMethod<"Returns the tensor or buffer rank",
      "int64_t", "getRank", (ins)>,
    InterfaceMethod<"Returns the element type bit width",
      "int64_t", "getElementTypeBitWidth", (ins)>,
  ];
}

#endif // TRITON_GPU_TYPE_INTERFACES
