// Copyright 2026 FlagOS Contributors
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.

#ifndef TRITONGPU_ENUMS
#define TRITONGPU_ENUMS

include "mlir/IR/EnumAttr.td"
include "triton/Dialect/TritonGPU/IR/TritonGPUDialect.td"

// Bitmask enum describing which memory domains a barrier/fence orders.
def TTG_AddrSpace : I32BitEnumAttr<
    "AddrSpace", "",
    [
      I32BitEnumAttrCase<"None", 0b0000, "none">,
      I32BitEnumAttrCase<"Local", 0b0001, "local">,
      I32BitEnumAttrCase<"GlobalRead", 0b0010, "global_read">,
      I32BitEnumAttrCase<"GlobalWrite", 0b0100, "global_write">,
      I32BitEnumAttrCase<"TensorRead", 0b1000, "tensor_read">,
      I32BitEnumAttrCase<"TensorWrite", 0b10000, "tensor_write">,
      I32BitEnumAttrCase<"All", 0b11111, "all">
    ]> {
  let cppNamespace = "::mlir::triton::gpu";
}

#endif // TRITONGPU_ENUMS
