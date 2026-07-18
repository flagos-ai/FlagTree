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

#ifndef TRITON_TYPE_INTERFACES
#define TRITON_TYPE_INTERFACES

include "mlir/IR/OpBase.td"

//===----------------------------------------------------------------------===//
// TensorDescInterface
//===----------------------------------------------------------------------===//

def TT_TensorDescInterface : TypeInterface<"TensorDescInterface"> {
  let cppNamespace = "::mlir::triton";

  let description = [{
    Common interface for tensor descriptor types.

    This interface provides a unified API for different tensor descriptor
    implementations (e.g., tiled TensorDescType, im2col TensorDescIm2ColType).
    All tensor descriptors share the concept of a "block type" which describes
    the shape and element type of the data block being accessed.

    Concrete implementations:
    - TensorDescType (Triton dialect): Basic tiled tensor descriptor
    - TensorDescIm2ColType (TritonNvidiaGPU dialect): Im2col tensor descriptor
      with additional convolution parameters
  }];

  let methods = [
    InterfaceMethod<
      /*desc=*/"Returns the block type of the tensor descriptor",
      /*retType=*/"mlir::RankedTensorType",
      /*methodName=*/"getBlockType",
      /*args=*/(ins)
    >,
    InterfaceMethod<
      /*desc=*/"Returns the block type with signless integer element type",
      /*retType=*/"mlir::RankedTensorType",
      /*methodName=*/"getSignlessBlockType",
      /*args=*/(ins),
      /*methodBody=*/"",
      /*defaultImpl=*/[{
        auto resTy = $_type.getBlockType();
        if (auto intTy = llvm::dyn_cast<mlir::IntegerType>(resTy.getElementType())) {
          auto width = resTy.getElementTypeBitWidth();
          auto signlessTy = mlir::IntegerType::get($_type.getContext(), width);
          resTy = resTy.clone(signlessTy);
        }
        return resTy;
      }]
    >,
  ];
}

#endif // TRITON_TYPE_INTERFACES
