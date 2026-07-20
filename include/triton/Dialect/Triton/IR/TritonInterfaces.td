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

#ifndef TRITON_INTERFACES
#define TRITON_INTERFACES

include "mlir/IR/OpBase.td"
include "mlir/Interfaces/InferTypeOpInterface.td"

def TensorSizeTrait : NativeOpTrait<"TensorSizeTrait">;
def VerifyTensorLayoutsTrait : NativeOpTrait<"VerifyTensorLayoutsTrait">;
def SameOperandsEncoding : NativeOpTrait<"SameOperandsEncoding">;
def SameOperandsAndResultEncoding : NativeOpTrait<"SameOperandsAndResultEncoding">;
def SameLoadStoreOperandsShape : NativeOpTrait<"SameLoadStoreOperandsShape">;
def SameLoadStoreOperandsAndResultShape : NativeOpTrait<"SameLoadStoreOperandsAndResultShape">;
def SameLoadStoreOperandsEncoding : NativeOpTrait<"SameLoadStoreOperandsEncoding">;
def SameLoadStoreOperandsAndResultEncoding : NativeOpTrait<"SameLoadStoreOperandsAndResultEncoding">;
def AsyncRegions : NativeOpTrait<"AsyncRegions">;

// A trait equivalent to InferTypeOpAdaptor, but that checks for structural
// equivalence of the layouts of the result rather than just layout equality.
def InferTypeOpWithLayoutEquivalence : InferTypeOpAdaptorBase<[{
  static bool isCompatibleReturnTypes(TypeRange lhs, TypeRange rhs) {
    if (lhs.size() != rhs.size())
      return false;
    return llvm::all_of(llvm::zip(lhs, rhs), [](auto tup) {
      auto [lhs, rhs] = tup;
      return succeeded(OpTrait::impl::verifyEquivalentType(lhs, rhs));
    });
  }
}]>;

#endif // TRITON_INTERFACES
