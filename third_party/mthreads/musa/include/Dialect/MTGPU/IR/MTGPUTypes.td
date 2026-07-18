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

#ifndef MTGPU_TYPES
#define MTGPU_TYPES

include "mlir/IR/AttrTypeBase.td"
include "MTGPUDialect.td"

class MTGPU_TypeDef<string name, string _mnemonic, list<Trait> traits = []>
    : TypeDef<MTGPUDialect, name, traits> {
  let mnemonic = _mnemonic;
}

def MTGPU_SqmmaAccumulatorType
    : MTGPU_TypeDef<"SqmmaAccumulator", "sqmma_accumulator"> {
  let summary = "backend-private SQMMA accumulator carrier";
  let description = [{
    A backend-private carrier type that preserves SQMMA loop-carried
    accumulators in compact fragment form while keeping the wrapped tensor
    type as the logical accumulator semantics.
  }];

  let parameters = (ins "Type":$tensorType);
  let assemblyFormat = "`<` $tensorType `>`";
  let genVerifyDecl = 1;

  let extraClassDeclaration = [{
    RankedTensorType getAccumulatorType() const {
      return llvm::cast<RankedTensorType>(getTensorType());
    }
  }];
}

#endif
