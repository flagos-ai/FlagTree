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

#ifndef GLUON_OPS
#define GLUON_OPS

include "triton/Dialect/Gluon/IR/GluonDialect.td"
include "triton/Dialect/Gluon/IR/GluonAttrDefs.td"
include "triton/Dialect/Triton/IR/TritonInterfaces.td"
include "triton/Dialect/Triton/IR/TritonTypes.td"

class Gluon_Op<string mnemonic, list<Trait> traits = []> :
    Op<Gluon_Dialect, mnemonic,
       !listconcat(traits, [VerifyTensorLayoutsTrait])> {
}

def Gluon_SetAutoLayoutOp : Gluon_Op<"set_auto_layout",
                                 [SameOperandsAndResultShape,
                                  SameOperandsAndResultElementType]> {
  let summary = "set auto encoding to a concrete encoding type";

  let arguments = (ins TT_Tensor:$src);

  let results = (outs TT_Tensor:$result);

  let builders = [
    OpBuilder<(ins "Attribute":$encoding, "Value":$value)>
  ];

  let hasVerifier = 1;

  let assemblyFormat = "$src attr-dict `:` type($src) `->` type($result)";
}

#endif // GLUON_OPS
