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

#ifndef ILUVATAR_TLE_DIALECT
#define ILUVATAR_TLE_DIALECT

include "mlir/IR/OpBase.td"

#ifdef __ILUVATAR_TLE__
def IluvatarTle_Dialect : Dialect {
  let name = "iluvatar_tle";
  let cppNamespace = "::mlir::triton::iluvatar_tle";
  let description = [{
    Iluvatar backend-local Triton Language Extension dialect.
  }];

  let dependentDialects = [
    "triton::TritonDialect",
    "triton::gpu::TritonGPUDialect",
  ];
  let usePropertiesForAttributes = 1;
}
#endif // __ILUVATAR_TLE__

#endif // ILUVATAR_TLE_DIALECT
