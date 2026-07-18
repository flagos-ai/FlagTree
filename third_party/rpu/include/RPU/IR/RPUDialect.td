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

#ifndef RPU_EXEC_DIALECT
#define RPU_EXEC_DIALECT

include "mlir/IR/OpBase.td"

def RPU_Dialect : Dialect {
  let name = "rpu";
  let cppNamespace = "::mlir::rpu::exec";

  let summary = "Internal executable RPU dialect";

  let description = [{
    The `rpu` dialect is an internal executable IR used between RPUPlan and
    RPU DSL source emission. It is not a user-facing Triton API and is not the
    RPU C++ DSL API.
  }];

  let dependentDialects = [
    "mlir::triton::TritonDialect"
  ];

  let useDefaultTypePrinterParser = 1;
}

#endif
