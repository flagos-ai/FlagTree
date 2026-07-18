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

#ifndef RPU_PLAN_DIALECT
#define RPU_PLAN_DIALECT

include "mlir/IR/OpBase.td"

def RPUPlan_Dialect : Dialect {
  let name = "rpu_plan";
  let cppNamespace = "::mlir::rpu::plan";

  let summary = "Internal RPU kernel plan dialect";

  let description = [{
    RPUPlan is an internal compiler dialect. It records the structured plan
    produced by RPU TTIR recognition before the plan is exported as stable
    rpuplan JSON or consumed by later RPU DSL emission passes.
  }];

  let dependentDialects = [
    "mlir::triton::TritonDialect"
  ];
}

#endif
