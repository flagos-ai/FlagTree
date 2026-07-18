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

#ifndef RPU_PLAN_OPS
#define RPU_PLAN_OPS

include "RPUPlan/IR/RPUPlanDialect.td"
include "mlir/IR/OpBase.td"
include "mlir/IR/SymbolInterfaces.td"

class RPUPlan_Op<string mnemonic, list<Trait> traits = []> :
    Op<RPUPlan_Dialect, mnemonic, traits>;

def RPUPlan_KernelOp : RPUPlan_Op<"kernel", [Symbol]> {
  let summary = "Internal RPU kernel plan";

  let description = [{
    `rpu_plan.kernel` is a regionless symbol op that describes one recognized
    RPU kernel plan. It is not executable IR and does not model tile, load,
    store, memory-scope, or async-pipe behavior.
  }];

  let arguments = (ins
    SymbolNameAttr:$sym_name,
    StrAttr:$kernel_name,
    FlatSymbolRefAttr:$source_func,
    I32Attr:$version,
    StrAttr:$pattern,
    DictionaryAttr:$signature,
    DictionaryAttr:$shape,
    DictionaryAttr:$args,
    DictionaryAttr:$layout,
    DictionaryAttr:$mask,
    ArrayAttr:$required_dsl_features,
    DictionaryAttr:$emission
  );

  let assemblyFormat = "$sym_name attr-dict";
  let hasVerifier = 1;
}

#endif
