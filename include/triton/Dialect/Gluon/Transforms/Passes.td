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

#ifndef GLUON_PASSES
#define GLUON_PASSES

include "mlir/Pass/PassBase.td"

def GluonResolveAutoEncodingsPass : Pass<"gluon-resolve-auto-encodings", "mlir::ModuleOp"> {
  let summary = "Resolve automatic encodings";
  let dependentDialects = [
    "mlir::triton::gpu::TritonGPUDialect",
  ];
}

def GluonInferCoalescedEncodingsPass : Pass<"gluon-infer-coalesced-encodings", "mlir::ModuleOp"> {
  let summary = "Infer coalesced encodings based on axis analysis";
  let dependentDialects = [
    "mlir::triton::gpu::TritonGPUDialect",
  ];
}

def GluonCanonicalize: Pass<"gluon-canonicalize"> {
  let summary = "reduced set of simplifications for TTGIR";

  let description = [{
    The `gluon-canonicalize` pass applies a reduced set of simplification
    and canonicalization patterns to the module.
  }];
  let dependentDialects = [
    "mlir::arith::ArithDialect",
    "mlir::cf::ControlFlowDialect",
    "mlir::scf::SCFDialect",
  ];
}

def GluonInline: Pass<"gluon-inline"> {
  let summary = "reduced set of simplifications for TTGIR";

  let description = [{
    The `gluon-inline` pass applies a reduced set of simplification
    and canonicalization patterns to the module.
  }];
  let dependentDialects = [];
}

def GluonSimplifyControlFlow: Pass<"gluon-slimplify-control-flow"> {
  let summary = "simplications for control flow ops";

  let description = [{
    The `gluon-inline` pass applies a reduced set of simplification
    and canonicalization patterns to the module.
  }];
  let dependentDialects = [];
}

#endif
