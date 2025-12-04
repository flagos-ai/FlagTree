//===----------------------------------------------------------------------===//
//
// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.
//
//===----------------------------------------------------------------------===//

#ifndef TRITON_TO_CORE_DIALECTS_CONVERSION_PASSES
#define TRITON_TO_CORE_DIALECTS_CONVERSION_PASSES

include "mlir/Pass/PassBase.td"

def TritonToCoreDialects : Pass<"triton-to-core-dialects", "mlir::ModuleOp"> {
  let summary = "Convert Triton to core dialects including Linalg, Memref etc";
  let constructor = "triton::createTritonToCoreDialectsPass()";
}

#endif
