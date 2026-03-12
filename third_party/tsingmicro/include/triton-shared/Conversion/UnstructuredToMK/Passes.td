//===----------------------------------------------------------------------===//
//
// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.
//
//===----------------------------------------------------------------------===//

#ifndef UNSTRUCTURED_TO_MK_CONVERSION_PASSES
#define UNSTRUCTURED_TO_MK_CONVERSION_PASSES

include "mlir/Pass/PassBase.td"

def UnstructuredToMK : Pass<"unstructured-to-mk", "mlir::ModuleOp"> {
  let summary = "Convert unstructured triton ptr (gather / scatter) to mk";
  let constructor = "triton::createUnstructuredToMKPass()";
}

#endif
