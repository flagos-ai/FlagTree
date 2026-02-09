/*
 * Copyright (c) Huawei Technologies Co.
 * Licensed under the MIT license.
 */

#ifndef TRITON_TO_HIVM_CONVERSION_PASSES
#define TRITON_TO_HIVM_CONVERSION_PASSES

include "mlir/Pass/PassBase.td"

def TritonToHIVM : Pass<"triton-to-hivm", "mlir::ModuleOp"> {
  let summary = "Convert Triton to HIVM dialect";
  let constructor = "triton::createTritonToHIVMPass()";
  let dependentDialects = ["hivm::HIVMDialect"];
}

#endif // TRITON_TO_HIVM_CONVERSION_PASSES
