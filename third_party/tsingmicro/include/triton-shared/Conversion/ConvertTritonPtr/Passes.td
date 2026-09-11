//===----------------------------------------------------------------------===//
//
// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.
//
//===----------------------------------------------------------------------===//

#ifndef TRITON_PTR_TO_MEMREF_CONVERSION_PASSES
#define TRITON_PTR_TO_MEMREF_CONVERSION_PASSES

include "mlir/Pass/PassBase.td"

def TritonPtrToAddress : Pass<"triton-ptr-to-address", "mlir::ModuleOp"> {
  let summary = "Convert Triton ops on pointers to the address dialect";
  let constructor = "triton::createTritonPtrToAddressPass()";
}

#endif
