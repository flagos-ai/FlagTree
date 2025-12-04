//===----------------------------------------------------------------------===//
//
// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.
//
//===----------------------------------------------------------------------===//

#ifndef TRITON_PTR_TO_MEMREF_CONVERSION_PASSES
#define TRITON_PTR_TO_MEMREF_CONVERSION_PASSES

include "mlir/Pass/PassBase.td"

def TritonPtrToMemref : Pass<"triton-ptr-to-memref", "mlir::ModuleOp"> {
  let summary = "Convert triton pointer to unranked memref";
  let constructor = "triton::createTritonPtrToMemrefPass()";
}

def ReconcilePtrCasts : Pass<"reconcile-ptr-casts", "mlir::ModuleOp"> {
  let summary = "Convert unrealized_cast op between address to memref to addr.to_memref or addr.from_memref";
  let constructor = "triton::createReconcilePtrCastsPass()";
}

def TritonPtrToAddress : Pass<"triton-ptr-to-address", "mlir::ModuleOp"> {
  let summary = "Convert Triton ops on pointers to the address dialect";
  let constructor = "triton::createTritonPtrToAddressPass()";
}

#endif
