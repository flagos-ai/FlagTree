//===----------------------------------------------------------------------===//
//
// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.
//
//===----------------------------------------------------------------------===//

#ifndef RECONCILE_PTR_CASTS_CONVERSION_PASSES
#define RECONCILE_PTR_CASTS_CONVERSION_PASSES

include "mlir/Pass/PassBase.td"

def ReconcilePtrCasts : Pass<"reconcile-ptr-casts", "mlir::ModuleOp"> {
  let summary = "Convert unrealized_cast op between tt.ptr or ptr.ptr to memref to to_memref or from_memref";
  let constructor = "triton::createReconcilePtrCastsPass()";
}

#endif
