//===----------------------------------------------------------------------===//
//
// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.
//
//===----------------------------------------------------------------------===//

#ifndef TRITON_CONVERSION_TRITON_PTR_TO_MEMREF_RECONCILE_PTR_CASTS_H
#define TRITON_CONVERSION_TRITON_PTR_TO_MEMREF_RECONCILE_PTR_CASTS_H

#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"

namespace mlir {
namespace triton {

std::unique_ptr<OperationPass<ModuleOp>> createReconcilePtrCastsPass();

} // namespace triton
} // namespace mlir

#endif // TRITON_CONVERSION_TRITON_PTR_TO_MEMREF_RECONCILE_PTR_CASTS_H
