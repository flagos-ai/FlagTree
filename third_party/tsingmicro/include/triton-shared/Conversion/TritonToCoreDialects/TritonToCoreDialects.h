//===----------------------------------------------------------------------===//
//
// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.
//
//===----------------------------------------------------------------------===//
//
// This pass is the wrapall pass that populates all the conversion patterns from
// triton to core dialects such as linalg, memref, buf etc.
//
//===----------------------------------------------------------------------===//

#ifndef TRITON_CONVERSION_TRITON_TO_CORE_DIALECTS_H
#define TRITON_CONVERSION_TRITON_TO_CORE_DIALECTS_H

#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"

namespace mlir {
namespace triton {

std::unique_ptr<OperationPass<ModuleOp>> createTritonToCoreDialectsPass();

} // namespace triton
} // namespace mlir

#endif // TRITON_CONVERSION_TRITON_TO_CORE_DIALECTS_H
