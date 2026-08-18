//===----------------------------------------------------------------------===//
//
// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.
//
//===----------------------------------------------------------------------===//

#ifndef TRITON_TRANSFORM_ADDLLVMDEBUGINFO_ADDLLVMDEBUGINFO_H
#define TRITON_TRANSFORM_ADDLLVMDEBUGINFO_ADDLLVMDEBUGINFO_H

#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"

#include "triton/Dialect/Triton/IR/Dialect.h"

namespace mlir {
namespace triton {

std::unique_ptr<OperationPass<ModuleOp>> createAddLLVMDebugInfoPass();

} // namespace triton
} // namespace mlir

#endif // TRITON_TRANSFORM_ADDLLVMDEBUGINFO_ADDLLVMDEBUGINFO_H
