//===----------------------------------------------------------------------===//
//
// SPDX-FileCopyrightText: Copyright (c) 2026 SpacemiT. All rights reserved.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

#ifndef TRITON_CONVERSION_TRITONTOLINALG_LoopPtrCarryToOffset_H
#define TRITON_CONVERSION_TRITONTOLINALG_LoopPtrCarryToOffset_H

#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"

namespace mlir {
namespace triton {

std::unique_ptr<OperationPass<ModuleOp>> createLoopPtrCarryToOffsetPass();

} // namespace triton
} // namespace mlir

#endif // TRITON_CONVERSION_TRITONTOLINALG_LoopPtrCarryToOffset_H
