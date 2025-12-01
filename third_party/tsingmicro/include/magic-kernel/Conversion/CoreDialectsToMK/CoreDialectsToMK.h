//===------------------- CoreDialectsToMK.h -------------------*- C++ -*---===//
//
// Copyright (C) 2020-2025 Terapines Technology (Wuhan) Co., Ltd
// All rights reserved.
//
//===----------------------------------------------------------------------===//
//
// This pass is the wrap all pass that populates all the conversion patterns
// from core dialects such as linalg, memref, buf etc to mk dialect.
//
//===----------------------------------------------------------------------===//

#ifndef TRITON_CONVERSION_CORE_DIALECTS_TO_MK_H
#define TRITON_CONVERSION_CORE_DIALECTS_TO_MK_H

#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"

namespace mlir {
namespace triton {

std::unique_ptr<OperationPass<ModuleOp>> createCoreDialectsToMKPass();

} // namespace triton
} // namespace mlir

#endif // TRITON_CONVERSION_CORE_DIALECTS_TO_MK_H
