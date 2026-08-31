//===------------------- Passes.h ----------------------------*- C++ -*----===//
//
// Copyright (C) 2020-2025 Terapines Technology (Wuhan) Co., Ltd
// All rights reserved.
//
//===----------------------------------------------------------------------===//
//
// TsingMicro Tx81 target-level IR transformations.
//
//===----------------------------------------------------------------------===//

#ifndef TX81_TRANSFORMS_PASSES_H
#define TX81_TRANSFORMS_PASSES_H

#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"

namespace mlir {
namespace triton {

std::unique_ptr<OperationPass<ModuleOp>> createInsertBarrierPass();
std::unique_ptr<OperationPass<ModuleOp>> createTx81ResolveDmaBaseAddrPass();

#define GEN_PASS_REGISTRATION
#define GEN_PASS_DECL
#include "tsingmicro-tx81/Transforms/Passes.h.inc"

} // namespace triton
} // namespace mlir

#endif // TX81_TRANSFORMS_PASSES_H
