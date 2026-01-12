//===------------------- ExportKernelSymbols.h -------------------------*- C++
//-*---===//
//
// Copyright (C) 2020-2025 Terapines Technology (Ludt) Co., Ltd
// All rights reserved.
//
//===----------------------------------------------------------------------===//

#ifndef EXPORT_KERNEL_SYMBOLS_CONVERSION_H
#define EXPORT_KERNEL_SYMBOLS_CONVERSION_H

#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"

namespace mlir {
namespace triton {

#define GEN_PASS_DECL
#include "tsingmicro-tx81/Conversion/ExportKernelSymbols/Passes.h.inc"

std::unique_ptr<OperationPass<ModuleOp>> createExportKernelSymbolsPass();

} // namespace triton
} // namespace mlir

#endif // EXPORT_KERNEL_SYMBOLS_CONVERSION_H
