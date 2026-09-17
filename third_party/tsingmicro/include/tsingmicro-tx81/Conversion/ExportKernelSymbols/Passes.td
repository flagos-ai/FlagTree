//===------------------- Passes.td ----------------------------*- C++ -*---===//
//
// Copyright (C) 2020-2025 Terapines Technology (Ludt) Co., Ltd
// All rights reserved.
//
//===----------------------------------------------------------------------===//

#ifndef EXPORT_KERNEL_SYMBOLS_CONVERSION_PASSES
#define EXPORT_KERNEL_SYMBOLS_CONVERSION_PASSES

include "mlir/Pass/PassBase.td"

def ExportKernelSymbols : Pass<"export-kernel-symbols", "mlir::ModuleOp"> {
  let summary = "Export kernel function to a dynamic symbol table that can be accessed at runtime.";
  let constructor = "triton::createExportKernelSymbolsPass()";
  let options = [];
  let dependentDialects = ["LLVM::LLVMDialect"];
}

#endif
