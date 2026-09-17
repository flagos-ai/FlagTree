//===------------------- Passes.h -----------------------------*- C++ -*---===//
//
// Copyright (C) 2020-2025 Terapines Technology (Ludt) Co., Ltd
// All rights reserved.
//
//===----------------------------------------------------------------------===//

#ifndef EXPORT_KERNEL_SYMBOLS_CONVERSION_PASSES_H
#define EXPORT_KERNEL_SYMBOLS_CONVERSION_PASSES_H

#include "tsingmicro-tx81/Conversion/ExportKernelSymbols/ExportKernelSymbols.h"

namespace mlir {
namespace triton {

#define GEN_PASS_REGISTRATION
#include "tsingmicro-tx81/Conversion/ExportKernelSymbols/Passes.h.inc"

} // namespace triton
} // namespace mlir

#endif //  EXPORT_KERNEL_SYMBOLS_CONVERSION_PASSES_H
