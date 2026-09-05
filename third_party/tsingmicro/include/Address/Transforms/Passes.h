//===- Passes.h - Address passes  -------------------------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This header file defines prototypes that expose pass constructors.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_DIALECT_ADDRESS_TRANSFORMS_PASSES_H
#define MLIR_DIALECT_ADDRESS_TRANSFORMS_PASSES_H

#include "Address/Dialect/IR/AddressDialect.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Pass/Pass.h"
#include <memory>

namespace mlir {
namespace addr {
void populateBarePtrConvetion(RewritePatternSet &patterns);

#define GEN_PASS_DECL
#include "Address/Transforms/Passes.h.inc"

#define GEN_PASS_REGISTRATION
#include "Address/Transforms/Passes.h.inc"
} // namespace addr
} // namespace mlir

#endif // MLIR_DIALECT_ADDRESS_TRANSFORMS_PASSES_H
