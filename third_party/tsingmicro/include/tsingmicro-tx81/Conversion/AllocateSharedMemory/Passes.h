//===------------------- Passes.h -----------------------------*- C++ -*---===//
//
// Copyright (C) 2020-2025 Terapines Technology (Wuhan) Co., Ltd
// All rights reserved.
//
//===----------------------------------------------------------------------===//

#ifndef ALLOCATE_SHARED_MEMORY_PASSES_H
#define ALLOCATE_SHARED_MEMORY_PASSES_H

#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"
#include "triton/Dialect/Triton/IR/Dialect.h"

namespace mlir::triton::alloc {

#define GEN_PASS_DECL
#include "tsingmicro-tx81/Conversion/AllocateSharedMemory/Passes.h.inc"

#define GEN_PASS_REGISTRATION
#include "tsingmicro-tx81/Conversion/AllocateSharedMemory/Passes.h.inc"

} // namespace mlir::triton::alloc

#endif //  ALLOCATE_SHARED_MEMORY_PASSES_H
