//===------------------- MagicKernelDialect.h -----------------*- C++ -*---===//
//
// Copyright (C) 2020-2025 Terapines Technology (Wuhan) Co., Ltd
// All rights reserved.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_DIALECT_MAGIC_KERNEL_IR_DIALECT_H_
#define MLIR_DIALECT_MAGIC_KERNEL_IR_DIALECT_H_

#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/IR/TypeSupport.h"
#include "mlir/IR/Types.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "triton/Dialect/Triton/IR/Dialect.h"

//===----------------------------------------------------------------------===//
// MagicKernel Operations
//===----------------------------------------------------------------------===//
#include "magic-kernel/Dialect/IR/MagicKernelDialect.h.inc"

// Include the auto-generated header file containing the declarations of the
// TritonStructured operations.
#define GET_OP_CLASSES
#include "magic-kernel/Dialect/IR/MagicKernelOps.h.inc"

#endif // MLIR_DIALECT_MAGIC_KERNEL_IR_DIALECT_H_
