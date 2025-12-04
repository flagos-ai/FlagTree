//===- AddressDialect.h - Address dialect -----------------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines the Address dialect.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_DIALECT_ADDRESS_IR_ADDRESSDIALECT_H
#define MLIR_DIALECT_ADDRESS_IR_ADDRESSDIALECT_H

#include "mlir/Bytecode/BytecodeOpInterface.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/Interfaces/CastInterfaces.h"
#include "mlir/Interfaces/InferTypeOpInterface.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"

#include "Address/Dialect/IR/AddressOpsDialect.h.inc"

namespace mlir {
class PatternRewriter;
}

#define GET_TYPEDEF_CLASSES
#include "Address/Dialect/IR/AddressOpsTypes.h.inc"

#define GET_OP_CLASSES
#include "Address/Dialect/IR/AddressOps.h.inc"

#endif // MLIR_DIALECT_ADDRESS_IR_ADDRESSDIALECT_H
