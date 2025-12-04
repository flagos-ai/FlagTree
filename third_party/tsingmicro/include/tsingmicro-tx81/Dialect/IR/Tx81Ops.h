//===-------------------------- Tx81Ops.h ---------------------*- C++ -*---===//
//
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_DIALECT_TSINGMICRO_TX81_IR_OPS_H
#define MLIR_DIALECT_TSINGMICRO_TX81_IR_OPS_H

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

#define GET_OP_CLASSES
#include "tsingmicro-tx81/Dialect/IR/Tx81Enums.h.inc"
#include "tsingmicro-tx81/Dialect/IR/Tx81Ops.h.inc"

#endif // MLIR_DIALECT_TSINGMICRO_TX81_IR_DIALECT_H
