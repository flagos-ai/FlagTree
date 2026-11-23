//===- DsaDialect.h - TLE DSA dialect ---------------------------*- C++ -*-===//
//
// Template dialect for TLE-Struct style DSA extensions.
//
//===----------------------------------------------------------------------===//

#ifndef TLE_DSA_DIALECT_IR_DSADIALECT_H
#define TLE_DSA_DIALECT_IR_DSADIALECT_H

#include "mlir/Bytecode/BytecodeOpInterface.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/Interfaces/DestinationStyleOpInterface.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"

// DsaOps.td uses TT_Tensor / TT_Ptr / TT_Int type constraints from the
// Triton dialect, so the generated verifiers need these types visible.
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/Triton/IR/Types.h"

#include "tle-dsa/Dialect/IR/DsaOpsDialect.h.inc"

namespace mlir {
class PatternRewriter;
} // namespace mlir

#define GET_TYPEDEF_CLASSES
#include "tle-dsa/Dialect/IR/DsaOpsTypes.h.inc"

#define GET_OP_CLASSES
#include "tle-dsa/Dialect/IR/DsaOps.h.inc"

#endif // TLE_DSA_DIALECT_IR_DSADIALECT_H
