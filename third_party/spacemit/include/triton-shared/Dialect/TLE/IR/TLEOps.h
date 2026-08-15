#ifndef TRITON_SHARED_DIALECT_TLE_IR_OPS_H
#define TRITON_SHARED_DIALECT_TLE_IR_OPS_H

#include "mlir/IR/Dialect.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/Interfaces/InferTypeOpInterface.h"

#define GET_OP_CLASSES
#include "triton-shared/Dialect/TLE/IR/TLEOps.h.inc"
#undef GET_OP_CLASSES

#endif // TRITON_SHARED_DIALECT_TLE_IR_OPS_H
