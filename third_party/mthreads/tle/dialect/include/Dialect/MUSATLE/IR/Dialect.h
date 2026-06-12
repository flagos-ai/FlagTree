#ifndef TRITON_DIALECT_MUSATLE_IR_DIALECT_H_
#define TRITON_DIALECT_MUSATLE_IR_DIALECT_H_

#ifdef __TLE__

#include "mlir/IR/Dialect.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/Triton/IR/OpInterfaces.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/IR/TritonGPUInterfaces.h"

// clang-format off
#include "Dialect/MUSATLE/IR/Dialect.h.inc"
// clang-format on

#define GET_OP_CLASSES
#include "Dialect/MUSATLE/IR/Ops.h.inc"

#endif // __TLE__

#endif // TRITON_DIALECT_MUSATLE_IR_DIALECT_H_
