#ifndef TRITON_DIALECT_ILUVATAR_TLE_IR_DIALECT_H_
#define TRITON_DIALECT_ILUVATAR_TLE_IR_DIALECT_H_

#ifdef __ILUVATAR_TLE__

#include "mlir/IR/Dialect.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/Triton/IR/OpInterfaces.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/IR/TritonGPUInterfaces.h"

#include "IR/Dialect.h.inc"

#define GET_OP_CLASSES
#include "IR/Ops.h.inc"

#endif // __ILUVATAR_TLE__

#endif // TRITON_DIALECT_ILUVATAR_TLE_IR_DIALECT_H_
