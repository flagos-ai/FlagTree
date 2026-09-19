#ifndef TRITON_DIALECT_ILUVATAR_TLE_IR_DIALECT_H_
#define TRITON_DIALECT_ILUVATAR_TLE_IR_DIALECT_H_

#ifdef __ILUVATAR_TLE__

#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/LLVMIR/LLVMTypes.h"
#include "mlir/IR/Dialect.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/Triton/IR/OpInterfaces.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/IR/TritonGPUInterfaces.h"

#include "IR/Dialect.h.inc"

#define GET_OP_CLASSES
#include "IR/Ops.h.inc"

#ifdef FLAGCX_ENABLED
#define GET_OP_CLASSES
#include "IR/FlagCxOps.h.inc"
#endif

#endif // __ILUVATAR_TLE__

#endif // TRITON_DIALECT_ILUVATAR_TLE_IR_DIALECT_H_
