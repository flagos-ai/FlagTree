#ifndef TRITON_TLE_RAW_IR_DIALECT_H_
#define TRITON_TLE_RAW_IR_DIALECT_H_

#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/LLVMIR/LLVMTypes.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"

#include "mctle/dialect/include/IR/Dialect.h.inc"

#define GET_ATTRDEF_CLASSES
#include "mctle/dialect/include/IR/AttrDefs.h.inc"

#define GET_OP_CLASSES
#include "mctle/dialect/include/IR/Ops.h.inc"
#endif
