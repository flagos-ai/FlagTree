#ifndef TRITON_DIALECT_MUSA_IR_DIALECT_H_
#define TRITON_DIALECT_MUSA_IR_DIALECT_H_

#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/IR/Dialect.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/Triton/IR/OpInterfaces.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/IR/TritonGPUInterfaces.h"
#include "llvm/ADT/StringRef.h"

// clang-format off
#include "Dialect/MUSA/IR/Dialect.h.inc"
#include "Dialect/MUSA/IR/OpsEnums.h.inc"
// clang-format on

#define GET_ATTRDEF_CLASSES
#include "Dialect/MUSA/IR/MUSAAttrDefs.h.inc"

#define GET_OP_CLASSES
#include "Dialect/MUSA/IR/Ops.h.inc"

#endif // TRITON_DIALECT_MUSA_IR_DIALECT_H_
