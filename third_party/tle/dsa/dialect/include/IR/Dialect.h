// Copyright 2026- Xcoresigma Technology Co., Ltd

#ifndef TRITON_TLE_IR_DIALECT_H_
#define TRITON_TLE_IR_DIALECT_H_

#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/LLVMIR/LLVMTypes.h"
#include "triton/Dialect/Triton/IR/Dialect.h"

#include "tle/dsa/dialect/include/IR/Dialect.h.inc"

#define GET_ATTRDEF_CLASSES
#include "tle/dsa/dialect/include/IR/TleAttrDefs.h.inc"

#define GET_OP_CLASSES
#include "tle/dsa/dialect/include/IR/TleOps.h.inc"

#endif
