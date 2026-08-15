#ifndef TRITON_SHARED_DIALECT_XSMT_IR_XSMT_OPS_H
#define TRITON_SHARED_DIALECT_XSMT_IR_XSMT_OPS_H

#include "mlir/IR/Dialect.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/OpImplementation.h"

#include "triton-shared/Dialect/XSMT/IR/XSMTTypes.h"

#define GET_OP_CLASSES
#include "triton-shared/Dialect/XSMT/IR/XSMTOps.h.inc"
#undef GET_OP_CLASSES

#endif // TRITON_SHARED_DIALECT_XSMT_IR_XSMT_OPS_H
