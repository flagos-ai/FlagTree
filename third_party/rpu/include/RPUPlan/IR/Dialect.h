#pragma once

#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/OpDefinition.h"
#include "triton/Dialect/Triton/IR/Dialect.h"

#include "RPUPlan/IR/RPUPlanDialect.h.inc"

#define GET_OP_CLASSES
#include "RPUPlan/IR/RPUPlanOps.h.inc"
