#pragma once

#include "mlir/Bytecode/BytecodeOpInterface.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/Region.h"

#include "Debugger/IR/Dialect.h.inc"

#define GET_OP_CLASSES
#include "Debugger/IR/Ops.h.inc"
