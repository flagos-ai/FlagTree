//===-------------------------- Tx81Dialect.cpp ---------------------------===//
//
//
//===----------------------------------------------------------------------===//

#include "tsingmicro-tx81/Dialect/IR/Tx81Dialect.h"

using namespace mlir;
using namespace mlir::tx;

/// Dialect creation, the instance will be owned by the context. This is the
/// point of registration of custom types and operations for the dialect.
void Tx81Dialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include "tsingmicro-tx81/Dialect/IR/Tx81Ops.cpp.inc"
      >();
}

//===----------------------------------------------------------------------===//
// TableGen'd op method definitions
//===----------------------------------------------------------------------===//

#define GET_OP_CLASSES
#include "tsingmicro-tx81/Dialect/IR/Tx81Enums.cpp.inc"
#include "tsingmicro-tx81/Dialect/IR/Tx81Ops.cpp.inc"

#include "tsingmicro-tx81/Dialect/IR/Tx81Dialect.cpp.inc"
