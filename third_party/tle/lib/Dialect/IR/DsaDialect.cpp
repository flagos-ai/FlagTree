//===- DsaDialect.cpp - TLE DSA dialect -------------------------*- C++ -*-===//
//
// Template dialect for TLE-Struct style DSA extensions.
//
//===----------------------------------------------------------------------===//

#include "tle-dsa/Dialect/IR/DsaDialect.h"

#include "mlir/IR/DialectImplementation.h"
#include "llvm/ADT/TypeSwitch.h"

using namespace mlir;

namespace mlir::dsa {

void DsaDialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include "tle-dsa/Dialect/IR/DsaOps.cpp.inc"
      >();
  registerTypes();
}

void DsaDialect::registerTypes() {
  addTypes<
#define GET_TYPEDEF_LIST
#include "tle-dsa/Dialect/IR/DsaOpsTypes.cpp.inc"
      >();
}

} // namespace mlir::dsa

#include "tle-dsa/Dialect/IR/DsaOpsDialect.cpp.inc"

#define GET_OP_CLASSES
#include "tle-dsa/Dialect/IR/DsaOps.cpp.inc"

#define GET_TYPEDEF_CLASSES
#include "tle-dsa/Dialect/IR/DsaOpsTypes.cpp.inc"
