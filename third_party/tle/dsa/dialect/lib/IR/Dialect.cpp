// Copyright 2026- Xcoresigma Technology Co., Ltd

#include "tle/dsa/dialect/include/IR/Dialect.h"
#include "mlir/Support/LLVM.h"
#include "tle/dsa/dialect/include/IR/Dialect.cpp.inc"

#define GET_ATTRDEF_CLASSES
#include "tle/dsa/dialect/include/IR/TleAttrDefs.cpp.inc"

#define GET_OP_CLASSES
#include "tle/dsa/dialect/include/IR/TleOps.cpp.inc"

namespace mlir::triton::tle {
void TleDialect::initialize() {
  addAttributes<
#define GET_ATTRDEF_LIST
#include "tle/dsa/dialect/include/IR/TleAttrDefs.cpp.inc"
      >();

  addOperations<
#define GET_OP_LIST
#include "tle/dsa/dialect/include/IR/TleOps.cpp.inc"
      >();
}
} // namespace mlir::triton::tle
