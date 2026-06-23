// Copyright 2026- Xcoresigma Technology Co., Ltd

#include "tle/dsa/dialect/include/IR/Dialect.h"
#include "mlir/Support/LLVM.h"

#define GET_ATTRDEF_CLASSES
#include "tle/dsa/dialect/include/IR/TleDSAAttrDefs.cpp.inc"

#define GET_OP_CLASSES
#include "tle/dsa/dialect/include/IR/TleDSAOps.cpp.inc"

namespace mlir::triton::tle {
void TleDialect::dsaInitialize() {
  addAttributes<
#define GET_ATTRDEF_LIST
#include "tle/dsa/dialect/include/IR/TleDSAAttrDefs.cpp.inc"
      >();

  addOperations<
#define GET_OP_LIST
#include "tle/dsa/dialect/include/IR/TleDSAOps.cpp.inc"
      >();
}
} // namespace mlir::triton::tle
