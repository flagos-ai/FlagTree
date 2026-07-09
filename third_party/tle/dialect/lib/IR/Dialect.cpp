#include "tle/dialect/include/IR/Dialect.h"
#include "mlir/Support/LLVM.h"
#include "tle/dialect/include/IR/Dialect.cpp.inc"

#define GET_ATTRDEF_CLASSES
#include "tle/dialect/include/IR/TleAttrDefs.cpp.inc"

#define GET_OP_CLASSES
#include "tle/dialect/include/IR/Ops.cpp.inc"

#ifdef FLAGCX_ENABLED
#define GET_OP_CLASSES
#include "tle/dialect/include/IR/FlagCxOps.cpp.inc"
#endif

namespace mlir::triton::tle {
void TleDialect::initialize() {
  addAttributes<
#define GET_ATTRDEF_LIST
#include "tle/dialect/include/IR/TleAttrDefs.cpp.inc"
      >();
  addOperations<
#define GET_OP_LIST
#include "tle/dialect/include/IR/Ops.cpp.inc"
      >();

#ifdef FLAGCX_ENABLED
  addOperations<
#define GET_OP_LIST
#include "tle/dialect/include/IR/FlagCxOps.cpp.inc"
      >();
#endif
}
} // namespace mlir::triton::tle
