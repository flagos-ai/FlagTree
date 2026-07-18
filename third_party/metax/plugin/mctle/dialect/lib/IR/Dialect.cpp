#include "mctle/dialect/include/IR/Dialect.h"
#include "mctle/dialect/include/IR/Dialect.cpp.inc"
#include "mlir/Support/LLVM.h"

#define GET_ATTRDEF_CLASSES
#include "mctle/dialect/include/IR/AttrDefs.cpp.inc"

#define GET_OP_CLASSES
#include "mctle/dialect/include/IR/Ops.cpp.inc"

namespace mlir::triton::mctle {
void McTleDialect::initialize() {
  addAttributes<
#define GET_ATTRDEF_LIST
#include "mctle/dialect/include/IR/AttrDefs.cpp.inc"
      >();
  addOperations<
#define GET_OP_LIST
#include "mctle/dialect/include/IR/Ops.cpp.inc"
      >();
}
} // namespace mlir::triton::mctle
