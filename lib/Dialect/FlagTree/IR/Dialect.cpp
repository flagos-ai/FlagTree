#include "triton/Dialect/FlagTree/IR/Dialect.h"

#include "mlir/Support/LLVM.h"

#include "triton/Dialect/FlagTree/IR/Dialect.cpp.inc"

#define GET_ATTRDEF_CLASSES
#include "triton/Dialect/FlagTree/IR/FlagTreeAttrDefs.cpp.inc"

#define GET_OP_CLASSES
#include "triton/Dialect/FlagTree/IR/Ops.cpp.inc"

namespace mlir::triton::flagtree {
void FlagTreeDialect::initialize() {
  addAttributes<
#define GET_ATTRDEF_LIST
#include "triton/Dialect/FlagTree/IR/FlagTreeAttrDefs.cpp.inc"
      >();
  addOperations<
#define GET_OP_LIST
#include "triton/Dialect/FlagTree/IR/Ops.cpp.inc"
      >();
}
} // namespace mlir::triton::flagtree
