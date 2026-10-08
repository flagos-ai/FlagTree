#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/DialectImplementation.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/Support/LLVM.h"

// clang-format off
#include "Dialect/ThvTile/IR/Dialect.h"
#include "Dialect/ThvTile/IR/Dialect.cpp.inc"
#include "llvm/ADT/TypeSwitch.h"
// clang-format on

using namespace mlir;
using namespace mlir::thvtile;

void mlir::thvtile::ThvTileDialect::initialize() {
  addTypes<
#define GET_TYPEDEF_LIST
#include "Dialect/ThvTile/IR/Types.cpp.inc"
      >();

  addAttributes<
#define GET_ATTRDEF_LIST
#include "Dialect/ThvTile/IR/ThvTileAttrDefs.cpp.inc"
      >();

  addOperations<
#define GET_OP_LIST
#include "Dialect/ThvTile/IR/Ops.cpp.inc"
      >();
}

#define GET_TYPEDEF_CLASSES
#include "Dialect/ThvTile/IR/Types.cpp.inc"

#define GET_ATTRDEF_CLASSES
#include "Dialect/ThvTile/IR/ThvTileAttrDefs.cpp.inc"

#include "Dialect/ThvTile/IR/ThvTileEnums.cpp.inc"
