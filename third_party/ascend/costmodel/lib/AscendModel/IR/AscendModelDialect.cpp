//===- AscendModelDialect.cpp - AscendModel dialect implementation --------===//
//
// This file implements the AscendModel dialect.
//
//===----------------------------------------------------------------------===//

#include "AscendModel/IR/AscendModelDialect.h"

#include "mlir/IR/Builders.h"
#include "mlir/IR/DialectImplementation.h"
#include "mlir/IR/OpImplementation.h"
#include "llvm/ADT/TypeSwitch.h"

using namespace mlir;
using namespace mlir::ascend;

//===----------------------------------------------------------------------===//
// AscendModel Dialect
//===----------------------------------------------------------------------===//

#include "AscendModel/IR/AscendModelDialect.cpp.inc"

void AscendModelDialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include "AscendModel/IR/AscendModelOps.cpp.inc"
      >();
}

//===----------------------------------------------------------------------===//
// Enum Definitions
//===----------------------------------------------------------------------===//

#include "AscendModel/IR/AscendModelOpsEnums.cpp.inc"

//===----------------------------------------------------------------------===//
// Attribute Definitions
//===----------------------------------------------------------------------===//

#define GET_ATTRDEF_CLASSES
#include "AscendModel/IR/AscendModelOpsAttrDefs.cpp.inc"

//===----------------------------------------------------------------------===//
// Type Definitions
//===----------------------------------------------------------------------===//

#define GET_TYPEDEF_CLASSES
#include "AscendModel/IR/AscendModelOpsTypes.cpp.inc"
