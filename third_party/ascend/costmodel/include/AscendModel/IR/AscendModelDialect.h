//===- AscendModelDialect.h - AscendModel dialect definition -----*- C++
//-*-===//
//
// This file defines the AscendModel dialect.
//
//===----------------------------------------------------------------------===//

#ifndef ASCEND_MODEL_DIALECT_H
#define ASCEND_MODEL_DIALECT_H

#include "mlir/Bytecode/BytecodeOpInterface.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/Interfaces/InferTypeOpInterface.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"

// Forward declarations
namespace mlir {
namespace ascend {
class HardwareConfig;
} // namespace ascend
} // namespace mlir

// Include interface header (contains include guard for .inc)
#include "AscendModel/IR/AscendModelInterfaces.h"

// Include generated dialect declaration
#include "AscendModel/IR/AscendModelDialect.h.inc"

// Include generated enum declarations
#include "AscendModel/IR/AscendModelOpsEnums.h.inc"

#define GET_ATTRDEF_CLASSES
#include "AscendModel/IR/AscendModelOpsAttrDefs.h.inc"

#define GET_TYPEDEF_CLASSES
#include "AscendModel/IR/AscendModelOpsTypes.h.inc"

// Include generated op declarations
#define GET_OP_CLASSES
#include "AscendModel/IR/AscendModelOps.h.inc"

#endif // ASCEND_MODEL_DIALECT_H
