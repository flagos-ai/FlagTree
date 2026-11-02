#ifndef DIALECT_THVTILE_IR_DIALECT_H_
#define DIALECT_THVTILE_IR_DIALECT_H_

#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Interfaces/CallInterfaces.h"
#include "mlir/Interfaces/ControlFlowInterfaces.h"
#include "mlir/Interfaces/DestinationStyleOpInterface.h"
#include "mlir/Interfaces/FunctionInterfaces.h"
#include "mlir/Interfaces/InferTypeOpInterface.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"

// clang-format off
#include "Dialect/ThvTile/IR/Dialect.h.inc"
#include "Dialect/ThvTile/IR/ThvTileEnums.h.inc"
// clang-format on

#define GET_TYPEDEF_CLASSES
#include "Dialect/ThvTile/IR/Types.h.inc"

#define GET_ATTRDEF_CLASSES
#include "Dialect/ThvTile/IR/ThvTileAttrDefs.h.inc"

#define GET_OP_CLASSES
#include "Dialect/ThvTile/IR/Ops.h.inc"

namespace mlir {
namespace thvtile {} // namespace thvtile
} // namespace mlir

#endif // DIALECT_THVTILE_IR_DIALECT_H_
