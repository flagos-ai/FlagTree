#ifndef EVAS_DIALECT_LINALG_IR_LINALGOPSEXT_H
#define EVAS_DIALECT_LINALG_IR_LINALGOPSEXT_H

#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Bufferization/IR/BufferizableOpInterface.h"
#include "mlir/IR/DialectRegistry.h"

#include "evas/Dialect/Linalg/IR/LinalgOpsExtEnums.h.inc"

#define GET_OP_CLASSES
#include "evas/Dialect/Linalg/IR/LinalgOpsExt.h.inc"

namespace mlir::linalg {

void registerEvasLinalgOps(DialectRegistry &registry);

} // namespace mlir::linalg

#endif // EVAS_DIALECT_LINALG_IR_LINALGOPSEXT_H
