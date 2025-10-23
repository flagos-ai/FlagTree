// TODO: When upgrading to Triton 3.4.0, remove this file and use the upstream Triton file.
#ifndef TRITON_IR_OP_INTERFACES_H_
#define TRITON_IR_OP_INTERFACES_H_

#include "mlir/IR/OpDefinition.h"
#include "triton/Dialect/Triton/IR/Types.h"

namespace mlir {

namespace triton {

namespace impl {

LogicalResult verifyTransposeOpInterface(Operation *op);

LogicalResult verifyDotOpInterface(Operation *op);

} // namespace impl

} // namespace triton
} // namespace mlir

#include "triton/Dialect/Triton/IR/OpInterfaces.h.inc"

#endif // TRITON_IR_OP_INTERFACES_H_
