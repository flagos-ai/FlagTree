#ifndef TRITON_METAX_COMMON_IR_LOWER_TENSOR_TILES_H
#define TRITON_METAX_COMMON_IR_LOWER_TENSOR_TILES_H

#include "mlir/IR/Operation.h"

namespace mlir::triton::metax {

bool isMctleExtractTile(Operation *op);
bool isMctleInsertTile(Operation *op);
bool isStaticMctleTile(Operation *op);

// Whether input and result can use encoding without layout conversions.
bool canLowerTensorTileInEncoding(Operation *op, Attribute encoding);

} // namespace mlir::triton::metax

#endif
