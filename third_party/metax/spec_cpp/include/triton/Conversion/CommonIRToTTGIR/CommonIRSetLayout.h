#ifndef TRITON_METAX_COMMON_IR_SET_LAYOUT_H
#define TRITON_METAX_COMMON_IR_SET_LAYOUT_H

#include "mlir/IR/BuiltinOps.h"

namespace mlir::triton::metax {

// Apply explicit CommonIR layout hints and insert required conversions.
LogicalResult applyCommonIRLayoutHints(ModuleOp module);

} // namespace mlir::triton::metax

#endif
