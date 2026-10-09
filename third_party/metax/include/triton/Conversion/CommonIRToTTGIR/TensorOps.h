#ifndef TRITON_METAX_COMMON_IR_TENSOR_OPS_H
#define TRITON_METAX_COMMON_IR_TENSOR_OPS_H

#include "mlir/Transforms/DialectConversion.h"

namespace mlir::triton::metax {

// Read-only query sharing the physical planner used by tensor tile lowering.
// True means both tensors can use encoding without boundary conversions.
bool canLowerTensorTileInEncoding(Operation *op, Attribute encoding);

void populateCommonIRTensorPatternsAndLegality(TypeConverter &typeConverter,
                                               RewritePatternSet &patterns,
                                               ConversionTarget &target);

} // namespace mlir::triton::metax

#endif
