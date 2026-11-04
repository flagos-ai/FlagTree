#ifndef TRITON_METAX_COMMON_IR_TENSOR_OPS_H
#define TRITON_METAX_COMMON_IR_TENSOR_OPS_H

#include "mlir/Transforms/DialectConversion.h"

namespace mlir::triton::metax {

// Whether input and result can use encoding without layout conversions.
bool canLowerTensorTileInEncoding(Operation *op, Attribute encoding);

LogicalResult applyCommonIRLayoutHints(ModuleOp module);

void populateCommonIRTensorPatternsAndLegality(TypeConverter &typeConverter,
                                               RewritePatternSet &patterns,
                                               ConversionTarget &target);

} // namespace mlir::triton::metax

#endif
