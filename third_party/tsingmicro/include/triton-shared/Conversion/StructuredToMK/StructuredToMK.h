#ifndef TRITON_CONVERSION_STRUCTUREDTOMK_StructuredToMK_H
#define TRITON_CONVERSION_STRUCTUREDTOMK_StructuredToMK_H

#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"

#include "triton/Dialect/Triton/IR/Dialect.h"

namespace mlir {
class TypeConverter;
namespace triton {

#define GEN_PASS_DECL
#include "triton-shared/Conversion/StructuredToMK/Passes.h.inc"

void populateStructuredToMKConversionPatterns(RewritePatternSet &patterns,
                                              TypeConverter &typeConverter);

std::unique_ptr<OperationPass<ModuleOp>> createStructuredToMKPass();

} // namespace triton
} // namespace mlir

#endif // TRITON_CONVERSION_StructuredToMK_StructuredToMK_H
