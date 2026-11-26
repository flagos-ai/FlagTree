#ifndef TRITON_CONVERSION_TRITONARITHTOLINALG_TRITONARITHTOLINALG_H
#define TRITON_CONVERSION_TRITONARITHTOLINALG_TRITONARITHTOLINALG_H

#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"

#include "triton/Dialect/Triton/IR/Dialect.h"

namespace mlir {
namespace triton {

#define GEN_PASS_DECL
#include "triton-shared/Conversion/TritonArithToLinalg/Passes.h.inc"

void populateTritonArithToLinalgCanonicalizationPatterns(
    RewritePatternSet &patterns);

void populateTritonArithToLinalgConversionPatterns(bool pidsToFuncArgs,
                                                   bool addptrToLinalg,
                                                   bool assertToCf,
                                                   RewritePatternSet &patterns);

std::unique_ptr<OperationPass<ModuleOp>> createTritonArithToLinalgPass();

} // namespace triton
} // namespace mlir

#endif // TRITON_CONVERSION_TRITONARITHTOLINALG_TRITONARITHTOLINALG_H
