//===------------------- LinalgTiling.h ------------------------*- C++-*---===//
//
//
//===----------------------------------------------------------------------===//
//
// This file implements the patterns to tile linalg operations for better
// performance. It applies tiling transformations to improve data locality
// and parallelism.
//
//===----------------------------------------------------------------------===//

#ifndef TRITON_CONVERSION_LINALG_TILING_H
#define TRITON_CONVERSION_LINALG_TILING_H

#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"

namespace mlir {
namespace triton {

#define GEN_PASS_DECL
#include "tsingmicro-tx81/Conversion/LinalgTiling/Passes.h.inc"

void populateLinalgTilingPatterns(RewritePatternSet &patterns);

std::unique_ptr<OperationPass<ModuleOp>> createLinalgTilingPass();

} // namespace triton
} // namespace mlir

#endif // TRITON_CONVERSION_LINALG_TILING_H
