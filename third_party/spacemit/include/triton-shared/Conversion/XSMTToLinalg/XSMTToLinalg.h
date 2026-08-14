//===----------------------------------------------------------------------===//
//
// SPDX-FileCopyrightText: Copyright (c) 2025 SpacemiT. All rights reserved.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

#ifndef TRITON_CONVERSION_XSMTTOLINALG_XSMTTOLINALG_H
#define TRITON_CONVERSION_XSMTTOLINALG_XSMTTOLINALG_H

#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"

#include "triton/Dialect/Triton/IR/Dialect.h"

namespace mlir {
namespace triton {

#define GEN_PASS_DECL
#include "triton-shared/Conversion/XSMTToLinalg/Passes.h.inc"

/// MBarrier checks + MBarrier release insertion.
void populateXSMTValidationPatterns(RewritePatternSet &patterns);

/// Core xsmt ops → linalg/tensor/memref conversion,
/// loop parallelization, and Proton record lowering.
void populateXSMTConversionPatterns(RewritePatternSet &patterns);

/// Cleanup patterns after xsmt bufferization/conversion.
void populateXSMTBufferizationCleanupPatterns(RewritePatternSet &patterns);

std::unique_ptr<OperationPass<ModuleOp>> createXSMTToLinalgPass();

} // namespace triton
} // namespace mlir

#endif // TRITON_CONVERSION_XSMTTOLINALG_XSMTTOLINALG_H
