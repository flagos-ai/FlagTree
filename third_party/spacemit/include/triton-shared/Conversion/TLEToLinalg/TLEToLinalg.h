//===----------------------------------------------------------------------===//
//
// SPDX-FileCopyrightText: Copyright (c) 2025 SpacemiT. All rights reserved.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

#ifndef TRITON_CONVERSION_TLETOLINALG_TLETOLINALG_H
#define TRITON_CONVERSION_TLETOLINALG_TLETOLINALG_H

#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"

namespace mlir {
namespace triton {

#define GEN_PASS_DECL
#include "triton-shared/Conversion/TLEToLinalg/Passes.h.inc"

void populateTLEToLinalgConversionPatterns(RewritePatternSet &patterns);

std::unique_ptr<OperationPass<ModuleOp>> createTLEToLinalgPass();

} // namespace triton
} // namespace mlir

#endif // TRITON_CONVERSION_TLETOLINALG_TLETOLINALG_H
