//===------------------- TLEToMK.h -------------------------*- C++ -*---===//
//
// Copyright (C) 2020-2025 Terapines Technology (Wuhan) Co., Ltd
// All rights reserved.
//
//===----------------------------------------------------------------------===//
//
// Lowering TLE communication ops into mk ops.
//
//===----------------------------------------------------------------------===//

#ifndef ZTC_CONVERSION_TLE_TO_MK_H
#define ZTC_CONVERSION_TLE_TO_MK_H

#include "magic-kernel/Dialect/IR/MagicKernelDialect.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"

namespace mlir {
namespace triton {

#define GEN_PASS_DECL
#include "magic-kernel/Conversion/TLEToMK/Passes.h.inc"

void populateTLEToMKConversionPatterns(RewritePatternSet &patterns);

// std::unique_ptr<OperationPass<ModuleOp>> createTLEToMKPass();

} // namespace triton
} // namespace mlir

#endif // ZTC_CONVERSION_TLE_TO_MK_H
