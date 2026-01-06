//===------------------- LinalgToMK.h -------------------------*- C++ -*---===//
//
// Copyright (C) 2020-2025 Terapines Technology (Wuhan) Co., Ltd
// All rights reserved.
//
//===----------------------------------------------------------------------===//
//
// Lowering all linalg ops into mk ops.
//
//===----------------------------------------------------------------------===//

#ifndef ZTC_CONVERSION_LINALG_TO_MK_H
#define ZTC_CONVERSION_LINALG_TO_MK_H

#include "magic-kernel/Dialect/IR/MagicKernelDialect.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "utils/utils.h"

namespace mlir {
namespace triton {

#define GEN_PASS_DECL
#include "magic-kernel/Conversion/LinalgToMK/Passes.h.inc"

// Fusion, identity reduction init, etc. Other ops which need to decompose into
// multiple integer type operators also are converted.
void populateLinalgToMKPreProcessPatterns(RewritePatternSet &patterns);

// Type conversion: trans integer type to float type, etc.
void populateLinalgToMKTypeConversionPatterns(RewritePatternSet &patterns,
                                              bool precisionPriority = false);

// Pattern rewrite to target dependent operation
void populateLinalgToMKCanonicalizationPatterns(RewritePatternSet &patterns,
                                                bool precisionPriority = false);

// Reshape input shape to destination shape
void populateLinalgToMKShapeCanonicalizationPatterns(
    RewritePatternSet &patterns, bool precisionPriority = false);

// Convertion patterns
void populateLinalgToMKConversionPatterns(RewritePatternSet &patterns);

std::unique_ptr<OperationPass<ModuleOp>> createLinalgToMKPass();
std::unique_ptr<OperationPass<ModuleOp>>
createLinalgToMKPass(LinalgToMKOptions &options);

} // namespace triton
} // namespace mlir

namespace {

using namespace mlir;
using namespace triton;

// Extract the operations from a linalg op region
template <typename T> static bool checkGenericOp(linalg::GenericOp op) {
  auto regionBlock = op.getBody();
  auto regionOps = llvm::map_to_vector(regionBlock->without_terminator(),
                                       [](Operation &op) { return &op; });

  return regionOps.size() == 1 && isa<T>(regionOps[0]);
}

static bool isConstantTensor(Value &v, double targetValue,
                             bool isApprox = false);

// Check if the given value is a tensor filled with 0.
static bool isZeroTensor(Value &v);

// Check if the given value is a tensor filled with 1.
static bool isOneTensor(Value &v);

static bool isHalfTensor(Value &v);

static bool isTwoTensor(Value &v);

} // namespace

#endif // ZTC_CONVERSION_MEMREF_TO_MAGICKERNEL_H
