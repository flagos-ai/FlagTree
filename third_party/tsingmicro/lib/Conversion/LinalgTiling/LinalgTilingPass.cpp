//===------------------- LinalgTilingPass.cpp -----------------------------===//
//
// Copyright (C) 2020-2025 Terapines Technology (Wuhan) Co., Ltd
// All rights reserved.
//
//===----------------------------------------------------------------------===//
//
// This file implements the pass infrastructure for linalg tiling
// transformations. The pass applies tiling patterns to improve performance of
// linalg operations.
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "tsingmicro-tx81/Conversion/LinalgTiling/LinalgTiling.h"
#include <memory>
#include <mlir/IR/DialectRegistry.h>
#include <mlir/Transforms/Passes.h>

#define DEBUG_TYPE "linalg-tiling"

using namespace mlir;

namespace mlir {
namespace triton {

#define GEN_PASS_DEF_LINALGTILING
#include "tsingmicro-tx81/Conversion/LinalgTiling/Passes.h.inc"
} // namespace triton
} // namespace mlir

namespace {

class LinalgTilingPass
    : public triton::impl::LinalgTilingBase<LinalgTilingPass> {
public:
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<linalg::LinalgDialect>();
  }

  void runOnOperation() override {
    ModuleOp module = getOperation();
    MLIRContext *context = &getContext();

    RewritePatternSet patterns(context);

    mlir::triton::populateLinalgTilingPatterns(patterns);

    if (failed(applyPatternsGreedily(module, std::move(patterns)))) {
      signalPassFailure();
    }
  }
};
} // namespace

std::unique_ptr<OperationPass<ModuleOp>>
mlir::triton::createLinalgTilingPass() {
  return std::make_unique<LinalgTilingPass>();
}
