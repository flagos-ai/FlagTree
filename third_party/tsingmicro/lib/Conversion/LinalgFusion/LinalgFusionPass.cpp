//===------------------- LinalgFusionPass.cpp -----------------------------===//
//
// Copyright (C) 2020-2025 Terapines Technology (Wuhan) Co., Ltd
// All rights reserved.
//
//===----------------------------------------------------------------------===//
//
// This file implements the pass infrastructure for linalg fusion
// transformations. The pass applies fusion patterns to improve performance of
// linalg operations.
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "tsingmicro-tx81/Conversion/LinalgFusion/LinalgFusion.h"
#include <memory>
#include <mlir/IR/DialectRegistry.h>
#include <mlir/Transforms/Passes.h>

#define DEBUG_TYPE "linalg-fusion"

using namespace mlir;

namespace mlir {
namespace triton {

#define GEN_PASS_DEF_LINALGFUSION
#include "tsingmicro-tx81/Conversion/LinalgFusion/Passes.h.inc"
} // namespace triton
} // namespace mlir

namespace {

class LinalgFusionPass
    : public triton::impl::LinalgFusionBase<LinalgFusionPass> {
public:
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<linalg::LinalgDialect>();
    registry.insert<mk::MagicKernelDialect>();
  }

  void runOnOperation() override {
    ModuleOp module = getOperation();
    MLIRContext *context = &getContext();
    {
      RewritePatternSet selfPatterns(context);

      mlir::triton::populateLinalgBinaryOpFusionPatterns(selfPatterns);

      if (failed(applyPatternsGreedily(module, std::move(selfPatterns)))) {
        signalPassFailure();
      }
    }
  }
};
} // namespace

std::unique_ptr<OperationPass<ModuleOp>>
mlir::triton::createLinalgFusionPass() {
  return std::make_unique<LinalgFusionPass>();
}
