//===----------------------------------------------------------------------===//
//
// SPDX-FileCopyrightText: Copyright (c) 2025 SpacemiT. All rights reserved.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

#include "triton-shared/Conversion/TLEToLinalg/TLEToLinalg.h"
#include "triton-shared/Dialect/TLE/IR/TLEDialect.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#define DEBUG_TYPE "tle-to-linalg"

using namespace mlir;
using namespace triton;

namespace mlir {
namespace triton {
#define GEN_PASS_DEF_TLETOLINALG
#include "triton-shared/Conversion/TLEToLinalg/Passes.h.inc"
} // namespace triton
} // namespace mlir

namespace {

class TLEToLinalgPass : public triton::impl::TLEToLinalgBase<TLEToLinalgPass> {
  using TLEToLinalgBase<TLEToLinalgPass>::TLEToLinalgBase;

public:
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<arith::ArithDialect, tensor::TensorDialect,
                    mlir::tle::TLEDialect>();
  }

  void runOnOperation() override {
    auto moduleOp = getOperation();

    // Quick check: skip if no TLE ops present.
    bool hasTLEOp = false;
    moduleOp.walk([&](Operation *op) {
      if (op->getName().getStringRef().starts_with("xtle.")) {
        hasTLEOp = true;
        return WalkResult::interrupt();
      }
      return WalkResult::advance();
    });
    if (!hasTLEOp)
      return;

    RewritePatternSet patterns(&getContext());
    triton::populateTLEToLinalgConversionPatterns(patterns);
    if (failed(applyPatternsGreedily(moduleOp, std::move(patterns))))
      signalPassFailure();
  }
};

} // namespace

std::unique_ptr<OperationPass<ModuleOp>> triton::createTLEToLinalgPass() {
  return std::make_unique<TLEToLinalgPass>();
}
