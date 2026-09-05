//===------------------- TLEToMKPass.cpp -------------------------===//
//
// Copyright (C) 2020-2025 Terapines Technology (Wuhan) Co., Ltd
// All rights reserved.
//
//===----------------------------------------------------------------------===//
//
// Lowering TLE communication ops to backend dialects
//
//===----------------------------------------------------------------------===//

#include "magic-kernel/Conversion/TLEToMK/TLEToMK.h"
#include "magic-kernel/Dialect/IR/MagicKernelDialect.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "tle/dialect/include/IR/Dialect.h"
#include "tle/include/tle-dsa/Dialect/IR/DsaDialect.h"
#include "triton-shared/Dialect/TritonStructured/IR/TritonStructuredDialect.h"
#include "triton/Dialect/Triton/IR/Dialect.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "mlir/Transforms/Passes.h"

using namespace mlir;
using namespace triton;

#define GEN_PASS_CLASSES
#include "magic-kernel/Conversion/TLEToMK/Passes.h.inc"

namespace {

class TLEToMKPass : public TLEToMKBase<TLEToMKPass> {

public:
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<func::FuncDialect, mk::MagicKernelDialect,
                    mlir::dsa::DsaDialect, tts::TritonStructuredDialect,
                    triton::TritonDialect, arith::ArithDialect,
                    memref::MemRefDialect, tensor::TensorDialect>();
  }

  void runOnOperation() override {
    auto moduleOp = getOperation();
    RewritePatternSet patterns(&getContext());
    populateTLEToMKConversionPatterns(patterns);

    if (failed(applyPatternsGreedily(moduleOp, std::move(patterns)))) {
      signalPassFailure();
      return;
    }

    // Postcondition: these ops are owned by this pass; a survivor means an
    // unlowerable op reached the pipeline (e.g. reverse cumsum).
    WalkResult bad = moduleOp.walk([&](Operation *op) -> WalkResult {
      if (isa<mlir::triton::tle::ExclusiveCumsumOp,
              mlir::triton::tle::ExtractTileOp, mlir::triton::tle::InsertTileOp,
              mlir::dsa::CumsumOp>(op)) {
        op->emitError("not lowered by tle-to-mk");
        return WalkResult::interrupt();
      }
      return WalkResult::advance();
    });
    if (bad.wasInterrupted())
      signalPassFailure();
  }
};
} // namespace

std::unique_ptr<Pass> triton::createTLEToMK() {
  return std::make_unique<TLEToMKPass>();
}
