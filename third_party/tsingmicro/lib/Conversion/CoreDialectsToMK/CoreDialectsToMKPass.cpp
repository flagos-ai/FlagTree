//===------------------- CoreDialectsToMKPass.cpp -------------------------===//
//
// Copyright (C) 2020-2025 Terapines Technology (Wuhan) Co., Ltd
// All rights reserved.
//
//===----------------------------------------------------------------------===//
//
// Lowering core dialects to backend dialects
//
//===----------------------------------------------------------------------===//

#include "magic-kernel/Conversion/CoreDialectsToMK/CoreDialectsToMK.h"
#include "magic-kernel/Conversion/LinalgToMK/LinalgToMK.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"

#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/Passes.h"

using namespace mlir;
using namespace triton;

#define GEN_PASS_CLASSES
#include "magic-kernel/Conversion/CoreDialectsToMK/Passes.h.inc"
#include "magic-kernel/Dialect/IR/MagicKernelDialect.h"

namespace {

class CoreDialectsToMKPass : public CoreDialectsToMKBase<CoreDialectsToMKPass> {

public:
  void getDependentDialects(DialectRegistry &registry) const override {
    registry
        .insert<func::FuncDialect, arith::ArithDialect, math::MathDialect,
                linalg::LinalgDialect, affine::AffineDialect, scf::SCFDialect,
                tensor::TensorDialect, bufferization::BufferizationDialect,
                memref::MemRefDialect, mk::MagicKernelDialect>();
  }

  void runOnOperation() override {
    auto moduleOp = getOperation();
    PassManager pm(&getContext(), moduleOp.getOperationName());

    pm.addPass(createLinalgToMKPass());

    // Erase dead code and fold constants created during lowering
    pm.addPass(createCSEPass());
    pm.addPass(createCanonicalizerPass());

    if (failed(runPipeline(pm, getOperation()))) {
      signalPassFailure();
    }
  }
};
} // namespace

std::unique_ptr<OperationPass<ModuleOp>> triton::createCoreDialectsToMKPass() {
  return std::make_unique<CoreDialectsToMKPass>();
}
