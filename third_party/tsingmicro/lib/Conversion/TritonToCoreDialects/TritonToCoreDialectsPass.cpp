//===----------------------------------------------------------------------===//
//
// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.
//
//===----------------------------------------------------------------------===//

#include "Address/Dialect/IR/AddressDialect.h"
#include "magic-kernel/Dialect/IR/MagicKernelDialect.h"
#include "triton-shared/Conversion/ConvertTritonPtr/ReconcilePtrCasts.h"
#include "triton-shared/Conversion/ConvertTritonPtr/TritonPtrToAddress.h"
#include "triton-shared/Conversion/ConvertTritonPtr/TritonPtrToMemref.h"
#include "triton-shared/Conversion/StructuredToMemref/StructuredToMemref.h"
#include "triton-shared/Conversion/TritonArithToLinalg/TritonArithToLinalg.h"
#include "triton-shared/Conversion/TritonPtrToMemref/TritonPtrToMemref.h"
#include "triton-shared/Conversion/TritonToCoreDialects/TritonToCoreDialects.h"
#include "triton-shared/Conversion/TritonToStructured/TritonToStructured.h"
#include "triton-shared/Conversion/TritonToUnstructured/TritonToUnstructured.h"
#include "triton-shared/Conversion/UnstructuredToMemref/UnstructuredToMemref.h"
#include "triton-shared/Dialect/TritonStructured/IR/TritonStructuredDialect.h"
#include "triton-shared/Dialect/TritonTilingExt/IR/TritonTilingExtDialect.h"

#include "mlir/Conversion/ReconcileUnrealizedCasts/ReconcileUnrealizedCasts.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"

#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/Passes.h"

using namespace mlir;
using namespace triton;

#define GEN_PASS_CLASSES
#include "triton-shared/Conversion/TritonToCoreDialects/Passes.h.inc"

namespace {

class TritonToCoreDialectsPass
    : public TritonToCoreDialectsBase<TritonToCoreDialectsPass> {

public:
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<func::FuncDialect, arith::ArithDialect, math::MathDialect,
                    linalg::LinalgDialect, affine::AffineDialect,
                    scf::SCFDialect, tensor::TensorDialect,
                    bufferization::BufferizationDialect, memref::MemRefDialect,
                    ttx::TritonTilingExtDialect, tts::TritonStructuredDialect,
                    mk::MagicKernelDialect, addr::AddressDialect>();
  }

  void runOnOperation() override {
    auto moduleOp = getOperation();
    PassManager pm(&getContext(), moduleOp.getOperationName());
    pm.addPass(createTritonToStructuredPass());

    // Erase dead code and fold constants created during lowering
    pm.addPass(createCSEPass());
    pm.addPass(createCanonicalizerPass());
    pm.addPass(createTritonToUnstructuredPass());

    pm.addPass(createTritonArithToLinalgPass());
    pm.addPass(createStructuredToMemrefPass());

    pm.addPass(createCSEPass());
    pm.addPass(createCanonicalizerPass());

    pm.addPass(createUnstructuredToMemrefPass());

    pm.addPass(createCSEPass());
    pm.addPass(createCanonicalizerPass());

    // Convert triton pointers to memref + address dialect
    // TODO: Un-ranked memref will all converted to address dialect pointers
    pm.addPass(createTritonPtrToMemrefPass());
    pm.addPass(createTritonPtrToAddressPass());
    pm.addPass(createReconcileUnrealizedCastsPass());
    pm.addPass(createReconcilePtrCastsPass());

    // FIXME: RemoveDeadValuesPass is not working now
    // pm.addPass(createRemoveDeadValuesPass());
    pm.addPass(createCSEPass());
    pm.addPass(createCanonicalizerPass());

    pm.addPass(createTritonPtrToMemrefPass());

    if (failed(runPipeline(pm, getOperation()))) {
      signalPassFailure();
    }
  }
};
} // namespace

std::unique_ptr<OperationPass<ModuleOp>>
triton::createTritonToCoreDialectsPass() {
  return std::make_unique<TritonToCoreDialectsPass>();
}
