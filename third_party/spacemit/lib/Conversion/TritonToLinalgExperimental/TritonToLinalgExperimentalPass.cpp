//===----------------------------------------------------------------------===//
//
// SPDX-FileCopyrightText: Copyright (c) 2025 SpacemiT. ALL rights reserved.
// SPDX-FileCopyrightText: Copyright (c) Microsoft Corporation. All rights
// reserved. SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/DLTI/DLTI.h"
#include "mlir/Dialect/Ptr/IR/PtrDialect.h"
#include "proton/Dialect/include/Dialect/Proton/IR/Dialect.h"
#include "triton-shared/Conversion/AddTargetDescription/AddTargetDescription.h"
#include "triton-shared/Conversion/StructuredToMemref/StructuredToMemref.h"
#include "triton-shared/Conversion/TLEToLinalg/TLEToLinalg.h"
#include "triton-shared/Conversion/TritonArithToLinalg/TritonArithToLinalg.h"
#include "triton-shared/Conversion/TritonPtrToMemref/TritonPtrToMemref.h"
#include "triton-shared/Conversion/TritonToLinalgExperimental/CollapseShape.h"
#include "triton-shared/Conversion/TritonToLinalgExperimental/ConvertScanOp.h"
#include "triton-shared/Conversion/TritonToLinalgExperimental/ReconcileLlvmPtrCasts.h"
#include "triton-shared/Conversion/TritonToLinalgExperimental/ReconcilePtrCasts.h"
#include "triton-shared/Conversion/TritonToLinalgExperimental/ScfbufferStandardized.h"
#include "triton-shared/Conversion/TritonToLinalgExperimental/TritonToLinalgExperimental.h"
#include "triton-shared/Conversion/TritonToLinalgExperimental/TritonToPtr.h"
#include "triton-shared/Conversion/TritonToStructured/TritonToStructured.h"
#include "triton-shared/Conversion/TritonToUnstructured/TritonToUnstructured.h"
#include "triton-shared/Conversion/UnstructuredToMemref/UnstructuredToMemref.h"
#include "triton-shared/Conversion/XSMTToLinalg/XSMTToLinalg.h"
#include "triton-shared/Dialect/TritonStructured/IR/TritonStructuredDialect.h"
#include "triton-shared/Dialect/TritonTilingExt/IR/TritonTilingExtDialect.h"

#include "mlir/Conversion/ReconcileUnrealizedCasts/ReconcileUnrealizedCasts.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Transforms/Passes.h"
#include "triton-shared/Dialect/XSMT/IR/XSMTDialect.h"

using namespace mlir;
using namespace triton;

namespace mlir::triton {
#define GEN_PASS_DECL_TRITONTOLINALGEXPERIMENTAL
#define GEN_PASS_DEF_TRITONTOLINALGEXPERIMENTAL
#include "triton-shared/Conversion/TritonToLinalgExperimental/Passes.h.inc"
} // namespace mlir::triton

namespace {

class TritonToLinalgExperimentalPass
    : public triton::impl::TritonToLinalgExperimentalBase<
          TritonToLinalgExperimentalPass> {

public:
  void getDependentDialects(DialectRegistry &registry) const override {
    registry
        .insert<func::FuncDialect, arith::ArithDialect, math::MathDialect,
                linalg::LinalgDialect, affine::AffineDialect, scf::SCFDialect,
                tensor::TensorDialect, bufferization::BufferizationDialect,
                memref::MemRefDialect, ttx::TritonTilingExtDialect,
                tts::TritonStructuredDialect, ptr::PtrDialect, DLTIDialect,
                LLVM::LLVMDialect, vector::VectorDialect, xsmt::XSMTDialect,
                triton::proton::ProtonDialect>();
  }

  void runOnOperation() override {
    auto moduleOp = getOperation();
    PassManager pm(&getContext(), moduleOp.getOperationName());

    // Lower tt.scan before Triton-to-structured / PtrAnalysis.
    //
    // PtrAnalysis only understands a limited set of value-producing ops for
    // masked load/store operands. When a masked tt.store consumes a tt.scan
    // result directly, rewriteStoreOp ends up rejecting the value as produced
    // by an unsupported instruction. Converting tt.scan early turns the scan
    // into vector/scf/tensor ops before pointer analysis runs.
    //
    // This also avoids relying on later passes to legalize scans after the
    // structured-pointer pipeline has already seen them.
    pm.addPass(createConvertScanOpPass());

    pm.addPass(createTritonToStructuredPass(enableMakeGatherScatterTensorPtr));

    // Erase dead code and fold constants created during lowering
    pm.addPass(createCSEPass());
    pm.addPass(createCanonicalizerPass());

    pm.addPass(createTritonToUnstructuredPass());
    pm.addPass(createTritonArithToLinalgPass(/*tensorPtrToLinalg=*/true));
    pm.addPass(createScfbufferStandardizedPass());

    pm.addPass(createStructuredToMemrefPass());
    pm.addPass(createUnstructuredToMemrefPass());
    pm.addPass(createTritonPtrToMemrefPass());
    pm.addPass(createTritonToPtrPass());
    pm.addPass(createAddTargetDescriptionPass());
    // Now that remove-dead-values fully works with linalg ops, clean up the IR
    // again, particularly unused loop iter-args that were created
    // during triton-to-structured.
    pm.addPass(createRemoveDeadValuesPass());
    pm.addPass(createXSMTToLinalgPass());
    pm.addPass(createTLEToLinalgPass());
    pm.addPass(createReconcileUnrealizedCastsPass());
    pm.addPass(createReconcilePtrCastsPass());
    pm.addPass(createReconcileLlvmPtrCastsPass());

    pm.addPass(createCSEPass());
    pm.addPass(createCanonicalizerPass());
    if (enableCollapseShape) {
      // TODO Enable this
      // Canonicalizer pass will rewrite tensor.expand_shape(linalg.fill) to
      // linalg.fill(tensor.expand_shape) so we need to run it before
      // collapseShape pass
      // pm.addPass(createCollapseShapePass());
    }

    if (failed(runPipeline(pm, getOperation()))) {
      signalPassFailure();
    }

    llvm::SmallPtrSet<mlir::Operation *, 16> maybeDeadCasts;
    moduleOp.walk([&](linalg::Mmt4DOp mmt4dOp) {
      for (mlir::OpOperand &operand : mmt4dOp->getOpOperands()) {
        auto castOp = operand.get().getDefiningOp<mlir::tensor::CastOp>();
        if (!castOp)
          continue;
        operand.set(castOp.getSource());
        maybeDeadCasts.insert(castOp.getOperation());
      }
    });

    for (mlir::Operation *op : maybeDeadCasts) {
      if (op->use_empty())
        op->erase();
    }
  }
};
} // namespace

std::unique_ptr<OperationPass<ModuleOp>>
triton::createTritonToLinalgExperimentalPass() {
  return std::make_unique<TritonToLinalgExperimentalPass>();
}
