//===--------------------- MKToTx81Pass.cpp -------------------------------===//
//
//
//===----------------------------------------------------------------------===//

#include "magic-kernel/Dialect/IR/MagicKernelDialect.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "tsingmicro-tx81/Conversion/MKToTx81/MKToTx81.h"
#include "tsingmicro-tx81/Dialect/IR/Tx81Dialect.h"
#include "utils/utils.h"
#include "llvm/Support/Debug.h"
#include <memory>
#include <mlir/IR/DialectRegistry.h>
#include <mlir/Transforms/Passes.h>

#define DEBUG_TYPE "mk-to-tx81"

using namespace mlir;

namespace mlir {
namespace triton {
#define GEN_PASS_DEF_MKTOTX81
#include "tsingmicro-tx81/Conversion/MKToTx81/Passes.h.inc"
} // namespace triton
} // namespace mlir

namespace {

class MKToTx81Pass : public triton::impl::MKToTx81Base<MKToTx81Pass> {
  using MKToTx81Base<MKToTx81Pass>::MKToTx81Base;

public:
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<linalg::LinalgDialect, memref::MemRefDialect,
                    arith::ArithDialect, mk::MagicKernelDialect,
                    tx::Tx81Dialect, LLVM::LLVMDialect>();
  }

  void runOnOperation() override {
    auto moduleOp = getOperation();

    RewritePatternSet canonicalizePatterns(&getContext());
    triton::populateMKToTx81CanonicalizationPatterns(canonicalizePatterns);
    if (failed(
            applyPatternsGreedily(moduleOp, std::move(canonicalizePatterns)))) {
      signalPassFailure();
    }

    // Use to memory::CopyOp to tx dialect op
    moduleOp->walk([&](Operation *op) {
      if (isa<memref::CopyOp>(op)) {
        auto copyOp = cast<memref::CopyOp>(op);
        op->setAttr("srcSpm",
                    IntegerAttr::get(
                        IntegerType::get(op->getContext(), 32),
                        llvm::APInt(32, triton::utils::isOperandMemorySpaceSPM(
                                            copyOp.getSource()))));
        op->setAttr("dstSpm",
                    IntegerAttr::get(
                        IntegerType::get(op->getContext(), 32),
                        llvm::APInt(32, triton::utils::isOperandMemorySpaceSPM(
                                            copyOp.getTarget()))));
      }
    });

    RewritePatternSet patterns(&getContext());
    ConversionTarget target(getContext());

    // Register illegal ops for Dialect Conversion
    target.addIllegalDialect<linalg::LinalgDialect,
                             bufferization::BufferizationDialect,
                             mk::MagicKernelDialect>();

    target.addLegalDialect<
        func::FuncDialect, arith::ArithDialect, math::MathDialect,
        affine::AffineDialect, scf::SCFDialect, memref::MemRefDialect,
        cf::ControlFlowDialect, tx::Tx81Dialect, LLVM::LLVMDialect>();

    target.addIllegalOp<memref::CopyOp>();

    target.addLegalOp<ModuleOp, linalg::YieldOp>();

    triton::populateMKToTx81ConversionPatterns(patterns);

    if (failed(applyPartialConversion(moduleOp, target, std::move(patterns)))) {
      signalPassFailure();
    }

    // linalg::linalgOpToLoops will generate memref::LoadOp/memref::StoreOp
    // before and after the arith calculation.
    // Use to check whether add spm mapping offset in
    // memref::LoadOp/memref::StoreOp lowering
    moduleOp->walk([&](Operation *op) {
      if (isa<memref::LoadOp, memref::StoreOp>(op)) {
        bool isSpm =
            isa<memref::LoadOp>(op)
                ? triton::utils::isOperandMemorySpaceSPM(op->getOperand(0))
                : triton::utils::isOperandMemorySpaceSPM(op->getOperand(1));

        op->setAttr("isSpm",
                    IntegerAttr::get(IntegerType::get(op->getContext(), 32),
                                     llvm::APInt(32, isSpm)));
      }
    });
  }
};

} // namespace

std::unique_ptr<OperationPass<ModuleOp>> triton::createMKToTx81Pass() {
  return std::make_unique<MKToTx81Pass>();
}
