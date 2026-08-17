#include "triton/Dialect/Triton/IR/Dialect.h"

#include "proton/Dialect/include/Dialect/Proton/IR/Dialect.h"
#include "triton-shared/Conversion/XSMTToLinalg/XSMTToLinalg.h"
#include "triton-shared/Dialect/TritonStructured/IR/TritonStructuredDialect.h"
#include "triton-shared/Dialect/TritonTilingExt/IR/TritonTilingExtDialect.h"
#include "triton-shared/Dialect/XSMT/IR/XSMTDialect.h"
#include "triton-shared/Dialect/XSMT/IR/XSMTOps.h"
#include "triton-shared/Dialect/XSMTAsync/IR/XSMTAsyncDialect.h"
#include "triton-shared/Dialect/XSMTAsync/IR/XSMTAsyncOps.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/Support/LogicalResult.h"

#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/SCF/Transforms/Patterns.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "triton/Dialect/Triton/IR/Types.h"

#define DEBUG_TYPE "xsmt-to-linalg"

using namespace mlir;
using namespace triton;

namespace mlir {
namespace triton {
#define GEN_PASS_DEF_XSMTTOLINALG
#include "triton-shared/Conversion/XSMTToLinalg/Passes.h.inc"

// Defined in XSMTToLinalg.cpp — not in public header since it's only
// used internally by this pass as a post-conversion cleanup stage.
void populateXSMTBufferizationCleanupPatterns(RewritePatternSet &patterns);

} // namespace triton
} // namespace mlir

namespace {

template <typename PopulateFn>
LogicalResult applyPatternStage(Operation *op, MLIRContext *ctx,
                                PopulateFn &&populatePatterns) {
  RewritePatternSet patterns(ctx);
  populatePatterns(patterns);
  return applyPatternsGreedily(op, std::move(patterns));
}

/// Build a ConversionTarget that marks all xsmt ops as illegal and
/// standard dialects as legal, then apply partial conversion.
LogicalResult applyXSMTConversionStage(Operation *op, MLIRContext *ctx) {
  ConversionTarget target(*ctx);

  // XSMT + Proton ops that must be eliminated
  target.addIllegalOp<xsmt::PackOp, xsmt::UnpackOp, xsmt::RepackOp,
                      xsmt::DescriptorLoadViewOp, xsmt::MMT4DOp,
                      xsmt::GetThreadOp, proton::RecordOp>();

  // All target dialects are legal
  target.addLegalDialect<
      func::FuncDialect, arith::ArithDialect, math::MathDialect,
      linalg::LinalgDialect, affine::AffineDialect, scf::SCFDialect,
      tensor::TensorDialect, bufferization::BufferizationDialect,
      memref::MemRefDialect, LLVM::LLVMDialect, triton::TritonDialect,
      ttx::TritonTilingExtDialect, tts::TritonStructuredDialect>();

  // scf::ForOp with bind_sub_block attr → scf::ForallOp
  // (must come after addLegalDialect<scf::SCFDialect> to override)
  target.addDynamicallyLegalOp<scf::ForOp>([](scf::ForOp op) {
    auto attr = op->getAttrOfType<BoolAttr>("bind_sub_block");
    return !attr || !attr.getValue();
  });

  // xsmt/xsmt_async ops not in the illegal list are left as-is (legal by
  // default for unknown ops). Mark the dialects as dynamically legal so that
  // ops we didn't explicitly mark illegal are allowed through.
  target.addLegalDialect<xsmt::XSMTDialect, xsmt_async::XSMTAsyncDialect>();

  // Re-mark the specific ops as illegal (overrides the dialect-level legal)
  target.addIllegalOp<xsmt::PackOp, xsmt::UnpackOp, xsmt::RepackOp,
                      xsmt::DescriptorLoadViewOp, xsmt::MMT4DOp,
                      xsmt::GetThreadOp, proton::RecordOp>();

  RewritePatternSet patterns(ctx);
  triton::populateXSMTConversionPatterns(patterns);
  return applyPartialConversion(op, target, std::move(patterns));
}

class XSMTToLinalgPass
    : public triton::impl::XSMTToLinalgBase<XSMTToLinalgPass> {
  using XSMTToLinalgBase<XSMTToLinalgPass>::XSMTToLinalgBase;

public:
  void getDependentDialects(DialectRegistry &registry) const override {
    registry
        .insert<func::FuncDialect, arith::ArithDialect, math::MathDialect,
                linalg::LinalgDialect, affine::AffineDialect, scf::SCFDialect,
                tensor::TensorDialect, bufferization::BufferizationDialect,
                triton::TritonDialect, ttx::TritonTilingExtDialect,
                tts::TritonStructuredDialect, mlir::LLVM::LLVMDialect,
                mlir::xsmt::XSMTDialect, mlir::xsmt_async::XSMTAsyncDialect>();
  }

  void runOnOperation() override {
    auto moduleOp = getOperation();
    auto *ctx = &getContext();

    bool hasXSMTUser = false;
    moduleOp.walk([&](Operation *op) {
      if (op->getName().getStringRef().starts_with("xsmt.") ||
          op->getName().getStringRef().starts_with("xsmt_async.")) {
        hasXSMTUser = true;
        return WalkResult::interrupt();
      }
      return WalkResult::advance();
    });

    if (!hasXSMTUser) {
      // No XSMT ops: still run conversion for ForToForall + Proton lowering
      if (failed(applyXSMTConversionStage(moduleOp, ctx))) {
        signalPassFailure();
      }
      return;
    }

    // Phase 1: Validation + MBarrier checks + MBarrier release insertion
    if (failed(applyPatternStage(moduleOp, ctx,
                                 triton::populateXSMTValidationPatterns))) {
      signalPassFailure();
      return;
    }

    // Phase 2: XSMT Dialect Conversion + Loop parallelization + Proton lowering
    if (failed(applyXSMTConversionStage(moduleOp, ctx))) {
      signalPassFailure();
      return;
    }

    // Phase 3: Bufferization cleanup — rewrite unpack+materialize chains
    // into direct subview writes (requires ops generated in Phase 2).
    if (failed(applyPatternStage(
            moduleOp, ctx, triton::populateXSMTBufferizationCleanupPatterns))) {
      signalPassFailure();
      return;
    }
  }
};

} // namespace

std::unique_ptr<OperationPass<ModuleOp>> triton::createXSMTToLinalgPass() {
  return std::make_unique<XSMTToLinalgPass>();
}
