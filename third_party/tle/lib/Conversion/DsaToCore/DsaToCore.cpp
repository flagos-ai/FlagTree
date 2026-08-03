//===- DsaToCore.cpp - Lower dsa memory ops to core dialects ----*- C++ -*-===//
//
// Lower dsa.alloc/copy into standard memref ops
// ops before one-shot-bufferize.
//
//===----------------------------------------------------------------------===//

#include "tle-dsa/Conversion/DsaToCore/DsaToCore.h"

#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "tle-dsa/Dialect/IR/DsaDialect.h"

using namespace mlir;

namespace {

struct DsaAllocToMemRefPattern : public OpRewritePattern<mlir::dsa::AllocOp> {
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(mlir::dsa::AllocOp op,
                                PatternRewriter &rewriter) const override {
    auto memrefTy = dyn_cast<MemRefType>(op.getResult().getType());
    if (!memrefTy)
      return failure();
    // tx81-memref-to-llvm expects integer/default memref address spaces.
    // Canonicalize away non-integer memory-space attrs (e.g. "local").
    if (Attribute ms = memrefTy.getMemorySpace(); ms && !isa<IntegerAttr>(ms)) {
      memrefTy = MemRefType::get(memrefTy.getShape(), memrefTy.getElementType(),
                                 memrefTy.getLayout());
    }
    rewriter.replaceOpWithNewOp<memref::AllocOp>(op, memrefTy);
    return success();
  }
};

struct DsaCopyToMemRefPattern : public OpRewritePattern<mlir::dsa::CopyOp> {
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(mlir::dsa::CopyOp op,
                                PatternRewriter &rewriter) const override {
    rewriter.create<memref::CopyOp>(op.getLoc(), op.getSrc(), op.getDst());
    rewriter.eraseOp(op);
    return success();
  }
};

struct DsaMemoryToCorePass
    : public PassWrapper<DsaMemoryToCorePass, OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(DsaMemoryToCorePass)
  StringRef getArgument() const final { return "dsa-memory-to-core"; }
  StringRef getDescription() const final {
    return "Lower dsa.alloc/copy to memref";
  }
  void runOnOperation() override {
    RewritePatternSet patterns(&getContext());
    patterns.add<DsaAllocToMemRefPattern, DsaCopyToMemRefPattern>(
        &getContext());
    if (failed(applyPatternsGreedily(getOperation(), std::move(patterns))))
      signalPassFailure();
  }
};

} // namespace

namespace mlir::dsa {
std::unique_ptr<Pass> createDsaMemoryToCorePass() {
  return std::make_unique<DsaMemoryToCorePass>();
}

void registerDsaMemoryToCorePass() { PassRegistration<DsaMemoryToCorePass>(); }
} // namespace mlir::dsa
