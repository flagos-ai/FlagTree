#include "triton/Dialect/FlagTree/Transforms/DSLRegionInline.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "triton/Dialect/FlagTree/IR/Dialect.h"
#include "triton/Dialect/FlagTree/Transforms/Passes.h"
#include "triton/Dialect/TritonGPU/IR/Attributes.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/IR/Types.h"

namespace mlir::triton::flagtree {
#define GEN_PASS_DEF_FLAGTREEDSLREGIONINLINE
#include "triton/Dialect/FlagTree/Transforms/Passes.h.inc"
} // namespace mlir::triton::flagtree

using namespace mlir;
namespace fl = mlir::triton::flagtree;

namespace {
struct FlagTreeDSLRegionInlineConversion
    : public OpRewritePattern<fl::DSLRegionOp> {
  using OpRewritePattern::OpRewritePattern;

  FlagTreeDSLRegionInlineConversion(MLIRContext *context);
  LogicalResult matchAndRewrite(fl::DSLRegionOp op,
                                PatternRewriter &rewriter) const override;
};

struct FlagTreeDSLRegionInline
    : public fl::impl::FlagTreeDSLRegionInlineBase<FlagTreeDSLRegionInline> {
  void runOnOperation() override;
};

} // namespace

FlagTreeDSLRegionInlineConversion::FlagTreeDSLRegionInlineConversion(
    MLIRContext *context)
    : OpRewritePattern(context) {}

LogicalResult FlagTreeDSLRegionInlineConversion::matchAndRewrite(
    fl::DSLRegionOp op, PatternRewriter &rewriter) const {
  Operation *module = op;
  while (module && !isa<ModuleOp>(module)) {
    module = module->getParentOp();
  }
  IRMapping mapper;
  for (auto [arg, input] :
       llvm::zip(op.getBody().getArguments(), op.getInputs())) {
    mapper.map(arg, input);
  }
  for (Operation &operation : op.getOps()) {
    if (!isa<fl::YieldOp>(operation)) {
      rewriter.clone(operation, mapper);
    }
  }
  rewriter.eraseOp(op);
  return success();
}

void fl::populateDSLRegionInlinePatterns(RewritePatternSet &patterns) {
  patterns.add<FlagTreeDSLRegionInlineConversion>(patterns.getContext());
}

void FlagTreeDSLRegionInline::runOnOperation() {
  RewritePatternSet patterns(&getContext());
  fl::populateDSLRegionInlinePatterns(patterns);
  if (failed(applyPatternsGreedily(getOperation(), std::move(patterns)))) {
    signalPassFailure();
  }
}
