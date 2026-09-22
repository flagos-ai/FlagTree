#ifdef __ILUVATAR_TLE__

#include "Transforms/DSLRegionInline.h"
#include "IR/Dialect.h"
#include "Transforms/Passes.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/Value.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "triton/Dialect/TritonGPU/IR/Attributes.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/IR/Types.h"

namespace mlir::triton::iluvatar_tle {
#define GEN_PASS_DEF_TRITONILUVATARTLEDSLREGIONINLINE
#include "Transforms/Passes.h.inc"
} // namespace mlir::triton::iluvatar_tle

using namespace mlir;
namespace iluvatar_tle = mlir::triton::iluvatar_tle;

namespace {
struct TleDSLRegionInlineConversion
    : public OpRewritePattern<iluvatar_tle::DSLRegionOp> {
  using OpRewritePattern::OpRewritePattern;

  TleDSLRegionInlineConversion(MLIRContext *context);
  LogicalResult matchAndRewrite(iluvatar_tle::DSLRegionOp op,
                                PatternRewriter &rewriter) const override;
};

struct TritonIluvatarTleDSLRegionInline
    : public iluvatar_tle::impl::TritonIluvatarTleDSLRegionInlineBase<
          TritonIluvatarTleDSLRegionInline> {
  void runOnOperation() override;
};

} // namespace

TleDSLRegionInlineConversion::TleDSLRegionInlineConversion(MLIRContext *context)
    : OpRewritePattern(context) {}

LogicalResult
TleDSLRegionInlineConversion::matchAndRewrite(iluvatar_tle::DSLRegionOp op,
                                              PatternRewriter &rewriter) const {
  IRMapping mapper;
  Block *parent = op->getBlock(),
        *continuation = rewriter.splitBlock(parent, op->getIterator());
  continuation->addArguments(
      op->getResultTypes(),
      SmallVector<Location>(op->getNumResults(), op.getLoc()));
  for (auto [oldResult, newArg] :
       llvm::zip(op.getResults(), continuation->getArguments())) {
    rewriter.replaceAllUsesWith(oldResult, newArg);
  }
  auto &blocks = op.getBody().getBlocks();
  const size_t blockNum = blocks.size();
  SmallVector<Block *> newBlocks;
  for (auto [idx, block] : llvm::enumerate(blocks)) {
    auto locs = llvm::map_range(
        block.getArguments(),
        [](BlockArgument &arg) -> Location { return arg.getLoc(); });
    Block *newBlock =
        rewriter.createBlock(continuation, block.getArgumentTypes(),
                             SmallVector<Location>(locs.begin(), locs.end()));
    for (auto [oldArg, newArg] :
         llvm::zip(block.getArguments(), newBlock->getArguments())) {
      mapper.map(oldArg, newArg);
    }
    if (idx == 0) {
      PatternRewriter::InsertionGuard guard(rewriter);
      rewriter.setInsertionPointToEnd(parent);
      rewriter.create<LLVM::BrOp>(op.getLoc(), op.getInputs(), newBlock);
    }
    mapper.map(&block, newBlock);
    newBlocks.push_back(newBlock);
  }
  for (auto [oldBlock, newBlock] : llvm::zip(blocks, newBlocks)) {
    PatternRewriter::InsertionGuard guard(rewriter);
    rewriter.setInsertionPointToEnd(newBlock);
    for (Operation &operation : oldBlock.getOperations()) {
      if (auto yieldOp = dyn_cast<iluvatar_tle::YieldOp>(operation)) {
        rewriter.create<LLVM::BrOp>(
            operation.getLoc(),
            llvm::map_to_vector(
                yieldOp.getOperands(),
                [&mapper](Value v) -> Value { return mapper.lookup(v); }),
            continuation);
      } else {
        rewriter.clone(operation, mapper);
      }
    }
  }
  rewriter.eraseOp(op);
  return success();
}

void mlir::triton::iluvatar_tle::populateDSLRegionInlinePatterns(
    RewritePatternSet &patterns) {
  patterns.add<TleDSLRegionInlineConversion>(patterns.getContext());
}

void TritonIluvatarTleDSLRegionInline::runOnOperation() {
  RewritePatternSet patterns(&getContext());
  iluvatar_tle::populateDSLRegionInlinePatterns(patterns);
  if (failed(applyPatternsGreedily(getOperation(), std::move(patterns)))) {
    signalPassFailure();
  }
}

#endif // __ILUVATAR_TLE__
