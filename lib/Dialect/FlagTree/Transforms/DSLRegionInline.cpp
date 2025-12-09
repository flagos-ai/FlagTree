#include "triton/Dialect/FlagTree/Transforms/DSLRegionInline.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
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
  IRMapping mapper;
  Block *parent = op->getBlock(),
        *continuation = rewriter.splitBlock(parent, op->getIterator());
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
      if (isa<fl::YieldOp>(operation)) {
        rewriter.create<LLVM::BrOp>(operation.getLoc(), continuation);
      } else {
        rewriter.clone(operation, mapper);
      }
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
