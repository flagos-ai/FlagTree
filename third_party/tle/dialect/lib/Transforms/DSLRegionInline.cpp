/*
* Copyright 2018-2020 Philippe Tillet
* Copyright 2020-2022 OpenAI
* Copyright 2025-     FlagOS Contributors
*
* Permission is hereby granted, free of charge, to any person obtaining
* a copy of this software and associated documentation files
* (the "Software"), to deal in the Software without restriction,
* including without limitation the rights to use, copy, modify, merge,
* publish, distribute, sublicense, and/or sell copies of the Software,
* and to permit persons to whom the Software is furnished to do so,
* subject to the following conditions:
*
* The above copyright notice and this permission notice shall be
* included in all copies or substantial portions of the Software.
*
* THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
* EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
* MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
* IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY
* CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,
* TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE
* SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
*/

#include "tle/dialect/include/Transforms/DSLRegionInline.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/Value.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "tle/dialect/include/IR/Dialect.h"
#include "tle/dialect/include/Transforms/Passes.h"
#include "triton/Dialect/TritonGPU/IR/Attributes.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/IR/Types.h"

namespace mlir::triton::tle {
#define GEN_PASS_DEF_TLEDSLREGIONINLINE
#include "tle/dialect/include/Transforms/Passes.h.inc"
} // namespace mlir::triton::tle

using namespace mlir;
namespace tle = mlir::triton::tle;

namespace {
struct TleDSLRegionInlineConversion
    : public OpRewritePattern<tle::DSLRegionOp> {
  using OpRewritePattern::OpRewritePattern;

  TleDSLRegionInlineConversion(MLIRContext *context);
  LogicalResult matchAndRewrite(tle::DSLRegionOp op,
                                PatternRewriter &rewriter) const override;
};

struct TleDSLRegionInline
    : public tle::impl::TleDSLRegionInlineBase<TleDSLRegionInline> {
  void runOnOperation() override;
};

} // namespace

TleDSLRegionInlineConversion::TleDSLRegionInlineConversion(MLIRContext *context)
    : OpRewritePattern(context) {}

LogicalResult
TleDSLRegionInlineConversion::matchAndRewrite(tle::DSLRegionOp op,
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
      if (auto yieldOp = dyn_cast<tle::YieldOp>(operation)) {
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

void tle::populateDSLRegionInlinePatterns(RewritePatternSet &patterns) {
  patterns.add<TleDSLRegionInlineConversion>(patterns.getContext());
}

void TleDSLRegionInline::runOnOperation() {
  RewritePatternSet patterns(&getContext());
  tle::populateDSLRegionInlinePatterns(patterns);
  if (failed(applyPatternsGreedily(getOperation(), std::move(patterns)))) {
    signalPassFailure();
  }
}
