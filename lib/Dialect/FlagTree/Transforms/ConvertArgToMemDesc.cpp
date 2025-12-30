#include "triton/Dialect/FlagTree/Transforms/ConvertArgToMemDesc.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Dominance.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/Value.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "triton/Dialect/FlagTree/IR/Dialect.h"
#include "triton/Dialect/FlagTree/Transforms/Passes.h"
#include "triton/Dialect/TritonGPU/IR/Attributes.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/IR/Types.h"
#include "triton/Dialect/TritonGPU/Transforms/PipeliningUtility.h"
#include "llvm/ADT/iterator_range.h"
#include "llvm/Support/Casting.h"

namespace mlir::triton::flagtree {
#define GEN_PASS_DEF_FLAGTREECONVERTARGTOMEMDESC
#include "triton/Dialect/FlagTree/Transforms/Passes.h.inc"
} // namespace mlir::triton::flagtree

using namespace mlir;
namespace ttg = mlir::triton::gpu;
namespace fl = mlir::triton::flagtree;

namespace {

ttg::MemDescType getPlainMemDesc(RankedTensorType ty) {
  // TODO(EDSL): currently we set the order as a descending sequence to
  // match row-major layout, in the future we plan to align the order with the
  // original tensor layout, and update `ExtractStridesOpConversion` based on
  // the corresponding input shared memory layout to get correct strides
  ttg::CTALayoutAttr ctaLayout = ttg::getCTALayout(ty.getEncoding());
  llvm::iota_range<uint32_t> rOrderRange =
      llvm::iota_range<uint32_t>(0, ty.getRank(), false);
  llvm::SmallVector<uint32_t> order(rOrderRange.rbegin(), rOrderRange.rend());
  return ttg::MemDescType::get(ty.getShape(), ty.getElementType(),
                               ttg::SwizzledSharedEncodingAttr::get(
                                   ty.getContext(), 1, 1, 1, order, ctaLayout),
                               ttg::SharedMemorySpaceAttr::get(ty.getContext()),
                               true);
}

struct FlagTreeArgConversion : public OpRewritePattern<fl::DSLRegionOp> {
  using OpRewritePattern::OpRewritePattern;

  FlagTreeArgConversion(MLIRContext *context);
  LogicalResult matchAndRewrite(fl::DSLRegionOp op,
                                PatternRewriter &rewriter) const override;
};

struct FlagTreeConvertArgToMemDesc
    : public fl::impl::FlagTreeConvertArgToMemDescBase<
          FlagTreeConvertArgToMemDesc> {
  void runOnOperation() override;
};

} // namespace

FlagTreeArgConversion::FlagTreeArgConversion(MLIRContext *context)
    : OpRewritePattern(context) {}

LogicalResult
FlagTreeArgConversion::matchAndRewrite(fl::DSLRegionOp op,
                                       PatternRewriter &rewriter) const {
  SmallVector<Value> newOperands;
  IRMapping mapper;
  bool hasConversion = false;
  for (const auto &operand : op->getOperands()) {
    if (RankedTensorType tensorTy =
            dyn_cast<RankedTensorType>(operand.getType())) {
      Operation *defOp = operand.getDefiningOp();
      PatternRewriter::InsertionGuard guard(rewriter);
      rewriter.setInsertionPoint(op);
      ttg::LocalAllocOp allocOp = rewriter.create<ttg::LocalAllocOp>(
          op->getLoc(), getPlainMemDesc(tensorTy));
      rewriter.create<ttg::LocalStoreOp>(op->getLoc(), operand, allocOp);
      rewriter.setInsertionPointAfter(op);
      rewriter.create<ttg::LocalDeallocOp>(op->getLoc(), allocOp);
      newOperands.push_back(allocOp);
      mapper.map(operand, allocOp);
      hasConversion = true;
    } else {
      newOperands.push_back(operand);
    }
  }
  SmallVector<Type> newRetTys;
  for (auto result : op.getResults()) {
    if (RankedTensorType tensorTy =
            dyn_cast<RankedTensorType>(result.getType())) {
      newRetTys.push_back(getPlainMemDesc(tensorTy));
      hasConversion = true;
    } else {
      newRetTys.push_back(result.getType());
    }
  }
  if (!hasConversion) {
    return failure();
  }
  fl::DSLRegionOp newOp =
      rewriter.create<fl::DSLRegionOp>(op.getLoc(), newRetTys, newOperands);
  PatternRewriter::InsertionGuard guard(rewriter);
  for (auto [idx, oldBlock] : llvm::enumerate(op.getBody().getBlocks())) {
    Block *newBlock;
    if (idx == 0) {
      newBlock = rewriter.createBlock(
          &newOp.getBody(), {}, newOp->getOperandTypes(),
          SmallVector<Location>(newOp->getNumOperands(), op.getLoc()));
    } else {
      newBlock = rewriter.createBlock(
          &newOp.getBody(), {}, oldBlock.getArgumentTypes(),
          SmallVector<Location>(oldBlock.getNumArguments(), op.getLoc()));
    }
    for (auto [oldArg, newArg] :
         llvm::zip(oldBlock.getArguments(), newBlock->getArguments())) {
      mapper.map(oldArg, newArg);
    }
    mapper.map(&oldBlock, newBlock);
  }
  for (auto [oldBlock, newBlock] :
       llvm::zip(op.getBody().getBlocks(), newOp.getBody().getBlocks())) {
    rewriter.setInsertionPointToEnd(&newBlock);
    for (Operation &operation : oldBlock.getOperations()) {
      bool hasReplaced = false;
      if (fl::PackOp packOp = dyn_cast<fl::PackOp>(operation)) {
        if (auto tensorTy =
                dyn_cast<RankedTensorType>(packOp.getOutput().getType())) {
          fl::PackOp newPackOp = rewriter.create<fl::PackOp>(
              packOp.getLoc(), getPlainMemDesc(tensorTy),
              mapper.lookup(packOp.getInput()));
          mapper.map(packOp.getOutput(), newPackOp.getOutput());
          hasReplaced = true;
        } else {
          rewriter.clone(operation, mapper);
        }
      } else {
        rewriter.clone(operation, mapper);
      }
    }
  }
  rewriter.setInsertionPointAfter(newOp);
  SmallVector<Value> results;
  for (auto [oldResult, newResult] :
       llvm::zip(op.getResults(), newOp.getResults())) {
    if (RankedTensorType tensorTy =
            dyn_cast<RankedTensorType>(oldResult.getType())) {
      ttg::LocalLoadOp loadOp =
          rewriter.create<ttg::LocalLoadOp>(op.getLoc(), tensorTy, newResult);
      results.push_back(loadOp);
    } else {
      results.push_back(newResult);
    }
  }
  rewriter.replaceOp(op, results);
  return success();
}

void mlir::triton::flagtree::populateConvertArgToMemDescPatterns(
    RewritePatternSet &patterns) {
  patterns.add<FlagTreeArgConversion>(patterns.getContext());
}

void FlagTreeConvertArgToMemDesc::runOnOperation() {
  RewritePatternSet patterns(&getContext());
  fl::populateConvertArgToMemDescPatterns(patterns);
  if (failed(applyPatternsGreedily(getOperation(), std::move(patterns)))) {
    signalPassFailure();
  }
}
