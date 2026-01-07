#include "tle/dialect/include/Transforms/ConvertArgToMemDesc.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Dominance.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/Value.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "tle/dialect/include/IR/Dialect.h"
#include "tle/dialect/include/Transforms/Passes.h"
#include "triton/Dialect/TritonGPU/IR/Attributes.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/IR/Types.h"
#include "triton/Dialect/TritonGPU/Transforms/PipeliningUtility.h"
#include "llvm/ADT/iterator_range.h"
#include "llvm/Support/Casting.h"

namespace mlir::triton::tle {
#define GEN_PASS_DEF_TLECONVERTARGTOMEMDESC
#include "tle/dialect/include/Transforms/Passes.h.inc"
} // namespace mlir::triton::tle

using namespace mlir;
namespace ttg = mlir::triton::gpu;
namespace tle = mlir::triton::tle;

namespace {

ttg::MemDescType getPlainMemDesc(RankedTensorType ty) {
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

struct TleArgConversion : public OpRewritePattern<tle::DSLRegionOp> {
  using OpRewritePattern::OpRewritePattern;

  TleArgConversion(MLIRContext *context);
  LogicalResult matchAndRewrite(tle::DSLRegionOp op,
                                PatternRewriter &rewriter) const override;
};

struct TleConvertArgToMemDesc
    : public tle::impl::TleConvertArgToMemDescBase<TleConvertArgToMemDesc> {
  void runOnOperation() override;
};

} // namespace

TleArgConversion::TleArgConversion(MLIRContext *context)
    : OpRewritePattern(context) {}

LogicalResult
TleArgConversion::matchAndRewrite(tle::DSLRegionOp op,
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
  tle::DSLRegionOp newOp =
      rewriter.create<tle::DSLRegionOp>(op.getLoc(), newRetTys, newOperands);
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
      if (tle::PackOp packOp = dyn_cast<tle::PackOp>(operation)) {
        if (auto tensorTy =
                dyn_cast<RankedTensorType>(packOp.getOutput().getType())) {
          tle::PackOp newPackOp = rewriter.create<tle::PackOp>(
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

void mlir::triton::tle::populateConvertArgToMemDescPatterns(
    RewritePatternSet &patterns) {
  patterns.add<TleArgConversion>(patterns.getContext());
}

void TleConvertArgToMemDesc::runOnOperation() {
  RewritePatternSet patterns(&getContext());
  tle::populateConvertArgToMemDescPatterns(patterns);
  if (failed(applyPatternsGreedily(getOperation(), std::move(patterns)))) {
    signalPassFailure();
  }
}
