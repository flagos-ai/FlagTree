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
static tle::DSLRegionOp findNearestNextDSLRegionOp(Operation *op) {
  Block *b = op->getBlock();
  auto it = std::next(op->getIterator());
  for (; it != b->end(); ++it) {
    if (auto dsl = dyn_cast<tle::DSLRegionOp>(&*it))
      return dsl;
  }
  return nullptr;
}

ttg::MemDescType getPlainMemDesc(RankedTensorType ty) {
  ttg::CTALayoutAttr ctaLayout = ttg::getCTALayout(ty.getEncoding());
  llvm::iota_range<uint32_t> rOrderRange =
      llvm::iota_range<uint32_t>(0, ty.getRank(), false);
  llvm::SmallVector<uint32_t> order = ttg::getOrder(ty);
  return ttg::MemDescType::get(ty.getShape(), ty.getElementType(),
                               ttg::SwizzledSharedEncodingAttr::get(
                                   ty.getContext(), 1, 1, 1, order, ctaLayout),
                               ttg::SharedMemorySpaceAttr::get(ty.getContext()),
                               true);
}

struct TleArgConversion : public OpRewritePattern<tle::ExtractAlignedPtrOp> {
  using OpRewritePattern::OpRewritePattern;

  TleArgConversion(MLIRContext *context);
  LogicalResult matchAndRewrite(tle::ExtractAlignedPtrOp op,
                                PatternRewriter &rewriter) const override;
};

struct TlePackConversion : public OpRewritePattern<tle::PackOp> {
  using OpRewritePattern::OpRewritePattern;

  TlePackConversion(MLIRContext *context);
  LogicalResult matchAndRewrite(tle::PackOp op,
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
TleArgConversion::matchAndRewrite(tle::ExtractAlignedPtrOp op,
                                  PatternRewriter &rewriter) const {
  Value input = op.getInput();
  auto tensorTy = dyn_cast<RankedTensorType>(input.getType());
  if (!tensorTy)
    return failure();

  PatternRewriter::InsertionGuard guard(rewriter);
  rewriter.setInsertionPoint(op);
  auto allocOp = rewriter.create<ttg::LocalAllocOp>(op.getLoc(),
                                                    getPlainMemDesc(tensorTy));
  rewriter.create<ttg::LocalStoreOp>(op.getLoc(), input, allocOp);
  Block *block = op->getBlock();
  SmallVector<Operation *> targets;

  for (Operation &it : block->getOperations()) {
    Operation *other = &it;
    if (other == op.getOperation())
      continue;

    bool usesInput =
        llvm::any_of(other->getOperands(), [&](Value v) { return v == input; });
    if (!usesInput)
      continue;

    if (isa<tle::ExtractSizesOp, tle::ExtractStridesOp, tle::ExtractOffsetOp,
            tle::ExtractAllocatedPtrOp>(other)) {
      targets.push_back(other);
    }
  }

  rewriter.setInsertionPoint(op);
  auto newAligned = rewriter.create<tle::ExtractAlignedPtrOp>(
      op.getLoc(), op.getResult().getType(), allocOp);
  Operation *lastNew = newAligned.getOperation();
  tle::DSLRegionOp dsl = findNearestNextDSLRegionOp(op.getOperation());
  for (Operation *other : targets) {
    rewriter.setInsertionPointAfter(lastNew);
    IRMapping mapper;
    if (auto ex = dyn_cast<tle::ExtractSizesOp>(other)) {
      auto newEx = rewriter.create<tle::ExtractSizesOp>(
          ex.getLoc(),
          ex->getResultTypes(),
          allocOp);
      rewriter.replaceOp(ex, newEx->getResults());
      continue;
    }
    if (auto ex = dyn_cast<tle::ExtractStridesOp>(other)) {
      auto newEx = rewriter.create<tle::ExtractStridesOp>(
          ex.getLoc(),
          ex->getResultTypes(),
          allocOp );
      rewriter.replaceOp(ex, newEx->getResults());
      continue;
    }
    if (auto ex = dyn_cast<tle::ExtractAllocatedPtrOp>(other)) {
      auto newEx = rewriter.create<tle::ExtractAllocatedPtrOp>(
          ex.getLoc(),
          ex->getResultTypes(),
          allocOp );
      rewriter.replaceOp(ex, newEx->getResults());
      continue;
    }
  }

  rewriter.replaceOp(op, newAligned.getResult());

  rewriter.setInsertionPointAfter(dsl);

  rewriter.create<ttg::LocalDeallocOp>(op.getLoc(), allocOp);

  return success();
}

TlePackConversion::TlePackConversion(MLIRContext *context)
    : OpRewritePattern(context) {}

LogicalResult
TlePackConversion::matchAndRewrite(tle::PackOp op,
                                   PatternRewriter &rewriter) const {
  auto tensorTy = dyn_cast<RankedTensorType>(op.getOutput().getType());
  if (!tensorTy)
    return failure();
  rewriter.setInsertionPoint(op);
  auto newPackOp = rewriter.create<tle::PackOp>(
      op.getLoc(), getPlainMemDesc(tensorTy), op.getInput());

  rewriter.setInsertionPointAfter(op);
  auto loadOp = rewriter.create<ttg::LocalLoadOp>(
      newPackOp.getLoc(), tensorTy, newPackOp.getOutput());
  
  rewriter.replaceOp(op, loadOp.getResult());
  return success();
}

void mlir::triton::tle::populateConvertArgToMemDescPatterns(
    RewritePatternSet &patterns) {
  patterns.add<TleArgConversion, TlePackConversion>(patterns.getContext());
}

void TleConvertArgToMemDesc::runOnOperation() {
  RewritePatternSet patterns(&getContext());
  tle::populateConvertArgToMemDescPatterns(patterns);
  if (failed(applyPatternsGreedily(getOperation(), std::move(patterns)))) {
    signalPassFailure();
  }
}
