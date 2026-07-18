#include "tle/dialect/include/Transforms/RemoveRedundantCopy.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Dominance.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/Value.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "tle/dialect/include/IR/Dialect.h"
#include "tle/dialect/include/Transforms/Passes.h"
#include "tle/dialect/include/Transforms/TleUtility.h"
#include "triton/Dialect/TritonGPU/IR/Attributes.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/IR/Types.h"
#include "triton/Dialect/TritonGPU/Transforms/PipeliningUtility.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/iterator_range.h"
#include "llvm/Support/Casting.h"

namespace mlir::triton::tle {
#define GEN_PASS_DEF_TLEREMOVEREDUNDANTCOPY
#include "tle/dialect/include/Transforms/Passes.h.inc"
} // namespace mlir::triton::tle

using namespace mlir;
namespace ttg = mlir::triton::gpu;
namespace tle = mlir::triton::tle;

namespace {

struct OutputRewrite {
  unsigned yieldIdx;
  unsigned resultNum;
  unsigned operandIdx;
  ttg::LocalAllocOp innerAlloc;
  ttg::LocalStoreOp innerStore;
  ttg::LocalLoadOp innerLoad;
  ttg::LocalDeallocOp innerDealloc;
  ttg::LocalAllocOp hoistedAlloc;
};

struct ForOpArgConversion : public OpRewritePattern<scf::ForOp> {
  using OpRewritePattern::OpRewritePattern;

  ForOpArgConversion(MLIRContext *context);
  LogicalResult matchAndRewrite(scf::ForOp forOp,
                                PatternRewriter &rewriter) const override;
};

struct TleRemoveRedundantCopy
    : public tle::impl::TleRemoveRedundantCopyBase<TleRemoveRedundantCopy> {
  void runOnOperation() override;
};

} // namespace

ForOpArgConversion::ForOpArgConversion(MLIRContext *context)
    : OpRewritePattern(context) {}

LogicalResult
ForOpArgConversion::matchAndRewrite(scf::ForOp forOp,
                                    PatternRewriter &rewriter) const {
  tle::DSLRegionOp dslRegionOp;
  for (auto &op : forOp.getBody()->without_terminator()) {
    if (isa<tle::DSLRegionOp>(op)) {
      if (dslRegionOp)
        return failure();
      dslRegionOp = dyn_cast<tle::DSLRegionOp>(op);
    }
  }

  if (!dslRegionOp || !isSingleForLoop(forOp))
    return failure();

  auto outputIndices = dslRegionOp.getOutputOperandIndices();
  if (outputIndices.empty())
    return failure();

  auto yieldOp = cast<scf::YieldOp>(forOp.getBody()->getTerminator());
  DenseMap<Value, unsigned> yieldValueToIndex;
  for (auto [idx, operand] : llvm::enumerate(yieldOp.getOperands())) {
    yieldValueToIndex[operand] = idx;
  }

  SmallVector<OutputRewrite> rewrites;
  DenseMap<unsigned, unsigned> yieldIndexToRewrite;
  for (auto [resultNum, result] : llvm::enumerate(dslRegionOp.getResults())) {
    if (resultNum >= outputIndices.size())
      return failure();
    int64_t operandIdxSigned = outputIndices[resultNum];
    if (operandIdxSigned < 0 ||
        operandIdxSigned >= static_cast<int64_t>(dslRegionOp->getNumOperands()))
      return failure();
    unsigned operandIdx = static_cast<unsigned>(operandIdxSigned);

    ttg::LocalLoadOp yieldedLoad;
    for (Operation *user : result.getUsers()) {
      if (auto localLoadOp = dyn_cast<ttg::LocalLoadOp>(user)) {
        auto iter = yieldValueToIndex.find(localLoadOp.getResult());
        if (iter != yieldValueToIndex.end()) {
          if (yieldedLoad || yieldIndexToRewrite.count(iter->second))
            return failure();
          yieldedLoad = localLoadOp;
          yieldIndexToRewrite[iter->second] = rewrites.size();
        }
      }
    }
    if (!yieldedLoad)
      continue;

    Value operand = dslRegionOp.getOperand(operandIdx);
    auto innerAlloc = operand.getDefiningOp<ttg::LocalAllocOp>();
    if (!innerAlloc)
      return failure();

    ttg::LocalStoreOp innerStore;
    ttg::LocalDeallocOp innerDealloc;
    for (Operation *user : llvm::make_early_inc_range(operand.getUsers())) {
      if (auto store = dyn_cast<ttg::LocalStoreOp>(user)) {
        if (store.getDst() == operand) {
          if (innerStore)
            return failure();
          innerStore = store;
        }
      } else if (auto dealloc = dyn_cast<ttg::LocalDeallocOp>(user)) {
        if (dealloc.getSrc() == operand) {
          if (innerDealloc)
            return failure();
          innerDealloc = dealloc;
        }
      }
    }
    if (!innerStore || !innerDealloc)
      return failure();

    auto yieldIdx = yieldValueToIndex[yieldedLoad.getResult()];
    if (yieldIdx >= forOp.getNumResults() ||
        innerStore.getSrc() != forOp.getRegionIterArgs()[yieldIdx])
      return failure();

    rewrites.push_back({yieldIdx, static_cast<unsigned>(resultNum), operandIdx,
                        innerAlloc, innerStore, yieldedLoad, innerDealloc,
                        nullptr});
  }

  if (rewrites.empty())
    return failure();

  PatternRewriter::InsertionGuard guard(rewriter);
  rewriter.setInsertionPoint(forOp);

  DenseMap<unsigned, unsigned> yieldIndexToRewriteAfterHoist;
  for (auto [idx, rewrite] : llvm::enumerate(rewrites)) {
    auto memDescTy =
        dyn_cast<ttg::MemDescType>(rewrite.innerAlloc.getResult().getType());
    if (!memDescTy)
      return failure();
    rewrite.hoistedAlloc =
        rewriter.create<ttg::LocalAllocOp>(forOp.getLoc(), memDescTy);
    rewriter.create<ttg::LocalStoreOp>(forOp.getLoc(),
                                       forOp.getInitArgs()[rewrite.yieldIdx],
                                       rewrite.hoistedAlloc);
    yieldIndexToRewriteAfterHoist[rewrite.yieldIdx] = idx;
  }

  SmallVector<Value> newInitArgs(forOp.getInitArgs());
  for (auto &rewrite : rewrites)
    newInitArgs[rewrite.yieldIdx] = rewrite.hoistedAlloc;

  auto newForOp = rewriter.create<scf::ForOp>(
      forOp.getLoc(), forOp.getLowerBound(), forOp.getUpperBound(),
      forOp.getStep(), newInitArgs);
  Region &oldRegion = forOp.getRegion();
  Region &newRegion = newForOp.getRegion();
  if (!newRegion.empty()) {
    rewriter.eraseBlock(&newRegion.front());
  }
  rewriter.inlineRegionBefore(oldRegion, newRegion, newRegion.end());

  Block &newBlock = newRegion.front();
  for (auto &arg : newBlock.getArguments()) {
    if (arg.getArgNumber() < forOp.getNumInductionVars()) {
      continue;
    }

    auto idx = arg.getArgNumber() - forOp.getNumInductionVars();
    auto iter = yieldIndexToRewriteAfterHoist.find(idx);
    if (iter == yieldIndexToRewriteAfterHoist.end())
      continue;
    auto &rewrite = rewrites[iter->second];
    arg.setType(rewrite.hoistedAlloc.getResult().getType());
    dslRegionOp.setOperand(rewrite.operandIdx, arg);
  }

  if (auto newYieldOp = dyn_cast<scf::YieldOp>(newBlock.getTerminator())) {
    SmallVector<Value> newYieldOperands;
    for (auto [idx, operand] : llvm::enumerate(newYieldOp.getOperands())) {
      auto iter = yieldIndexToRewriteAfterHoist.find(idx);
      if (iter != yieldIndexToRewriteAfterHoist.end()) {
        newYieldOperands.push_back(
            dslRegionOp.getResult(rewrites[iter->second].resultNum));
      } else {
        newYieldOperands.push_back(operand);
      }
    }
    rewriter.setInsertionPoint(newYieldOp);
    rewriter.replaceOpWithNewOp<scf::YieldOp>(newYieldOp, newYieldOperands);
  }

  for (auto &rewrite : rewrites) {
    if (rewrite.innerLoad->use_empty())
      rewriter.eraseOp(rewrite.innerLoad);
    rewriter.eraseOp(rewrite.innerDealloc);
    rewriter.eraseOp(rewrite.innerStore);
    if (rewrite.innerAlloc->use_empty())
      rewriter.eraseOp(rewrite.innerAlloc);
  }

  rewriter.setInsertionPointAfter(newForOp);
  SmallVector<Value> results;
  for (auto [idx, newResult] : llvm::enumerate(newForOp.getResults())) {
    auto iter = yieldIndexToRewriteAfterHoist.find(idx);
    if (iter != yieldIndexToRewriteAfterHoist.end()) {
      auto oldResult = forOp.getResult(idx);
      auto localLoadOp = rewriter.create<ttg::LocalLoadOp>(
          forOp.getLoc(), oldResult.getType(), newResult);
      results.push_back(localLoadOp);
      rewriter.create<ttg::LocalDeallocOp>(forOp.getLoc(), newResult);
    } else {
      results.push_back(newResult);
    }
  }

  rewriter.replaceOp(forOp, results);
  return success();
}

void mlir::triton::tle::populateRemoveRedundantCopyPatterns(
    RewritePatternSet &patterns) {
  patterns.add<ForOpArgConversion>(patterns.getContext());
}

void TleRemoveRedundantCopy::runOnOperation() {
  RewritePatternSet patterns(&getContext());
  tle::populateRemoveRedundantCopyPatterns(patterns);
  if (failed(applyPatternsGreedily(getOperation(), std::move(patterns)))) {
    signalPassFailure();
  }
}
