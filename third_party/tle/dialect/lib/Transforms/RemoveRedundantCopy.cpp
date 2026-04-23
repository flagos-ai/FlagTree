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
      dslRegionOp = dyn_cast<tle::DSLRegionOp>(op);
    }
  }

  if (!dslRegionOp || !isSingleForLoop(forOp)) {
    return failure();
  }

  auto yieldOp = cast<scf::YieldOp>(forOp.getBody()->getTerminator());
  DenseMap<Value, unsigned> yieldValueToIndex;
  for (auto [idx, operand] : llvm::enumerate(yieldOp.getOperands())) {
    yieldValueToIndex[operand] = idx;
  }

  DenseMap<unsigned, Value> yieldIndexToDslRegionResult;
  for (auto result : dslRegionOp->getResults()) {
    for (Operation *user : result.getUsers()) {
      if (auto localLoadOp = dyn_cast<ttg::LocalLoadOp>(user)) {
        auto iter = yieldValueToIndex.find(localLoadOp.getResult());
        if (iter != yieldValueToIndex.end()) {
          yieldIndexToDslRegionResult[iter->second] = result;
        }
      }
    }
  }

  if (yieldIndexToDslRegionResult.empty()) {
    return failure();
  }

  PatternRewriter::InsertionGuard guard(rewriter);
  rewriter.setInsertionPoint(forOp);
  SmallVector<Value> newInitArgs;

  // rewrite initArgs
  for (auto [idx, arg] : llvm::enumerate(forOp.getInitArgs())) {
    if (yieldIndexToDslRegionResult.find(idx) !=
        yieldIndexToDslRegionResult.end()) {
      auto dslRegionResultIndex =
          (dyn_cast<OpResult>(yieldIndexToDslRegionResult[idx]))
              .getResultNumber();
      newInitArgs.push_back(dslRegionOp.getInputs()[dslRegionResultIndex]);
    } else {
      newInitArgs.push_back(arg);
    }
  }

  // rewrite forOp
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
    if (yieldIndexToDslRegionResult.find(idx) !=
        yieldIndexToDslRegionResult.end()) {
      auto dslRegionResultIndex =
          (dyn_cast<mlir::OpResult>(yieldIndexToDslRegionResult[idx]))
              .getResultNumber();
      arg.setType(dslRegionOp.getInputs()[dslRegionResultIndex].getType());
    }
  }

  // rewrite yieldOp
  if (auto newYieldOp = dyn_cast<scf::YieldOp>(newBlock.getTerminator())) {
    SmallVector<Value> newYieldOperands;
    for (auto [idx, operand] : llvm::enumerate(newYieldOp.getOperands())) {
      if (yieldIndexToDslRegionResult.find(idx) !=
          yieldIndexToDslRegionResult.end()) {
        auto dslRegionResultIndex =
            (dyn_cast<mlir::OpResult>(yieldIndexToDslRegionResult[idx]))
                .getResultNumber();
        newYieldOperands.push_back(dslRegionOp.getResult(dslRegionResultIndex));
      } else {
        newYieldOperands.push_back(operand);
      }
    }
    rewriter.setInsertionPoint(newYieldOp);
    rewriter.replaceOpWithNewOp<scf::YieldOp>(newYieldOp, newYieldOperands);
  }

  // replace forOp results
  rewriter.setInsertionPointAfter(newForOp);
  SmallVector<Value> results;
  for (auto [idx, newResult] : llvm::enumerate(newForOp.getResults())) {
    if (yieldIndexToDslRegionResult.find(idx) !=
        yieldIndexToDslRegionResult.end()) {
      auto oldResult = forOp.getResult(idx);
      auto localLoadOp = rewriter.create<ttg::LocalLoadOp>(
          forOp.getLoc(), oldResult.getType(), newResult);
      results.push_back(localLoadOp);
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
