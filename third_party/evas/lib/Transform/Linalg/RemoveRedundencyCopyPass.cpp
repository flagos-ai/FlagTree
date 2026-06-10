//===----------------------------------------------------------------------===//
//
// Copyright (c) 2024 The EVAS Intelligence Inc. All Rights Reserved.
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//    http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//==============================================================================

#include "mlir/Analysis/AliasAnalysis.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/Dominance.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Interfaces/ViewLikeInterface.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#define GEN_PASS_DEF_REMOVEREDUNDENCYCOPY
#include "evas/Transform/Linalg/Passes.h.inc"

namespace mlir::triton::ev {

namespace {

// Pattern 1: Eliminate self-copy: copy(A, A) -> no-op
struct EliminateSelfCopyPattern : public OpRewritePattern<memref::CopyOp> {
  using OpRewritePattern<memref::CopyOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(memref::CopyOp copyOp,
                                PatternRewriter &rewriter) const override {
    Value source = copyOp.getSource();
    Value target = copyOp.getTarget();

    // Check if source and target are the same value
    if (source == target) {
      rewriter.eraseOp(copyOp);
      return success();
    }

    return failure();
  }
};

// Pattern 2: Chain copy elimination: copy(A, B) + copy(B, C) -> copy(A, C)
struct ChainCopyEliminationPattern : public OpRewritePattern<memref::CopyOp> {
  using OpRewritePattern<memref::CopyOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(memref::CopyOp copyOp,
                                PatternRewriter &rewriter) const override {
    Value source = copyOp.getSource();
    Value target = copyOp.getTarget();

    for (Operation *user : target.getUsers()) {
      if (user == copyOp)
        continue;

      auto nextCopy = dyn_cast<memref::CopyOp>(user);
      if (!nextCopy)
        continue;

      // Check if nextCopy uses target as source: copy(B, C)
      if (nextCopy.getSource() == target) {
        // Check dominance: copyOp must dominate nextCopy
        DominanceInfo domInfo(copyOp->getParentOfType<func::FuncOp>());
        if (!domInfo.dominates(copyOp.getOperation(), nextCopy.getOperation()))
          continue;

        // Check if target is only used by these two copy operations
        // (or we need to be more careful about aliasing)
        bool hasUserBetween = false;
        bool hasUserAfter = false;
        for (Operation *targetUser : target.getUsers()) {
          if (targetUser != copyOp && targetUser != nextCopy) {
            if (copyOp->isBeforeInBlock(targetUser) &&
                targetUser->isBeforeInBlock(nextCopy)) {
              hasUserBetween = true;
              break;
            } else {
              hasUserAfter = true;
            }
          }
        }
        if (hasUserBetween)
          continue;
        // Create new copy: copy(A, C)
        Value finalTarget = nextCopy.getTarget();
        rewriter.setInsertionPoint(nextCopy);
        rewriter.create<memref::CopyOp>(copyOp.getLoc(), source, finalTarget);
        // If the target is only used after the copy, and the types are the
        // same, we can replace the target with the final target
        if (hasUserAfter && target.getType() == finalTarget.getType()) {
          rewriter.replaceAllUsesWith(target, finalTarget);
        }
        rewriter.eraseOp(nextCopy);
        rewriter.eraseOp(copyOp);
        return success();
      }
    }

    return failure();
  }
};

// Pattern 3: Eliminate redundant copy: copy(A, B) followed by another copy(A,
// B)
struct EliminateRedundantCopyPattern : public OpRewritePattern<memref::CopyOp> {
  using OpRewritePattern<memref::CopyOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(memref::CopyOp copyOp,
                                PatternRewriter &rewriter) const override {
    Value source = copyOp.getSource();
    Value target = copyOp.getTarget();

    // Look for another copy operation with the same source and target
    for (Operation *user : source.getUsers()) {
      if (user == copyOp)
        continue;

      auto otherCopy = dyn_cast<memref::CopyOp>(user);
      if (!otherCopy)
        continue;

      // Check if it's the same copy: copy(A, B)
      if (otherCopy.getSource() == source && otherCopy.getTarget() == target) {
        // Check dominance: one must dominate the other
        DominanceInfo domInfo(copyOp->getParentOfType<func::FuncOp>());
        bool otherDominates =
            domInfo.dominates(otherCopy.getOperation(), copyOp.getOperation());
        bool thisDominates =
            domInfo.dominates(copyOp.getOperation(), otherCopy.getOperation());

        if (!otherDominates && !thisDominates)
          continue;

        Operation *firstCopy =
            otherDominates ? otherCopy.getOperation() : copyOp.getOperation();
        Operation *secondCopy =
            otherDominates ? copyOp.getOperation() : otherCopy.getOperation();

        // Simple check: if there are any writes to source or target between
        // the two copies, we can't eliminate
        bool hasWriteBetween = false;
        for (Operation &op :
             llvm::make_range(std::next(firstCopy->getIterator()),
                              secondCopy->getIterator())) {
          if (auto storeOp = dyn_cast<memref::StoreOp>(&op)) {
            if (storeOp.getMemRef() == source ||
                storeOp.getMemRef() == target) {
              hasWriteBetween = true;
              break;
            }
          }
          if (auto otherCopyOp = dyn_cast<memref::CopyOp>(&op)) {
            if (otherCopyOp.getTarget() == source ||
                otherCopyOp.getTarget() == target) {
              hasWriteBetween = true;
              break;
            }
          }
        }

        if (hasWriteBetween)
          continue;

        // Eliminate the second copy
        rewriter.eraseOp(secondCopy);
        return success();
      }
    }

    return failure();
  }
};

struct RemoveRedundencyCopyPass
    : public ::impl::RemoveRedundencyCopyBase<RemoveRedundencyCopyPass> {
  using RemoveRedundencyCopyBase<
      RemoveRedundencyCopyPass>::RemoveRedundencyCopyBase;

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<memref::MemRefDialect>();
    registry.insert<func::FuncDialect>();
  }

  void runOnOperation() override {
    ModuleOp moduleOp = getOperation();
    MLIRContext &context = getContext();

    RewritePatternSet patterns(&context);
    patterns.add<EliminateSelfCopyPattern>(&context);
    patterns.add<ChainCopyEliminationPattern>(&context);
    patterns.add<EliminateRedundantCopyPattern>(&context);

    // Apply patterns with multiple iterations to handle cascading optimizations
    if (failed(applyPatternsGreedily(moduleOp, std::move(patterns)))) {
      signalPassFailure();
    }
  }
};

} // namespace

std::unique_ptr<mlir::Pass> createRemoveRedundencyCopyPass() {
  return std::make_unique<RemoveRedundencyCopyPass>();
}

} // namespace mlir::triton::ev
