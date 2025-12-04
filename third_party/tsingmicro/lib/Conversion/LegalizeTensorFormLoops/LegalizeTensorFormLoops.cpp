//===----------------- LegalizeTensorFormLoops.cpp ------------------------===//
//
// Copyright (C) 2020-2025 Terapines Technology (Wuhan) Co., Ltd
// All rights reserved.
//
//===----------------------------------------------------------------------===//

#include "magic-kernel/Conversion/LegalizeTensorFormLoops/Passes.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#define DEBUG_TYPE "legalize-tensor-form-loops"

using namespace mlir;

namespace mlir {
namespace triton {
#define GEN_PASS_DEF_LEGALIZETENSORFORMLOOPS
#include "magic-kernel/Conversion/LegalizeTensorFormLoops/Passes.h.inc"
} // namespace triton
} // namespace mlir

namespace {
struct ForOpRewrite : public OpRewritePattern<scf::ForOp> {
  using OpRewritePattern<scf::ForOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(scf::ForOp forOp,
                                PatternRewriter &rewriter) const override {
    auto result = failure();
    auto yieldOp = cast<scf::YieldOp>(forOp.getBody()->getTerminator());
    rewriter.setInsertionPoint(yieldOp);
    for (auto op : llvm::enumerate(yieldOp->getOperands())) {
      auto val = op.value();
      auto itArg = forOp.getRegionIterArgs()[op.index()];
      if (!isa<TensorType>(val.getType()) || val == itArg)
        continue;
      auto copyOp = dyn_cast<linalg::CopyOp>(val.getDefiningOp());

      // TODO: Use BufferizableOpInterface to analyze whether the operand is
      // equivalent to the corresponding iter bbArg.
      if (!copyOp || copyOp.getOutputs()[0] != itArg) {
        auto reduceVal =
            rewriter.create<linalg::CopyOp>(forOp.getLoc(), val, itArg);
        yieldOp->setOperand(op.index(), reduceVal->getResult(0));
        result = success();
      }
    }

    return result;
  }
};

class LegalizeTensorFormLoopsPass
    : public triton::impl::LegalizeTensorFormLoopsBase<
          LegalizeTensorFormLoopsPass> {
  using LegalizeTensorFormLoopsBase<
      LegalizeTensorFormLoopsPass>::LegalizeTensorFormLoopsBase;

public:
  void runOnOperation() override {
    RewritePatternSet patterns(&getContext());
    patterns.add<ForOpRewrite>(&getContext());
    if (failed(applyPatternsGreedily(getOperation(), std::move(patterns)))) {
      signalPassFailure();
    }
  }
};

} // namespace
