//===------------------- LinalgToMK.cpp -----------------------------------===//
//
// Copyright (C) 2020-2025 Terapines Technology (Wuhan) Co., Ltd
// All rights reserved.
//
//===----------------------------------------------------------------------===//

#include "magic-kernel/Conversion/LinalgToMK/LinalgToMK.h"
#include "magic-kernel/Dialect/IR/MagicKernelDialect.h"

#define DEBUG_TYPE "linalg-to-mk"

using namespace mlir;
using namespace mk;

#define GEN_PASS_CLASSES
#include "magic-kernel/Conversion/LinalgToMK/Passes.h.inc"

namespace {

// Convert tensor.empty + linalg.fill + linalg.matmul to mk.matmul
struct MatmulConverter : public OpConversionPattern<linalg::MatmulOp> {
private:
  using OpConversionPattern<linalg::MatmulOp>::OpConversionPattern;

public:
  LogicalResult
  matchAndRewrite(linalg::MatmulOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
#if 0
    auto tensorType = *op->getResultTypes().begin();

    Value output = op->getResult(0);
    auto fillOp = output.getDefiningOp();
    Value emptyTensor = fillOp->getResult(0);
    auto tensorEmptyOp = emptyTensor.getDefiningOp();

    auto dotOp = rewriter.create<DotOp>(loc, tensorType, op->getOperand(0),
        op->getOperand(1),
        op.getNumOperands() == 3 ? op->getOperand(2) : nullptr);
    rewriter.replaceOp(op, dotOp);
#endif
    return success();
  }
};

} // namespace

void mlir::triton::populateLinalgToMKCanonicalizationPatterns(
    RewritePatternSet &patterns) {}

void mlir::triton::populateLinalgToMKConversionPatterns(
    RewritePatternSet &patterns) {
  // patterns.add<MatmulConverter>(patterns.getContext());
  patterns.add<SigmoidFusionPattern>(patterns.getContext());
}
