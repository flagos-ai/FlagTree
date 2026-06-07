/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
 * THE SOFTWARE.
 */

#include "TritonToAnnotation/Passes.h"

#include "bishengir/Dialect/Annotation/IR/Annotation.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "Dialect/TritonAscend/IR/TritonAscendDialect.h"
namespace mlir {
namespace triton {
#define GEN_PASS_DEF_TRITONTOANNOTATION
#include "ascend/include/TritonToAnnotation/Passes.h.inc"
} // namespace triton
} // namespace mlir

using namespace mlir;

namespace {
struct TritonToAnnotationPass
    : public mlir::triton::impl::TritonToAnnotationBase<
          TritonToAnnotationPass> {
  void runOnOperation() override;
};
} // namespace

struct TritonAnnotationConversionPattern
    : OpRewritePattern<mlir::triton::ascend::AnnotationOp> {
  using OpRewritePattern<mlir::triton::ascend::AnnotationOp>::OpRewritePattern;

  LogicalResult matchAndRewrite(mlir::triton::ascend::AnnotationOp op,
                                PatternRewriter &rewriter) const final {
    auto markOp = rewriter.create<annotation::MarkOp>(op.getLoc(), op.getSrc());
    // Forward all annotations.
    markOp->setAttrs(op->getAttrs());
    rewriter.eraseOp(op);
    return success();
  }
};

void TritonToAnnotationPass::runOnOperation() {
  auto module = getOperation();
  ConversionTarget target(getContext());
  target.addLegalDialect<annotation::AnnotationDialect>();

  RewritePatternSet patterns(&getContext());
  patterns.add<TritonAnnotationConversionPattern>(patterns.getContext());
  if (failed(applyPartialConversion(module, target, std::move(patterns)))) {
    signalPassFailure();
  }
}

std::unique_ptr<OperationPass<ModuleOp>>
mlir::triton::createTritonToAnnotationPass() {
  return std::make_unique<TritonToAnnotationPass>();
}
