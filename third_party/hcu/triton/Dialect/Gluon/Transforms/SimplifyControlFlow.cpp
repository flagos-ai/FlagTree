// Copyright 2026 FlagOS Contributors
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.

#include "mlir/IR/OperationSupport.h"
#include "triton/Dialect/Gluon/Transforms/Passes.h"

#include "triton/Dialect/TritonGPU/Transforms/Utility.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlow.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

using namespace mlir;
using namespace triton;

namespace mlir::triton::gluon {
#define GEN_PASS_DEF_GLUONSIMPLIFYCONTROLFLOW
#include "triton/Dialect/Gluon/Transforms/Passes.h.inc"
} // namespace mlir::triton::gluon

namespace {
struct SimplifyControlFlow
    : public gluon::impl::GluonSimplifyControlFlowBase<SimplifyControlFlow> {
  void runOnOperation() override;
};
} // namespace

void SimplifyControlFlow::runOnOperation() {
  MLIRContext *ctx = &getContext();
  RewritePatternSet patterns(&getContext());

  // Populate `scf` and `cf` canonicalizers.
  ctx->getLoadedDialect<scf::SCFDialect>()->getCanonicalizationPatterns(
      patterns);
  ctx->getLoadedDialect<cf::ControlFlowDialect>()->getCanonicalizationPatterns(
      patterns);
  for (mlir::RegisteredOperationName op : ctx->getRegisteredOperationsByDialect(
           scf::SCFDialect::getDialectNamespace()))
    op.getCanonicalizationPatterns(patterns, ctx);
  for (mlir::RegisteredOperationName op : ctx->getRegisteredOperationsByDialect(
           cf::ControlFlowDialect::getDialectNamespace()))
    op.getCanonicalizationPatterns(patterns, ctx);
  populateForOpDeadArgumentElimination(patterns);

  GreedyRewriteConfig config;
  // This is intended to run before AutoLayouts are resolved, in which case
  // CSEing constants can lead to additional layout conflicts.
  config.enableConstantCSE(false);
  (void)applyPatternsGreedily(getOperation(), std::move(patterns), config);
}
