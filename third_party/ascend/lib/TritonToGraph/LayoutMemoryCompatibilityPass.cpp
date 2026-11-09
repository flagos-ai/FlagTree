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

#include "TritonToGraph/GraphOptimization.h"
#include "TritonToGraph/LayoutMemoryOptimization.h"
#include "TritonToGraph/LegacyMemoryAccess/ChunkCoalescing.h"
#include "TritonToGraph/LegacyMemoryAccess/StridedAxisCoalescing.h"
#include "TritonToGraph/LegacyMemoryAccess/StridedLoadStoreRewrite.h"

#include "Dialect/TritonAscend/IR/TritonAscendDialect.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "llvm/Support/Debug.h"

#include <cstddef>
#include <memory>
#include <utility>

#define DEBUG_TYPE "graph-optimize"

namespace mlir {
namespace triton {
namespace cfg {
namespace {

static void emitAppliedGraphOptimizationRule(GraphOptimizationRuleId rule) {
  LLVM_DEBUG(llvm::dbgs() << "[" DEBUG_TYPE "] applied graph optimization rule "
                          << static_cast<unsigned>(rule) << " ("
                          << getGraphOptimizationRuleName(rule) << ")\n");
}

static std::size_t countStridedLoadStoreRewrites(ModuleOp moduleOp) {
  std::size_t count = 0;
  moduleOp.walk([&](Operation *op) {
    if (op->hasAttr(
            StridedLoadStoreRewrite::RewrittenByStridedLoadStoreRewriteTAG))
      ++count;
  });
  return count;
}

class LayoutMemoryCompatibilityPass final
    : public PassWrapper<LayoutMemoryCompatibilityPass,
                         OperationPass<ModuleOp>> {
public:
  explicit LayoutMemoryCompatibilityPass(LayoutMemoryCompatibilityPhase phase)
      : phase(phase) {}

  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(LayoutMemoryCompatibilityPass)

  void getDependentDialects(DialectRegistry &registry) const override {
    registry
        .insert<arith::ArithDialect, cf::ControlFlowDialect, math::MathDialect,
                scf::SCFDialect, tensor::TensorDialect, triton::TritonDialect,
                triton::ascend::TritonAscendDialect>();
  }

  void runOnOperation() override {
    ModuleOp moduleOp = getOperation();
    bool wasCoalesced = false;
    switch (phase) {
    case LayoutMemoryCompatibilityPhase::BeforeDiagonal:
      // Preserve the original module-wide, no-op-on-bailout semantics.  This
      // phase intentionally has no cleanup pass.
      wasCoalesced = moduleOp->hasAttr("hacc.coalesce_factor");
      StridedAxisCoalescing::rewriteStridedAxisCoalesce(moduleOp);
      if (!wasCoalesced && moduleOp->hasAttr("hacc.coalesce_factor"))
        emitAppliedGraphOptimizationRule(
            GraphOptimizationRuleId::StridedAxisCoalescing);
      return;
    case LayoutMemoryCompatibilityPhase::AfterDiagonal:
      // Keep the legacy order.  Chunk sees Axis's module attr and consumes the
      // static-grid hint before the greedy StridedLoadStore patterns run.
      wasCoalesced = moduleOp->hasAttr("hacc.coalesce_factor");
      ChunkCoalescing::rewriteChunkCoalesce(moduleOp);
      if (!wasCoalesced && moduleOp->hasAttr("hacc.coalesce_factor"))
        emitAppliedGraphOptimizationRule(
            GraphOptimizationRuleId::ChunkCoalescing);
      break;
    }

    // Count rewrite tags only while this DEBUG_TYPE is active. The tag lets us
    // emit one pass-level match line even when several load/store ops rewrite.
    bool emitStridedLoadStoreMatch = false;
    LLVM_DEBUG(emitStridedLoadStoreMatch = true;);
    const std::size_t stridedLoadStoreRewritesBefore =
        emitStridedLoadStoreMatch ? countStridedLoadStoreRewrites(moduleOp) : 0;
    RewritePatternSet patterns(&getContext());
    patterns.add<StridedLoadStoreRewrite::LoadConverter,
                 StridedLoadStoreRewrite::StoreConverter>(
        patterns.getContext());
    if (failed(applyPatternsGreedily(moduleOp, std::move(patterns)))) {
      signalPassFailure();
      return;
    }
    if (emitStridedLoadStoreMatch && countStridedLoadStoreRewrites(moduleOp) >
                                         stridedLoadStoreRewritesBefore)
      emitAppliedGraphOptimizationRule(
          GraphOptimizationRuleId::StridedLoadStoreRewrite);

    // CSE/Canonicalizer deliberately remain in TritonToLinalg at the original
    // call site.  PtrAnalysis may create helper IR even on an SLS no-op, and
    // the original T2L pipeline owns the unconditional cleanup timing.
  }

private:
  LayoutMemoryCompatibilityPhase phase;
};

} // namespace

std::unique_ptr<OperationPass<ModuleOp>>
createLayoutMemoryCompatibilityPass(LayoutMemoryCompatibilityPhase phase) {
  return std::make_unique<LayoutMemoryCompatibilityPass>(phase);
}

} // namespace cfg
} // namespace triton
} // namespace mlir
