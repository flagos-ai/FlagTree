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

#include "llvm/Support/Debug.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#include "ascend/include/DynamicCVPipeline/StandardizeOp/PatternMatchRewrites.h"

#include "DynamicCVPipeline/Common/Utils.h"

using namespace mlir;
using namespace triton;
using namespace CVSplit;

static constexpr const char *DEBUG_TYPE = "PatternMatchRewrites";
#define LOG_DEBUG(...)                                                         \
  LLVM_DEBUG(llvm::dbgs() << "\n[" << DEBUG_TYPE << "] " << __VA_ARGS__ << "\n")

static constexpr llvm::StringLiteral needSplitAllFuncNme[]{""};

void PatternMatchRewritePass::runOnOperation() {
  auto moduleOp = getOperation();

  if (CVPipeline::hasFallbackAttr(moduleOp)) {
    return;
  }

  LOG_DEBUG("Input mlir:\n" << moduleOp);

  bool needSplitAll = false;
  moduleOp.walk([&](func::FuncOp funcOp) -> WalkResult {
    if (!llvm::is_contained(needSplitAllFuncNme, funcOp.getSymName())) {
      return WalkResult::advance();
    }
    LOG_DEBUG("[INFO] Matmul should split anyway: " << funcOp.getSymName());
    needSplitAll = true;
    return WalkResult::interrupt();
  });

  auto *ctx = &getContext();
  RewritePatternSet patterns(ctx);
  patterns.add<SplitMatmulPattern>(ctx, needSplitAll);
  patterns.add<FoldExpandExtCollapse>(ctx);

  // the way we handle matmul dependencies across for blocks
  // requres the patternmatching to go top-down
  GreedyRewriteConfig config = GreedyRewriteConfig().setUseTopDownTraversal();
  if (llvm::failed(
          applyPatternsGreedily(moduleOp, std::move(patterns), config))) {
    LOG_DEBUG("matchAndRewrite does not converge!");
    CVPipeline::setFallbackAttr(moduleOp, CVPipeline::ERRCODE_FAILED);
    return;
  }
  LOG_DEBUG("Output mlir:\n" << moduleOp);
}

namespace mlir::triton::CVSplit {

std::unique_ptr<OperationPass<ModuleOp>> createPatternMatchRewritePass() {
  return std::make_unique<PatternMatchRewritePass>();
}

} // namespace mlir::triton::CVSplit
