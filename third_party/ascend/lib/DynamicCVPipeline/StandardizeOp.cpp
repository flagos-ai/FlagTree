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
#include "llvm/Support/LogicalResult.h"
#include <memory>

#include "mlir/Pass/PassManager.h"
#include "mlir/Pass/PassRegistry.h"

#include "ascend/include/DynamicCVPipeline/AnalyzeDataFlow.h"
#include "ascend/include/DynamicCVPipeline/Common/Utils.h"
#include "ascend/include/DynamicCVPipeline/StandardizeOp.h"
#include "ascend/include/DynamicCVPipeline/StandardizeOp/PatternMatchRewrites.h"

using namespace mlir;
using namespace triton;
using namespace CVSplit;

static constexpr const char *DEBUG_TYPE = "StandardizeOp";
#define LOG_DEBUG(...)                                                         \
  LLVM_DEBUG(llvm::dbgs() << "\n[" << DEBUG_TYPE << "] " << __VA_ARGS__ << "\n")

namespace mlir::triton {

void StandardizeOpPass::runOnOperation() {
  auto op = getOperation();

  if (CVPipeline::hasFallbackAttr(op)) {
    return;
  }

  LOG_DEBUG("Input mlir:\n" << op);
  OpPassManager pm(op.getOperationName());
  pm.addPass(createPatternMatchRewritePass());

  if (llvm::failed(runPipeline(pm, op))) {
    LOG_DEBUG("Pipeline Failed!");
    CVPipeline::setFallbackAttr(op, CVPipeline::ERRCODE_FAILED);
    return;
  }

  bool findMayNotExec = false;
  op->walk([&](linalg::MatmulOp matmulOp) {
    if (matmulOp->hasAttr(CVPipeline::kMayNotExec)) {
      findMayNotExec = true;
    }
  });

  if (findMayNotExec) {
    LOG_DEBUG("Matmul may not execute!");
    CVPipeline::setFallbackAttr(op, CVPipeline::ERRCODE_IGNORED);
    return;
  }
}

std::unique_ptr<OperationPass<ModuleOp>> createStandardizeOpPass() {
  return std::make_unique<StandardizeOpPass>();
};

void registerStandardizeOpPasses() {
  registerPass(createStandardizeOpPass);
  registerPass(createPatternMatchRewritePass);
}

} // namespace mlir::triton
