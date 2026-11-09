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

#include "ascend/include/DynamicCVPipeline/AnalyzeDataFlow.h"
#include "ascend/include/DynamicCVPipeline/Common/Utils.h"
#include "mlir/Pass/PassManager.h"
#include "llvm/Support/Debug.h"

static constexpr const char *DEBUG_TYPE = "analyze-data-flow";
#define DBGS() (llvm::dbgs() << '[' << DEBUG_TYPE << "] ")
#define LDBG(X) LLVM_DEBUG(DBGS() << (X) << "\n")

using namespace mlir;
using namespace triton;

void AnalyzeDataFlowPass::runOnOperation() {
  ModuleOp module = getOperation();

  if (CVPipeline::hasFallbackAttr(module)) {
    return;
  }

  LDBG("Enter AnalyzeDataFlow pass.");

  PassManager pm(&getContext(), module.getOperationName());

  pm.addPass(createAnalyzeNamePass());

  pm.addPass(createAnalyzeScopePass());

  pm.addPass(createAnalyzeArgsPass());

  pm.addPass(createAnalyzeFlagPass());

  pm.addPass(createAnalyzeCubeContolFLowInputChainPass());

  pm.addPass(createAnalyzeWhileConditionArgsPass());

  pm.addPass(createAnalyzeSplitIfPass());

  if (failed(runPipeline(pm, module))) {
    if (!CVPipeline::hasFallbackAttr(module)) {
      LDBG("Pass failed; fallback to compilation without dynamic CV pipeline.");
      CVPipeline::setFallbackAttr(module, CVPipeline::ERRCODE_FAILED);
    }
  }

  LDBG("Exit AnalyzeDataFlow pass.");
}

namespace mlir {
namespace triton {

std::unique_ptr<OperationPass<ModuleOp>> createAnalyzeDataFlowPass() {
  return std::make_unique<AnalyzeDataFlowPass>();
}

void registerAnalyzeDataFlowPasses() {
  registerPass(createAnalyzeNamePass);
  registerPass(createAnalyzeArgsPass);
  registerPass(createAnalyzeFlagPass);
  registerPass(createAnalyzeScopePass);
  registerPass(createAnalyzeDataFlowPass);
  registerPass(createAnalyzeWhileConditionArgsPass);
  registerPass(createAnalyzeCubeContolFLowInputChainPass);
  registerPass(createAnalyzeSplitIfPass);
}

} // namespace triton
} // namespace mlir
