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

#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Pass/PassRegistry.h"

#include "ascend/include/DynamicCVPipeline/Common/Utils.h"
#include "ascend/include/DynamicCVPipeline/PreCheckAvailable.h"

using namespace mlir;
using namespace triton;

static constexpr const char *DEBUG_TYPE =
    "pre-check-dynamic-cv-pipeline-available";
#define DBGS() (llvm::dbgs() << '[' << DEBUG_TYPE << "] ")
#define LDBG(...)                                                              \
  LLVM_DEBUG({                                                                 \
    DBGS();                                                                    \
    llvm::dbgs() << __VA_ARGS__ << "\n";                                       \
  })

void PreCheckAvailablePass::runOnOperation() {
  ModuleOp module = getOperation();

  if (CVPipeline::hasFallbackAttr(module)) {
    return;
  }

  LDBG("Enter PreCheckAvailable pass.");
  PassManager pm(&getContext(), module.getOperationName());

  LDBG("Before PreCheck:\n" << module);
  pm.addPass(createPreCheckBlacklistPass());
  pm.addPass(createPreCheckMatmulPass());
  pm.addPass(createPreCheckDisablePreloadPass());

  if (failed(runPipeline(pm, module))) {
    CVPipeline::setFallbackAttr(module, CVPipeline::ERRCODE_IGNORED);
    return;
  }

  LDBG("Exit PreCheckAvailable pass.");
}

namespace mlir {
namespace triton {

std::unique_ptr<OperationPass<ModuleOp>> createPreCheckAvailablePass() {
  return std::make_unique<PreCheckAvailablePass>();
}

void registerPreCheckAvailablePasses() {
  registerPass(createPreCheckBlacklistPass);
  registerPass(createPreCheckMatmulPass);
  registerPass(createPreCheckDisablePreloadPass);
  registerPass(createPreCheckAvailablePass);
}

} // namespace triton
} // namespace mlir
