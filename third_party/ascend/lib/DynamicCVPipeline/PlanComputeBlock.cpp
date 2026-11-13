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

#include "ascend/include/DynamicCVPipeline/Common/Utils.h"
#include "ascend/include/DynamicCVPipeline/PlanComputeBlock/OpClassifier.h"
#include "ascend/include/DynamicCVPipeline/PlanComputeBlockPass.h"
#include "mlir/Pass/PassManager.h"

#include "DynamicCVPipeline/PlanComputeBlock/Passes.h"
#include "DynamicCVPipeline/PlanComputeBlock/PlanCubeBlockPass.h"
#include "DynamicCVPipeline/PlanComputeBlock/ReorderOpsByBlockId.h"
#include "ascend/include/DynamicCVPipeline/PlanComputeBlock/ComputeBlockIdManager.h"
#include "ascend/include/DynamicCVPipeline/PlanComputeBlock/Passes.h"
#include "ascend/include/DynamicCVPipeline/PlanComputeBlockPass.h"
#include "llvm/Support/Debug.h"

using namespace mlir;
using namespace triton;

static constexpr const char *DEBUG_TYPE = "plan-compute-block";
#define LOG_DEBUG(...)                                                         \
  LLVM_DEBUG(llvm::dbgs() << " [" << DEBUG_TYPE << "] " << __VA_ARGS__)

// Run the pass
void PlanComputeBlockPass::runOnOperation() {
  ModuleOp module = getOperation();

  if (CVPipeline::hasFallbackAttr(module)) {
    return;
  }

  OpPassManager pm(module.getOperationName());
  LOG_DEBUG("Enter pass.\n");

  // Step 1: Run OpClassifierPass to classify operations
  pm.addPass(createOpClassifierPass());

  // Step 2: Partition compute blocks for core_type=cube
  pm.addPass(createPlanCubeBlockPass());

  // Step 3: Partition compute blocks for core_type=vector
  pm.addPass(createPlanVectorBlockPass());

  // Step 4: Reorder
  pm.addPass(createReorderOpsByBlockIdPass());

  if (failed(runPipeline(pm, module))) {
    if (!CVPipeline::hasFallbackAttr(module)) {
      LOG_DEBUG("Pass failed!\n");
      CVPipeline::setFallbackAttr(module, CVPipeline::ERRCODE_FAILED);
    }
    return;
  }

  LOG_DEBUG("Process successfully\n");
}

namespace mlir {
namespace triton {
std::unique_ptr<OperationPass<ModuleOp>> createPlanComputeBlockPass() {
  return std::make_unique<PlanComputeBlockPass>();
}

void registerPlanComputeBlockPasses() {
  registerPass([]() -> std::unique_ptr<mlir::Pass> {
    return createPlanComputeBlockPass();
  });
  registerPass(createPlanCubeBlockPass);
  registerPass(createPlanVectorBlockPass);
  registerPass(createReorderOpsByBlockIdPass);
}

} // namespace triton
} // namespace mlir
