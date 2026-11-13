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

#include "ascend/include/DynamicCVPipeline/ComputeBlockOptPass.h"
#include "DynamicCVPipeline/ComputeBlockOpt/Passes.h"
#include "DynamicCVPipeline/PlanComputeBlock/Passes.h"
#include "DynamicCVPipeline/PlanComputeBlock/ReorderOpsByBlockId.h"
#include "ascend/include/DynamicCVPipeline/Common/Utils.h"
#include "ascend/include/DynamicCVPipeline/ComputeBlockOpt/MergeCubeBlockPass.h"
#include "ascend/include/DynamicCVPipeline/ComputeBlockOpt/Passes.h"
#include "ascend/include/DynamicCVPipeline/PlanComputeBlockPass.h"

#include "mlir/Pass/PassManager.h"

using namespace mlir;
using namespace triton;

void ComputeBlockOptPass::runOnOperation() {
  ModuleOp module = getOperation();

  if (CVPipeline::hasFallbackAttr(module)) {
    return;
  }

  OpPassManager pm(module.getOperationName());

  /**
      First, perform UnifyAllocBlock to merge load semantic operations into a
     unified block. Then, use UBUsageOpt to find the smallest UB dependency
     location and divide the computation blocks.
   */
  pm.addPass(createUnifyAllocBlockPass());
  pm.addPass(createReorderOpsByBlockIdPass());

  pm.addPass(createMergeVectorIfBlockPass());
  pm.addPass(createReorderOpsByBlockIdPass());
  pm.addPass(createReorderOpsByBlockIdPass());

  pm.addPass(createExpLoadPatternPass());
  pm.addPass(createUBUsageOptPass());
  pm.addPass(createBroadcastUBOptPass());
  pm.addPass(createPosMaskPatternPass());
  pm.addPass(createMergeSameSourceAxisPass());
  pm.addPass(createReorderOpsByBlockIdPass());
  pm.addPass(createMergeSmallBlockPass());
  pm.addPass(createReorderOpsByBlockIdPass());

  pm.addPass(createSinkI1ProducersIntoUsersPass());
  pm.addPass(createReorderOpsByBlockIdPass());

  pm.addPass(createFixpipeOptPass());
  pm.addPass(createSplitIfByBlockIdPass());
  pm.addPass(createReorderOpsByBlockIdPass());
  pm.addPass(createMoveLoadIntoUserPass());
  pm.addPass(createUnifyStoreBlockPass());
  pm.addPass(createExpSubfPatternPass());
  pm.addPass(createReorderOpsByBlockIdPass());
  pm.addPass(createMergeSmallBlockPass());
  pm.addPass(createReorderOpsByBlockIdPass());

  pm.addPass(createMergeComputeBlockPass());
  pm.addPass(createReorderOpsByBlockIdPass());

  pm.addPass(createMergeCubeBlockPass());
  pm.addPass(createMergeInputInitSharedCubeBlockPass());
  pm.addPass(createReorderOpsByBlockIdPass());
  pm.addPass(createRelocateMemrefDeclPass());
  if (failed(runPipeline(pm, module))) {
    if (!CVPipeline::hasFallbackAttr(module)) {
      CVPipeline::setFallbackAttr(module, CVPipeline::ERRCODE_FAILED);
    }
    return;
  }
}

namespace mlir {
namespace triton {

std::unique_ptr<OperationPass<ModuleOp>> createComputeBlockOptPass() {
  return std::make_unique<ComputeBlockOptPass>();
}

void registerComputeBlockOptPasses() {
  registerPass([]() -> std::unique_ptr<mlir::Pass> {
    return createComputeBlockOptPass();
  });
  registerPass(createUBUsageOptPass);
  registerPass(createMergeSameSourceAxisPass);
  registerPass(createUnifyAllocBlockPass);
  registerPass(createMergeVectorIfBlockPass);
  registerPass(createFixpipeOptPass);
  registerPass(createUnifyStoreBlockPass);
  registerPass(createExpSubfPatternPass);
  registerPass(createExpLoadPatternPass);
  registerPass(createSinkI1ProducersIntoUsersPass);
  registerPass(createBroadcastUBOptPass);
  registerPass(createMoveLoadIntoUserPass);
  registerPass(createPosMaskPatternPass);
  registerPass(createMergeSmallBlockPass);
  registerPass(createSplitIfByBlockIdPass);
  registerPass(createMergeComputeBlockPass);
  registerPass(createMergeInputInitSharedCubeBlockPass);
  registerPass(createMergeCubeBlockPass);
  registerPass(createRelocateMemrefDeclPass);
}

} // namespace triton
} // namespace mlir
