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

#include "mlir/Pass/PassManager.h"
#include "llvm/Support/Debug.h"
#include "ascend/include/DynamicCVPipeline/Passes.h"
#include "ascend/include/DynamicCVPipeline/PlanComputeBlockPass.h"
#include "ascend/include/DynamicCVPipeline/SeparateMemoryFromComputePass.h"
#include "ascend/include/DynamicCVPipeline/SplitDataflowPass.h"
#include "ascend/include/DynamicCVPipeline/AddControlFlowCondition.h"
#include "ascend/include/DynamicCVPipeline/AllocMultiCache.h"
#include "ascend/include/DynamicCVPipeline/PlanComputeBlock/PlanCubeBlockPass.h"

static constexpr const char *DEBUG_TYPE = "AddDynamicCVPipeline";
#define DBGS() (llvm::dbgs() << '[' << DEBUG_TYPE << "] ")
#define LDBG(X) LLVM_DEBUG(DBGS() << (X) << "\n")

namespace mlir {
namespace triton {
#define GEN_PASS_DEF_ADDDYNAMICCVPIPELINE
#include "ascend/include/DynamicCVPipeline/Passes.h.inc"
} // namespace triton
} // namespace mlir

using namespace mlir;

AddDynamicCVPipelinePass::AddDynamicCVPipelinePass(
    const AddDynamicCVPipelineOptions &options)
    : AddDynamicCVPipelineBase(options) {}

void AddDynamicCVPipelinePass::runOnOperation()
{
    auto moduleOp = getOperation();
    compileOn91095Flag = this->compileOn91095;

    LDBG("Enter pass");

    if (!compileOn91095Flag) {
        llvm::errs() << "Add-dynamic-cv-pipeline is only supported on 91095 now.\n";
        return;
    }

    PassManager pm(&getContext(), moduleOp.getOperationName());

    // todo: add related passes.
    pm.addPass(createPlanComputeBlockPass());
    pm.addPass(createSplitDataflowPass());
    pm.addPass(createSeparateMemoryFromComputePass());
    pm.addPass(createAllocMultiCachePass());
    pm.addPass(createAddControlFlowConditionPass());

    if (failed(runPipeline(pm, getOperation()))) {
        moduleOp->emitError() << "[" << DEBUG_TYPE << "] Pass failed!";
        signalPassFailure();
    }

    LDBG("Process successfully");
}

std::unique_ptr<OperationPass<ModuleOp>> mlir::triton::createAddDynamicCVPipelinePass(
    const AddDynamicCVPipelineOptions &options)
{
    return std::make_unique<AddDynamicCVPipelinePass>(options);
}
