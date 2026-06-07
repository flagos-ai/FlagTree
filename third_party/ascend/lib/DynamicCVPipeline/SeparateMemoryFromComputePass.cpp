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

#include "ascend/include/DynamicCVPipeline/SeparateMemoryFromComputePass.h"
#include "mlir/Pass/PassManager.h"
#include "llvm/Support/Debug.h"
#include "ascend/include/DynamicCVPipeline/SeparateMemoryFromCompute/AddMultiBufferToGMLoadPass.h"
#include "ascend/include/DynamicCVPipeline/SeparateMemoryFromCompute/AsyncLoadHoistingPass.h"

static constexpr const char *DEBUG_TYPE = "separate-memory-from-compute";
#define DBGS() (llvm::dbgs() << '[' << DEBUG_TYPE << "] ")
#define LDBG(X) LLVM_DEBUG(DBGS() << (X) << "\n")

using namespace mlir;
using namespace triton;

static constexpr int kDefaultBufferDepth = 2;

void SeparateMemoryFromComputePass::runOnOperation()
{
  ModuleOp module = getOperation();

  int depth = kDefaultBufferDepth;

  if (depth <= 1) {
    LDBG("Buffer depth <= 1, skip multi-buffer transformation");
    return;
  }

  OpPassManager pm(module.getOperationName());
  LDBG("Enter SeparateMemoryFromCompute pass");

  // Step 1: Hoist memory operations out of compute blocks
  pm.addPass(createAsyncLoadHoistingPass());

  // Step 2: Apply multi-buffering to memory operations
  pm.addPass(createAddMultiBufferToGMLoadPass());

  if (failed(runPipeline(pm, module))) {
    module->emitError() << "[" << DEBUG_TYPE << "] Pass failed!";
    signalPassFailure();
  }

  LDBG("Process successfully");
}

namespace mlir {
namespace triton {

std::unique_ptr<OperationPass<ModuleOp>> createSeparateMemoryFromComputePass()
{
  return std::make_unique<SeparateMemoryFromComputePass>();
}

} // namespace triton
} // namespace mlir