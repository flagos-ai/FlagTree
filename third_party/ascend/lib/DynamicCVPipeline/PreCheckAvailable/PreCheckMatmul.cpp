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

#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/IR/BuiltinOps.h"

#include "ascend/include/DynamicCVPipeline/Common/Utils.h"
#include "ascend/include/DynamicCVPipeline/PreCheckAvailable.h"

using namespace mlir;
using namespace triton;

static constexpr const char *DEBUG_TYPE = "pre-check-matmul";
#define DBGS() (llvm::dbgs() << '[' << DEBUG_TYPE << "] ")
#define LDBG(X) LLVM_DEBUG(DBGS() << (X) << "\n")

void PreCheckMatmul::runOnOperation() {
  ModuleOp module = getOperation();

  if (CVPipeline::hasFallbackAttr(module)) {
    return;
  }

  linalg::MatmulOp firstMatmulOp = nullptr;

  module.walk([&](linalg::MatmulOp matmulOp) -> WalkResult {
    firstMatmulOp = matmulOp;
    return WalkResult::interrupt();
  });

  if (firstMatmulOp) {
    LDBG("The linalg.matmul operation is found, passed.");
    return;
  }

  LDBG("SSBUFFER will be skipped because no linalg.matmul operation was found, "
       "which indicating that this op is a pure vector computation.");
  CVPipeline::setFallbackAttr(module, CVPipeline::ERRCODE_IGNORED);
}

namespace mlir {
namespace triton {

std::unique_ptr<OperationPass<ModuleOp>> createPreCheckMatmulPass() {
  return std::make_unique<PreCheckMatmul>();
}

} // namespace triton
} // namespace mlir
