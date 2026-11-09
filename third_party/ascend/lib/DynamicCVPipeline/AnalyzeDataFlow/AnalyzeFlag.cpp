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
#include "bishengir/Dialect/HIVM/IR/HIVM.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/BuiltinTypes.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/Support/Debug.h"

static constexpr const char *DEBUG_TYPE = "analyze-flag";
#define DBGS() (llvm::dbgs() << '[' << DEBUG_TYPE << "] ")
#define LDBG(...)                                                              \
  LLVM_DEBUG({                                                                 \
    DBGS();                                                                    \
    llvm::dbgs() << __VA_ARGS__;                                               \
    llvm::dbgs() << "\n";                                                      \
  })

using namespace llvm;
using namespace mlir;
using namespace triton;
using namespace CVPipeline;

namespace {

static constexpr const char *syncFlagIdAttr = "static_flag_id";

static bool checkFlagIdValidity(ModuleOp module) {
  bool shouldReturn = false;
  int invalidFlagNum = 0;

  module.walk([&](Operation *op) -> WalkResult {
    if (!isa<hivm::SyncBlockSetOp>(op) && !isa<hivm::SyncBlockWaitOp>(op)) {
      return WalkResult::advance();
    }

    if (auto intAttr = op->getAttrOfType<IntegerAttr>(syncFlagIdAttr)) {
      int flag = static_cast<int>(intAttr.getInt());
      if (flag < 0 || flag > 14) {
        shouldReturn = true;
        invalidFlagNum++;
      }
    }

    return WalkResult::advance();
  });

  if (shouldReturn) {
    LDBG("[warning]: flag_id is not enough for transfer, invalidFlagNum: "
         << invalidFlagNum << "\n");
  }
  return shouldReturn;
}

} // namespace

void AnalyzeFlagPass::runOnOperation() {
  ModuleOp module = getOperation();

  LDBG("Before AnalyzeFlag:\n" << module << "\n");

  if (CVPipeline::hasFallbackAttr(module)) {
    return;
  }

  if (checkFlagIdValidity(module)) {
    setFallbackAttr(module, ERRCODE_IGNORED);
    return;
  }

  LDBG("After AnalyzeFlag:\n");
}

namespace mlir {
namespace triton {

std::unique_ptr<OperationPass<ModuleOp>> createAnalyzeFlagPass() {
  return std::make_unique<AnalyzeFlagPass>();
}

} // namespace triton
} // namespace mlir
