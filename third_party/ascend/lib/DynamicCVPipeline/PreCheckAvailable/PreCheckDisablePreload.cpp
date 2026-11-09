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

#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Debug.h"

#include "DynamicCVPipeline/Common/BufferCountManager.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinOps.h"

#include "ascend/include/DynamicCVPipeline/Common/Utils.h"
#include "ascend/include/DynamicCVPipeline/PreCheckAvailable.h"

using namespace mlir;
using namespace triton;

// Functions that should use at most double buffering.
static const llvm::SmallVector<llvm::StringRef> kBlacklistFuncNames = {
    "_attn_bwd", "kernel_sdpa_fwd", "_swa_paged_decode_kernel",
    "paged_decode_fd_kernel"};

static constexpr const char *DEBUG_TYPE = "pre-check-disable-preload";
#define DBGS() (llvm::dbgs() << '[' << DEBUG_TYPE << "] ")
#define LDBG(...)                                                              \
  LLVM_DEBUG({                                                                 \
    DBGS();                                                                    \
    llvm::dbgs() << __VA_ARGS__ << "\n";                                       \
  })

void PreCheckDisablePreload::runOnOperation() {
  ModuleOp module = getOperation();

  if (CVPipeline::hasFallbackAttr(module)) {
    return;
  }

  func::FuncOp foundBlacklistFunc = nullptr;

  // Check for all blacklist operations
  module.walk([&](func::FuncOp func) -> WalkResult {
    llvm::StringRef funcName = func.getSymName();
    if (llvm::is_contained(kBlacklistFuncNames, funcName)) {
      foundBlacklistFunc = func;
      return WalkResult::interrupt();
    }
    return WalkResult::advance();
  });

  if (!foundBlacklistFunc) {
    LDBG("No preload blacklist func found, passed.");
    return;
  }

  // Keep the explicit 3-preload setting only for kernel_sdpa_fwd.
  if (foundBlacklistFunc.getSymName() == "kernel_sdpa_fwd") {
    auto intraBufferCount =
        module->getAttrOfType<IntegerAttr>(CVPipeline::kIntraBufCount);
    auto interBufferCount =
        module->getAttrOfType<IntegerAttr>(CVPipeline::kInterCoreBufCount);
    if (intraBufferCount && interBufferCount &&
        intraBufferCount.getInt() == 3 && interBufferCount.getInt() == 2) {
      return;
    }
  }

  LDBG("3-preload will be disabled because "
       << foundBlacklistFunc.getSymName()
       << " function was found, which is not supported by 3-preload now.");

  // This setting will disable the 3-preload feature.
  BufferCountManager bufferCountManager(module);
  bufferCountManager.setBufferCount(BufferCountManager::DepType::IntraCore, 2);
  bufferCountManager.setBufferCount(BufferCountManager::DepType::InterCore, 1);
}

namespace mlir {
namespace triton {

std::unique_ptr<OperationPass<ModuleOp>> createPreCheckDisablePreloadPass() {
  return std::make_unique<PreCheckDisablePreload>();
}

} // namespace triton
} // namespace mlir
