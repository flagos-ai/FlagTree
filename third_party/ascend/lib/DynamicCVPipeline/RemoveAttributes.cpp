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

#include <memory>

#include "llvm/ADT/StringExtras.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Debug.h"

#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassRegistry.h"

#include "ascend/include/DynamicCVPipeline/Common/Utils.h"
#include "ascend/include/DynamicCVPipeline/RemoveAttributes.h"

using namespace mlir;
using namespace triton;
using namespace CVPipeline;

static constexpr const char *DEBUG_TYPE = "RemoveAttributes";
#define LOG_DEBUG(...)                                                         \
  LLVM_DEBUG(llvm::dbgs() << " [" << DEBUG_TYPE << "] " << __VA_ARGS__)

// if extra attr is needed, add to ut @
// third_party/ascend/unittest/Conversion/General/DynamicCVPipeline/test-remove-attrs.mlir
static constexpr llvm::StringLiteral kAttrsToRemove[]{
    kBlockId,           kCoreType,
    kTransferId,        kMainLoop,
    kCubeFirst,         kVectorFirst,
    kAddFromMatmul,     kIntraDeps,
    kIntraBuffer,       kAnalyzeFlagId,
    kLoopCarriedL0C,    kCrossCoreDeps,
    kMemCrossDeps,      kClone,
    kIntraBufCount,     kInterCoreBufCount,
    kLoadStoreBufCount, kInsertionOptimization,
    kDepMark,           kIntraDeps,
    kSubBlock,          kMergeSmallBlockFirstRunDone,
    kSplittedIf,        kLoopId};

void RemoveSsbufAttrPass::runOnOperation() {
  auto module = getOperation();

  if (CVPipeline::hasFallbackAttr(module)) {
    return;
  }

  module->walk([](Operation *op) {
    LOG_DEBUG("Removing ssbuf attrs of " << *op);
    for (auto attrName : kAttrsToRemove) {
      op->removeAttr(attrName);
    }
  });
}

namespace mlir::triton {

std::unique_ptr<OperationPass<ModuleOp>> createRemoveSsbufAttrPass() {
  return std::make_unique<RemoveSsbufAttrPass>();
}

void registerRemoveSsbufAttrPasses() {
  registerPass(createRemoveSsbufAttrPass);
}

} // namespace mlir::triton
