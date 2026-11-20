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

#include "llvm/ADT/STLFunctionalExtras.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/LogicalResult.h"

#include "mlir/Dialect/SCF/IR/SCF.h"

#include "Common.h"
#include "DynamicCVPipeline/Common/Utils.h"

namespace mlir::CVPipeline::SplitIf {

llvm::FailureOr<WalkMainLoopResult>
walkMainLoop(Operation *op,
             llvm::function_ref<llvm::LogicalResult(Operation *)> pred) {
  CoreType coreType = CVPipeline::getOpCoreType(op);
  bool containsMainLoop = false;
  for (auto &region : op->getRegions()) {
    for (auto &block : region.getBlocks()) {
      for (auto &nestedOp : block) {
        auto nestedResult = walkMainLoop(&nestedOp, pred);
        if (failed(nestedResult)) {
          return failure();
        }
        auto nested = nestedResult.value();
        coreType =
            static_cast<CVPipeline::CoreType>(coreType | nested.coreType);
        containsMainLoop = containsMainLoop || nested.containsMainLoop;
      }
    }
  }

  if (llvm::isa<scf::WhileOp, scf::ForOp>(op) && coreType == CUBE_AND_VECTOR &&
      !containsMainLoop) {
    if (pred(op).failed()) {
      return failure();
    }
    containsMainLoop = true;
  }

  return WalkMainLoopResult{coreType, containsMainLoop};
}

} // namespace mlir::CVPipeline::SplitIf
