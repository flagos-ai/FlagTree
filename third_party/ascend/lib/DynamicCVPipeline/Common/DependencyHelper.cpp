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

#include "ascend/include/DynamicCVPipeline/Common/DependencyHelper.h"
#include "ascend/include/DynamicCVPipeline/Common/Utils.h"
#include "mlir/Support/WalkResult.h"

namespace mlir::CVPipeline {

namespace {

using SourceMode = DependencyHelper::SourceMode;

} // namespace

mlir::WalkResult DependencyHelper::forEachUserImpl(Operation *op,
                                                   CallbackFn pred) const {
  if (!op)
    return WalkResult::advance();
  for (auto *user : op->getUsers()) {
    if (pred(user).wasInterrupted()) {
      return WalkResult::interrupt();
    }
  }
  for (auto *user : memGraph.getExecAfter(op)) {
    if (pred(user).wasInterrupted()) {
      return WalkResult::interrupt();
    }
  }
  return WalkResult::advance();
}

template <SourceMode SM>
mlir::WalkResult DependencyHelper::forEachSourceImpl(Operation *op,
                                                     CallbackFn pred) const {
  return op->walk(
      [this, op, pred = std::forward<CallbackFn>(pred)](Operation *subOp) {
        for (auto operand : subOp->getOperands()) {
          if (auto *defOp = operand.getDefiningOp(); defOp) {
            if (!op->isAncestor(defOp)) {
              if (pred(defOp).wasInterrupted()) {
                return WalkResult::interrupt();
              }
            }
            continue;
          }

          if constexpr (SM == SourceMode::AcrossIterArg) {
            if (auto *defOp = getLoopCarriedDefOp(operand, op->getBlock())) {
              if (pred(defOp).wasInterrupted()) {
                return WalkResult::interrupt();
              }
            }
          }
        }
        for (auto *source : memGraph.getExecBefore(subOp)) {
          if (!op->isAncestor(
                  source)) { // this filters only the outer mem dependencies
            if (pred(source).wasInterrupted()) {
              return WalkResult::interrupt();
            }
          }
        }
        return WalkResult::advance();
      });
}

// Instantiate concrete functions for linking
template mlir::WalkResult
DependencyHelper::forEachSourceImpl<SourceMode::Default>(mlir::Operation *op,
                                                         CallbackFn pred) const;

template mlir::WalkResult
DependencyHelper::forEachSourceImpl<SourceMode::AcrossIterArg>(
    mlir::Operation *op, CallbackFn pred) const;
} // namespace mlir::CVPipeline
