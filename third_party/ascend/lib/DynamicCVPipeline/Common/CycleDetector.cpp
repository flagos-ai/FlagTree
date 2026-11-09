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

#include "DynamicCVPipeline/Common/CycleDetector.h"
#include "mlir/IR/Operation.h"

namespace mlir::CVPipeline {

bool DependencyCycleDetector::detectCycleFrom(Operation *cur) {
  if (group.contains(cur)) {
    return true;
  }
  if (!visited.insert(cur).second) {
    return false;
  }

  bool createsCycle = false;

  depHelper.forEachUserInSameBlock(cur, [&](Operation *user) {
    if (isExternalSyncOp(user)) {
      return;
    }
    if (createsCycle)
      return;
    createsCycle =
        llvm::any_of(bm.getOpsInSameBlock(user),
                     [this](Operation *user) { return detectCycleFrom(user); });
    return;
  });

  return createsCycle;
}

bool DependencyCycleDetector::detectCycle() {
  llvm::DenseSet<Operation *> externalUsers;
  for (auto *op : group) {
    depHelper.forEachUserInSameBlock(op, [&](Operation *user) {
      if (!group.contains(user)) {
        externalUsers.insert(user);
      }
    });
  }
  return llvm::any_of(externalUsers, [this](Operation *op) {
    return this->detectCycleFrom(op);
  });
}

} // namespace mlir::CVPipeline
