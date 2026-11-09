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

#include "llvm/ADT/iterator.h"

#include "mlir/IR/Value.h"

#include "ascend/include/DynamicCVPipeline/PlanComputeBlock/Common.h"

#include "DynamicCVPipeline/PlanComputeBlock/ComputeBlockIdManager.h"

namespace mlir {
namespace CVPipeline {

void initializeIndegreeForBlock(Block *block,
                                llvm::DenseMap<Operation *, int> &indegree,
                                const DependencyHelper &depHelper,
                                ComputeBlockIdManager &bm) {
  for (auto *op : llvm::make_pointer_range(block->getOperations())) {
    indegree[op] = 0;
    depHelper.forEachSource(op, [&](Operation *source) {
      if (source->getBlock() == block && !bm.isSameBlock(source, op)) {
        indegree[op]++;
      }
    });
  }
}

Operation *getAncestorInBlock(Operation *inner, Block *block) {
  Operation *cur = inner;
  while (cur) {
    if (cur->getBlock() == block) {
      return cur;
    }
    cur = cur->getParentOp();
  }
  return nullptr;
}

} // namespace CVPipeline
} // namespace mlir
