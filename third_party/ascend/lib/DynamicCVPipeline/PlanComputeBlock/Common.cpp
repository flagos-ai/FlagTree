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
 
#include "ascend/include/DynamicCVPipeline/PlanComputeBlock/Common.h"

namespace mlir {
namespace CVPipeline {

void initializeIndegreeForBlock(Block *block, llvm::DenseMap<Operation *, int> &indegree,
                                const MemoryDependenceGraph &memGraph)
{
    auto &bm = ComputeBlockIdManager::getInstance();

    block->walk([&](Operation *op) {
        if (op->getBlock() != block) { return; }
        indegree[op] = 0;
        // We need to consider op itself && op's region-contained ops.
        op->walk([&](Operation *nestedOp) {
            for (auto inValue : nestedOp->getOperands()) {
                if (auto defOp = inValue.getDefiningOp()) {
                    if (defOp->getBlock() == block && !bm.isSameBlock(defOp, op)) {
                        indegree[op]++;
                    }
                }
            }

            for (auto memDepUser : memGraph.getExecBefore(nestedOp)) {
                if (memDepUser->getBlock() == block && !bm.isSameBlock(memDepUser, op)) {
                    indegree[op]++;
                }
            }
        });
    });
}

Operation *getAncestorInBlock(Operation *inner, Block *block)
{
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
