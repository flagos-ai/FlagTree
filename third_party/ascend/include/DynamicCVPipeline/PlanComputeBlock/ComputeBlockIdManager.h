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

#ifndef TRITON_ADAPTER_DYNAMIC_CV_PIPELINE_PLAN_COMPUTE_BLOCK_COMPUTE_BLOCK_ID_MANAGER_H
#define TRITON_ADAPTER_DYNAMIC_CV_PIPELINE_PLAN_COMPUTE_BLOCK_COMPUTE_BLOCK_ID_MANAGER_H

#include <mutex>
#include "mlir/IR/Operation.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/LogicalResult.h"

namespace mlir {
namespace CVPipeline {

/**
    * the class is to promise CUBEID and VECTORID are unified.
 */
class ComputeBlockIdManager {
public:
    static ComputeBlockIdManager &getInstance()
    {
        static ComputeBlockIdManager instance;
        return instance;
    }

    unsigned int getNextId();

    bool isSameBlock(Operation *a, Operation *b);
    bool isWholeCubeReady(Operation *seedOp, llvm::DenseMap<Operation *, int> &indegree);

    llvm::LogicalResult markOpBlockId(Operation *op, int blockId);

    llvm::SmallVector<Operation *> getOpsByBlockId(int blockId);
    int getBlockIdByOp(Operation *op);
    llvm::LogicalResult markOpsWithNewId(llvm::SmallVectorImpl<Operation *> &ops);
    void reset();

private:
    ComputeBlockIdManager() : cntComputeBlockId(0) {}
    llvm::LogicalResult record(Operation *op, int blockId);

    unsigned int cntComputeBlockId;
    llvm::DenseMap<int, llvm::SmallVector<Operation *>> blockIdToOps;
    llvm::DenseMap<Operation *, int> opToBlockId;
    mutable std::mutex managerMutex;
    const int blockIdWidth = 32;
};

} // namespace CVPipeline
} // namespace mlir

#endif // TRITON_ADAPTER_DYNAMIC_CV_PIPELINE_PLAN_COMPUTE_BLOCK_COMPUTE_BLOCK_ID_MANAGER_H
