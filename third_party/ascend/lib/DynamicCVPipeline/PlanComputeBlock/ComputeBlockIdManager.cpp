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

#include "llvm/ADT/StringRef.h"
#include "llvm/Support/LogicalResult.h"

#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"

#include "ascend/include/DynamicCVPipeline/Common/Utils.h"
#include "ascend/include/DynamicCVPipeline/PlanComputeBlock/ComputeBlockIdManager.h"

namespace mlir {
namespace CVPipeline {

bool ComputeBlockIdManager::isWholeCubeReady(Operation *seedOp, llvm::DenseMap<Operation *, int> &indegree)
{
    auto id = getBlockIdByOp(seedOp);
    if (id == -1) {
        return (indegree[seedOp] == 0);
    }
    auto cubeBlock = getOpsByBlockId(id);
    for (auto op : cubeBlock) {
        if (!indegree.contains(op)) {
            continue;
        }
        if (indegree[op] != 0) {
            return false;
        }
    }
    return true;
}

bool ComputeBlockIdManager::isSameBlock(Operation *a, Operation *b)
{
    if (getBlockIdByOp(a) == getBlockIdByOp(b) && getBlockIdByOp(a) != -1) {
        return true;
    }
    return false;
}

unsigned int ComputeBlockIdManager::getNextId()
{
    std::lock_guard<std::mutex> lock(managerMutex);
    return cntComputeBlockId++;
}

llvm::LogicalResult ComputeBlockIdManager::markOpBlockId(Operation *op, int blockId)
{
    if (blockId < 0) {
        op->emitError("marking block id as negative");
        return llvm::failure();
    }
    if (!op) {
        return llvm::success();
    }
    MLIRContext *ctx = op->getContext();
    op->setAttr(kBlockId, IntegerAttr::get(IntegerType::get(ctx, blockIdWidth), blockId));
    return record(op, blockId);
}

llvm::SmallVector<Operation *> ComputeBlockIdManager::getOpsByBlockId(int blockId)
{
    if (blockId == -1) {
        return {};
    }
    std::lock_guard<std::mutex> lock(managerMutex);
    auto it = blockIdToOps.find(blockId);
    if (it == blockIdToOps.end()) {
        return {};
    }
    return llvm::SmallVector<Operation *>(it->second.begin(), it->second.end());
}

int ComputeBlockIdManager::getBlockIdByOp(Operation *op)
{
    std::lock_guard<std::mutex> lock(managerMutex);
    auto it = opToBlockId.find(op);
    if (it != opToBlockId.end()) {
        return it->second;
    }
    return -1;
}

llvm::LogicalResult ComputeBlockIdManager::markOpsWithNewId(llvm::SmallVectorImpl<Operation *> &ops)
{
    if (ops.empty()) {
        return llvm::success();
    }
    int id = static_cast<int>(getNextId());
    for (Operation *op : ops) {
        if (llvm::failed(markOpBlockId(op, id))) {
            return llvm::failure();
        }
    }
    return llvm::success();
}

void ComputeBlockIdManager::reset()
{
    std::lock_guard<std::mutex> lock(managerMutex);
    cntComputeBlockId = 0;
    blockIdToOps.clear();
    opToBlockId.clear();
}

llvm::LogicalResult ComputeBlockIdManager::record(Operation *op, int blockId)
{
    if (!op || blockId < 0) {
        return llvm::success();
    }
    std::lock_guard<std::mutex> lock(managerMutex);
    auto itOld = opToBlockId.find(op);
    if (itOld != opToBlockId.end()) {
        llvm::errs() << "Error: Operation already has a block id. Op: " << *op << ", old block id: " << itOld->second
                     << ", new block id: " << blockId << "\n";
        return llvm::failure();
    }

    opToBlockId[op] = blockId;
    auto &vec = blockIdToOps[blockId];
    for (Operation *existing : vec) {
        if (existing == op) {
            return llvm::success();
        }
    }
    vec.push_back(op);
    return llvm::success();
}

} // namespace CVPipeline
} // namespace mlir
