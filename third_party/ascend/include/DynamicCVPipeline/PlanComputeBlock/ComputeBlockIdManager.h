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

#include <optional>

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/LogicalResult.h"

#include "mlir/IR/Operation.h"

#include "DynamicCVPipeline/Common/Utils.h"

namespace mlir {
namespace CVPipeline {

/**
 * the class is to promise CUBEID and VECTORID are unified.
 */
class ComputeBlockIdManager {
public:
  ComputeBlockIdManager(Operation *root);
  bool isSameBlock(Operation *a, Operation *b);
  bool isWholeCubeReady(Operation *seedOp,
                        llvm::DenseMap<Operation *, int> &indegree);

  llvm::LogicalResult markOpBlockId(Operation *op);
  llvm::LogicalResult markOpsWithNewId(llvm::ArrayRef<Operation *> ops);
  void updateBlockIdWithInner(Operation *parentOp, int targetId);
  void updateBlockId(Operation *op, int blockId);

  void forgetOp(Operation *op);

  bool shouldInheritFromParent(Block *block, CoreType requiredCoreType) const;
  llvm::LogicalResult inheritFromParent(Block *block);

  llvm::SmallVector<Operation *> getOpsByBlockId(int blockId) const;
  llvm::ArrayRef<Operation *> getOpsRefByBlockId(int blockId) const;

  // Get operations that share the same block_id AND mlir block of op
  llvm::SmallVector<Operation *> getOpsInSameBlock(Operation *op) const;

  std::optional<int> getBlockIdByOpOpt(Operation *op) const;
  int getNextId();

  int getBlockIdByOp(Operation *op);

  ~ComputeBlockIdManager() = default;
  ComputeBlockIdManager(const ComputeBlockIdManager &) = delete;
  ComputeBlockIdManager &operator=(const ComputeBlockIdManager &) = delete;
  ComputeBlockIdManager(ComputeBlockIdManager &&) = default;
  ComputeBlockIdManager &operator=(ComputeBlockIdManager &&) = default;

private:
  int cntComputeBlockId = 0;
  llvm::DenseMap<int, llvm::SmallVector<Operation *>> blockIdToOps;
  llvm::DenseMap<Operation *, int> opToBlockId;
  static constexpr int kBlockIdWidth = 32;
  llvm::LogicalResult markAndRecord(Operation *op, int blockId);
};

} // namespace CVPipeline
} // namespace mlir

#endif // TRITON_ADAPTER_DYNAMIC_CV_PIPELINE_PLAN_COMPUTE_BLOCK_COMPUTE_BLOCK_ID_MANAGER_H
