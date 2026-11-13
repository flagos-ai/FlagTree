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

#ifndef TRITON_ADAPTER_DYNAMIC_CV_PIPELINE_COMMON_SYNC_WALL_H
#define TRITON_ADAPTER_DYNAMIC_CV_PIPELINE_COMMON_SYNC_WALL_H

#include "mlir/IR/Block.h"
#include "mlir/IR/Operation.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"

namespace mlir {
namespace CVPipeline {

class SyncWall {
public:
  explicit SyncWall(Block *block);

  // Source-order position of a block-level op inside the block (0-based).
  // Ops not owned by the block report position 0.
  unsigned positionOf(Operation *op) const;

  // True iff a synchronization op is strictly between @p a and @p b in the
  // block's linear source order.
  bool hasSyncBetween(Operation *a, Operation *b) const;

  // Number of synchronization ops strictly preceding @p op in source order.
  // Ops separated by at least one sync belong to different segments.
  unsigned segmentOf(Operation *op) const;

  bool sameSegment(Operation *a, Operation *b) const;

private:
  llvm::DenseMap<Operation *, unsigned> ordinal;
  llvm::DenseSet<unsigned> syncPositions;
};

} // namespace CVPipeline
} // namespace mlir

#endif // TRITON_ADAPTER_DYNAMIC_CV_PIPELINE_COMMON_SYNC_WALL_H
