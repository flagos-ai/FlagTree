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

#include "DynamicCVPipeline/Common/SyncWall.h"

using namespace mlir;
using namespace mlir::CVPipeline;

SyncWall::SyncWall(Block *block) {
  unsigned idx = 0;
  block->walk<WalkOrder::PreOrder>([&](Operation *op) {
    Operation *owner = getAncestorInBlock(op, block);
    if (owner == nullptr) {
      return;
    }
    if (!ordinal.contains(owner)) {
      ordinal[owner] = idx++;
    }
    if (isExternalSyncOp(op)) {
      syncPositions.insert(ordinal[owner]);
    }
  });
}

unsigned SyncWall::positionOf(Operation *op) const {
  auto it = ordinal.find(op);
  if (it != ordinal.end()) {
    return it->second;
  }
  return 0;
}

bool SyncWall::hasSyncBetween(Operation *a, Operation *b) const {
  unsigned lo = positionOf(a);
  unsigned hi = positionOf(b);
  if (lo > hi) {
    unsigned tmp = lo;
    lo = hi;
    hi = tmp;
  }
  for (unsigned pos = lo + 1; pos < hi; ++pos) {
    if (syncPositions.contains(pos)) {
      return true;
    }
  }
  return false;
}

unsigned SyncWall::segmentOf(Operation *op) const {
  unsigned pos = positionOf(op);
  unsigned seg = 0;
  for (unsigned p = 0; p < pos; ++p) {
    if (syncPositions.contains(p)) {
      ++seg;
    }
  }
  return seg;
}

bool SyncWall::sameSegment(Operation *a, Operation *b) const {
  return segmentOf(a) == segmentOf(b);
}
