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

#ifndef TRITON_DYNAMIC_CV_PIPELINE_COMMON_CYCLEDETECTOR_H
#define TRITON_DYNAMIC_CV_PIPELINE_COMMON_CYCLEDETECTOR_H

#include "DynamicCVPipeline/Common/DependencyHelper.h"
#include "DynamicCVPipeline/PlanComputeBlock/ComputeBlockIdManager.h"
#include "mlir/IR/Operation.h"
#include "mlir/Support/LLVM.h"
namespace mlir::CVPipeline {

class DependencyCycleDetector {
  const DenseSet<Operation *> &group;
  llvm::DenseSet<mlir::Operation *> visited;
  const DependencyHelper depHelper;
  ComputeBlockIdManager &bm;
  Block *const block;

  bool detectCycleFrom(Operation *cur);

public:
  DependencyCycleDetector(Block *block, const DependencyHelper &depHelper,
                          llvm::DenseSet<mlir::Operation *> &group,
                          ComputeBlockIdManager &bm)
      : block(block), depHelper(depHelper), group(group), bm(bm) {}

  bool detectCycle();
};

} // namespace mlir::CVPipeline

#endif
