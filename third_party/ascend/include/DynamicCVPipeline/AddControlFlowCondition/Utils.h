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

#ifndef TRITON_ADAPTER_DYNAMIC_CV_PIPELINE_UTILS_H
#define TRITON_ADAPTER_DYNAMIC_CV_PIPELINE_UTILS_H

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/SmallVector.h"
#include "mlir/Dialect/SCF/IR/SCF.h"

namespace mlir {
namespace triton {

// Collect all nested ops within an operation's regions
LogicalResult collectAllNestedOps(Operation *op, llvm::DenseSet<Operation *> &regionOps);

// Group operations by their block_id attribute
LogicalResult collectOpsByBlockId(scf::ForOp forOp,
                                  llvm::DenseMap<int, SmallVector<Operation *>> &blockOps);

// Topological sort of operations based on operand dependencies
LogicalResult topologicalSort(llvm::DenseSet<Operation *> &ops,
                              llvm::DenseMap<Operation *, int> *opOrder,
                              SmallVectorImpl<Operation *> &sorted);

LogicalResult topologicalSort(SmallVector<Operation *> &ops);

// Get block_ids in order of appearance in for loop body
SmallVector<int> getBlockIdsInOrder(scf::ForOp forOp);
} // namespace triton
} // namespace mlir
#endif // TRITON_ADAPTER_DYNAMIC_CV_PIPELINE_UTILS_H
