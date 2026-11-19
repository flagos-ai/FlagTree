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

#ifndef TRITON_ADAPTER_DYNAMIC_CV_PIPELINE_SPLIT_DATAFLOW_FLAG_ID_REUSE_H
#define TRITON_ADAPTER_DYNAMIC_CV_PIPELINE_SPLIT_DATAFLOW_FLAG_ID_REUSE_H

#include "ascend/include/DynamicCVPipeline/Common/MemoryEffectsTracker.h"
#include "ascend/include/DynamicCVPipeline/SplitDataflow/DataDependencyAnalysis.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Operation.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Support/LogicalResult.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SmallSet.h"
#include "llvm/ADT/SmallVector.h"

namespace mlir {
namespace triton {

class FlagIdReuseManager {

public:
  FlagIdReuseManager() {}

  void insertRelationBetweenSetAndWait(Operation *before, Operation *after);
  DenseMap<int, int>
  reuseInterCoreTransferFlagIds(const llvm::SmallVector<Operation *> &syncOps);

private:
  static int getFlagId(Operation *op);
  bool hasPath(llvm::SmallSet<Operation *, CVPipeline::INIT_SIZE> &visited,
               Operation *from, Operation *to);
  void preworkForAnalyze(const llvm::SmallVector<Operation *> &syncOps);

  enum class SyncDir { VectorToCube, CubeToVector };
  SyncDir getFlagDirection(int flagId);

  // Lifecycle boundary of a flag group: earliest set (acquire) / latest wait
  // (release).
  Operation *getEarliestSet(int flagId);
  Operation *getLatestWait(int flagId);
  // p is provably ordered before q: same op, same-block program order, or
  // reachable through the happens-before relations graph.
  bool opPrecedes(Operation *p, Operation *q);
  // `before`'s release point precedes `after`'s acquire point.
  bool flagReleasedBefore(int before, int after);
  // Two flags interfere iff neither is fully released before the other is
  // acquired.
  bool flagsInterfere(int lhs, int rhs);
  // Greedy interference-graph coloring + compact renumber. Returns origFlagId
  // -> newFlagId.
  DenseMap<int, int> colorInterferenceGraph();

  DenseMap<Operation *, llvm::SmallVector<Operation *>> relations;
  DenseMap<int, llvm::SmallVector<Operation *>> flagIdToOps;
  DenseMap<Operation *, int> opOrder;
};

} // namespace triton
} // namespace mlir

#endif // TRITON_ADAPTER_DYNAMIC_CV_PIPELINE_SPLIT_DATAFLOW_FLAG_ID_REUSE_H
