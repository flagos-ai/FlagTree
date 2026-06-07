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

#ifndef TRITON_ADAPTER_UPDATE_LOOP_ITER_TIMES_H
#define TRITON_ADAPTER_UPDATE_LOOP_ITER_TIMES_H

#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallVector.h"

#include "ascend/include/DynamicCVPipeline/AddControlFlowCondition.h"

namespace mlir {
namespace triton {

  // Loop iteration times info
struct IterationTimesInfo {
  int ifCount = 0;           // Number of if operations
  int requiredBuffers = 1;   // Number of required buffers
  int x = 1;                 // Number of producer buffers
  SmallVector<scf::IfOp> ifOpsInThisFor; // If operations in this loop
};

class UpdateLoopIterTimesPass
    : public PassWrapper<UpdateLoopIterTimesPass, OperationPass<ModuleOp>> {
public:
  UpdateLoopIterTimesPass() = default;
  void runOnOperation() override;
  void setConditionInfo(ControlFlowConditionInfo *info) { this->info = info; }

private:
  ControlFlowConditionInfo *info = nullptr;

  int GetMainLoopIdToLoopOpMap(ModuleOp module,
                                DenseMap<int, SmallVector<Operation *>> &cmap,
                                DenseMap<int, SmallVector<Operation *>> &vmap);

  int ComputeMainLoopTimes(DenseMap<int, SmallVector<Operation *>> &loopMap,
                            DenseMap<Operation *, IterationTimesInfo> &infoMap);

  int UpdateForLoopIteration(DenseMap<int, SmallVector<Operation *>> &cmap,
                              DenseMap<int, SmallVector<Operation *>> &vmap,
                              DenseMap<Operation *, IterationTimesInfo> &infoMap);

  int collectForOpsAndUpdateMax(DenseMap<int, SmallVector<Operation *>> &map,
                                  int id,
                                  SmallVector<Operation *> &allForOps,
                                  int &maxIfCount,
                                  int &maxRequiredBuffers,
                                  int &maxX,
                                  DenseMap<Operation *, IterationTimesInfo> &infoMap);

  int replaceForOpCounterInIfOps();

  // Calculate factor = requiredBuffers / x
  std::pair<int, int> calculateFactor(scf::ForOp forOp);

  // Extend for loop iteration count
  scf::ForOp extendForOpIterationCount(scf::ForOp oldForOp, int ifCount,
                                       int requiredBuffers, int x,
                                       IRMapping &mapper,
                                       SmallVector<scf::IfOp> &ifOpsInThisFor);

  Value computeNewLoopUpperBound(OpBuilder &builder, Location loc, scf::ForOp forOp,
                                int ifCount, int requiredBuffers, int x);

  scf::ForOp cloneForOpWithNewUpperBound(OpBuilder &builder, Location loc,
                                          scf::ForOp oldForOp, Value newUpperBound,
                                          IRMapping &mapper);

  int updateCntArgsAfterClone(scf::ForOp oldForOp, IRMapping &mapper,
                               SmallVector<scf::IfOp> &ifOpsInThisFor);
};

std::unique_ptr<OperationPass<ModuleOp>> createUpdateLoopIterTimesPass();

} // namespace triton
} // namespace mlir

#endif // TRITON_ADAPTER_UPDATE_LOOP_ITER_TIMES_H