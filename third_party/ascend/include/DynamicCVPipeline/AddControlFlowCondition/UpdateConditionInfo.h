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

#ifndef TRITON_ADAPTER_UPDATE_CONDITION_INFO_H
#define TRITON_ADAPTER_UPDATE_CONDITION_INFO_H

#include <optional>

#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/SmallVector.h"

#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/TransformOps/DialectExtension.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Operation.h"
#include "mlir/Pass/Pass.h"

#include "third_party/ascend/include/DynamicCVPipeline/AddControlFlowCondition.h"

namespace mlir {
namespace triton {
enum class VarUpdateType { INC, DEC };

struct OutputGroupInfo {
  SmallVector<Operation *> outputs;
  SmallVector<Value> inputVars;
};

class UpdateConditionInfoPass
    : public PassWrapper<UpdateConditionInfoPass, OperationPass<ModuleOp>> {
public:
  UpdateConditionInfoPass() = default;

  void runOnOperation() override;

  void setConditionInfo(ControlFlowConditionInfo *info) { this->info = info; }

private:
  SmallVector<SmallVector<Value>> allocSSBuffer(ModuleOp module);

  int updateIfConds(ModuleOp module,
                    SmallVector<SmallVector<Value>> ssbufferPtrs);

  // Collect ssbuffer ifOps: for walks body; while-do walks after-region only.
  int collectSSBufferIfOps(Operation *loopOp, SmallVector<scf::IfOp> &ifOps);

  // Validate blockCounters for for; while skips (counters are for-only).
  int validateBlockCounters(Operation *loopOp, size_t ifOpCount);

  void updateForIterTimes(ModuleOp module);

  scf::ForOp extendForOpIterationCount(scf::ForOp oldForOp, int ifCount,
                                       int requiredBuffers, int x,
                                       IRMapping &mapper,
                                       SmallVector<scf::IfOp> &ifOpsInThisFor);

  void replaceForOpCounterInIfOps(SmallVector<scf::IfOp> ifOpsInThisFor,
                                  scf::ForOp oldForOp, scf::ForOp newForOp,
                                  IRMapping &mapper);

  Value getVarValue(Operation *loopOp, int varIndex);

  void collectDependencyBuffers(
      ModuleOp module, SmallVector<Operation *> &mainLoopOps,
      DenseMap<int, DenseMap<Operation *, SmallVector<Operation *>>>
          &crossCoreBuffers,
      DenseMap<Operation *,
               DenseMap<int, DenseMap<Operation *, SmallVector<Operation *>>>>
          &intraCoreBuffersMap);

  int buildIdxToVarMap(
      Operation *loopOp,
      const DenseMap<int, DenseMap<Operation *, SmallVector<Operation *>>>
          &intraCoreBuffers,
      DenseMap<int, Value> &idxToVar);

  int getInputOutputValues(
      scf::IfOp ifOp,
      DenseMap<int, DenseMap<Operation *, SmallVector<Operation *>>>
          crossCoreBuffers,
      DenseMap<int, DenseMap<Operation *, SmallVector<Operation *>>>
          intraCoreBuffers,
      SmallVector<int> &crossCoreInputValues,
      SmallVector<int> &crossCoreOutputValues,
      SmallVector<int> &intraCoreInputValues,
      SmallVector<int> &intraCoreOutputValues);

  int buildOutputGroups(
      SmallVector<int> &intraCoreOutputValues,
      DenseMap<int, DenseMap<Operation *, SmallVector<Operation *>>>
          &intraCoreBuffers,
      DenseMap<int, Value> &idxToVar,
      SmallVector<OutputGroupInfo> &outputGroups);

  void collectIntraCoreInputConditions(
      OpBuilder &builder, Location loc, SmallVector<int> &intraCoreInputValues,
      DenseMap<int, Value> &idxToVar, SmallVector<Value> &conditions,
      DenseSet<Value> &usedVarsSet,
      DenseMap<Value, VarUpdateType> &varUpdateTypes);

  int collectIntraCoreOutputConditions(
      OpBuilder &builder, Location loc,
      DenseMap<int, DenseMap<Operation *, SmallVector<Operation *>>>
          &intraCoreBuffers,
      SmallVector<int> &intraCoreOutputValues, DenseMap<int, Value> &idxToVar,
      SmallVector<Value> &conditions, DenseSet<Value> &usedVarsSet,
      DenseMap<Value, VarUpdateType> &varUpdateTypes);

  // Build the ifOp variable mapping for the tensor iter_args
  int buildTensorIterArgIfOpVarMap(Operation *loopOp);

  // Collect the consumption conditions of the tensor iter_args consumer
  void collectTensorIterArgInputConditions(
      OpBuilder &builder, Location loc, scf::IfOp ifOp,
      SmallVector<Value> &conditions, DenseSet<Value> &usedVarsSet,
      DenseMap<Value, VarUpdateType> &varUpdateTypes);

  // Collect tensor iter_args producer conditions
  void collectTensorIterArgOutputConditions(
      OpBuilder &builder, Location loc, scf::IfOp ifOp,
      SmallVector<Value> &conditions, DenseSet<Value> &usedVarsSet,
      DenseMap<Value, VarUpdateType> &varUpdateTypes);

  SmallVector<Type> buildNewIfResultTypes(scf::IfOp oldIfOp, bool hasCounter,
                                          Value counter);

  void collectYieldOperands(Block &block, Operation *&yieldOp,
                            SmallVector<Value> &yieldOperands);

  void populateNewThenBlock(scf::IfOp newIfOp, Block &oldThenBlock,
                            Operation *oldThenYieldOp,
                            ArrayRef<Value> oldYieldOperands,
                            DenseMap<Value, VarUpdateType> &varUpdateTypes,
                            bool hasCounter, Value counter, Value step);

  void populateNewElseBlock(scf::IfOp newIfOp, scf::IfOp oldIfOp,
                            bool oldHasElse, bool hasCounter, Value counter);

  scf::IfOp
  createNewIfOpWithBlocks(scf::IfOp oldIfOp, Value combinedCond,
                          DenseMap<Value, VarUpdateType> &varUpdateTypes,
                          bool hasCounter, Value counter, Value step);

  int setIntraCoreCondition(
      ModuleOp module, scf::IfOp ifOp,
      DenseMap<int, DenseMap<Operation *, SmallVector<Operation *>>>
          &intraCoreBuffers,
      SmallVector<int> &intraCoreInputIndices,
      SmallVector<int> &intraCoreOutputIndices, DenseMap<int, Value> &idxToVar,
      DenseMap<Value, VarUpdateType> &varUpdateTypes, Value &intraCoreCond);

  void updateControlVarToLatestValue(scf::IfOp newIfOp, scf::IfOp oldIfOp,
                                     bool hasCounter, Value counter);

  int updateForOpYield(scf::ForOp forOp);

  // Update after-region yield for while when control vars were rewritten.
  int updateWhileOpYield(scf::WhileOp whileOp);

  // Dispatch yield update for scf.for / scf.while main_loop.
  int updateLoopYield(Operation *loopOp);

  // loopOp is scf.for or scf.while main_loop.
  int combineConditions(ModuleOp module, Value crossCoreCond,
                        Value intraCoreCond, Value flowOptCond, scf::IfOp ifOp,
                        Operation *loopOp, size_t &usedCounterNum,
                        DenseMap<Value, VarUpdateType> &varUpdateTypes);

  int setCrossCoreCondition(
      SmallVector<int> crossCoreInputValues,
      SmallVector<int> crossCoreOutputValues,
      DenseMap<int, DenseMap<Operation *, SmallVector<Operation *>>>
          &crossCoreBuffers,
      scf::IfOp ifOp, SmallVector<SmallVector<Value>> ssbufferPtrs,
      Value &crossCoreCond);

  // Set the FlowOpt extra condition for the third if block in the DAG.
  // Needs lb/ub/step from scf.for; scf.while leaves flowOptCond null.
  int setFlowOptCondition(scf::IfOp currentIfOp, Operation *loopOp,
                          Value &flowOptCond);

  // Update DAG nodes after ifOp replacement
  void updateDAGAfterIfOpReplacement(scf::IfOp oldIfOp, scf::IfOp newIfOp);

  // Helper function to get pointer based on core type
  Value getSSBufferMemref(bool isAIC, int groupIdx, int ptrSetIdx,
                          DenseMap<int, Value> &precomputedPtrs,
                          ArrayRef<SmallVector<Value>> ssbufferPtrs);

  // Compute pointers for VECTOR core SSBuffer
  std::optional<DenseMap<int, Value>>
  computeVectorSSBufferMemrefs(OpBuilder &builder, Location loc,
                               Operation *scopeOp,
                               SmallVector<int> crossCoreInputValues,
                               SmallVector<int> crossCoreOutputValues);

  // Part 2: Add cross-core conditions
  Value addCrossCoreConditions(
      OpBuilder &builder, Location loc, SmallVector<int> crossCoreInputValues,
      SmallVector<int> crossCoreOutputValues,
      DenseMap<int, DenseMap<Operation *, SmallVector<Operation *>>>
          &crossCoreBuffers,
      bool isAIC, Value zeroConst, DenseMap<int, Value> &precomputedPtrs,
      SmallVector<SmallVector<Value>> ssbufferPtrs);

  // Part 3: Update control variables in then block
  void updateCrossCoreControlVars(OpBuilder &builder, Location loc,
                                  scf::IfOp ifOp,
                                  SmallVector<int> crossCoreInputValues,
                                  SmallVector<int> crossCoreOutputValues,
                                  bool isAIC, Value oneConst,
                                  DenseMap<int, Value> &precomputedPtrs,
                                  SmallVector<SmallVector<Value>> ssbufferPtrs);

  DenseMap<Value, Value> controlVarToLatestValue;
  SmallVector<Value> currentUsedVars;
  ControlFlowConditionInfo *info = nullptr;
  // Record each ifOp as the variables that need to be controlled when it acts
  // as a consumer or a producer
  llvm::DenseMap<scf::IfOp, TensorIterArgIfOpVars> tensorIterArgIfOpVars;

  template <typename FuncTy>
  auto createSsbufLoads(OpBuilder &builder, Location loc, bool isAIC,
                        int groupIdx,
                        DenseMap<int, Value> &vectorSSBufferMemrefs,
                        llvm::ArrayRef<SmallVector<Value>> ssbufferMemrefs,
                        FuncTy &&pred);
};

std::unique_ptr<OperationPass<ModuleOp>> createUpdateConditionInfoPass();
} // namespace triton
} // namespace mlir
#endif // TRITON_ADAPTER_UPDATE_CONDITION_INFO_H
