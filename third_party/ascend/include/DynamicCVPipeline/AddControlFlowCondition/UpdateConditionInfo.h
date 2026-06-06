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

#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/TransformOps/DialectExtension.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Operation.h"
#include "mlir/Pass/Pass.h"
#include "third_party/ascend/include/DynamicCVPipeline/AddControlFlowCondition.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/SmallVector.h"

namespace mlir {
namespace triton {
enum class VarUpdateType { INC, DEC };

struct OutputGroupInfo {
    SmallVector<Value> outputs;
    SmallVector<Value> inputVars;
};

class UpdateConditionInfoPass : public PassWrapper<UpdateConditionInfoPass, OperationPass<ModuleOp> > {
public:
    UpdateConditionInfoPass() = default;

    void runOnOperation() override;

    void setConditionInfo(ControlFlowConditionInfo *info) { this->info = info; }

private:
    SmallVector<SmallVector<Value> > allocSSBuffer(ModuleOp module);

    int updateIfConds(ModuleOp module, SmallVector<SmallVector<Value> > ssbufferPtrs);

    void updateForIterTimes(ModuleOp module);

    scf::ForOp extendForOpIterationCount(scf::ForOp oldForOp, int ifCount, int requiredBuffers, int x,
                                         IRMapping &mapper, SmallVector<scf::IfOp> &ifOpsInThisFor);

    void replaceForOpCounterInIfOps(SmallVector<scf::IfOp> ifOpsInThisFor, scf::ForOp oldForOp, scf::ForOp newForOp,
                                    IRMapping &mapper);

    Value getVarValue(scf::ForOp forOp, int varIndex);

    void collectDependencyBuffers(scf::ForOp forOp,
                                  DenseMap<int, DenseMap<Value, SmallVector<Value> > > &crossCoreBuffers,
                                  DenseMap<int, DenseMap<Value, SmallVector<Value> > > &intraCoreBuffers);

    DenseMap<int, DenseMap<Value, SmallVector<Value> > >
    extendCrossCoreBuffersWithEquivalentValues(ModuleOp module,
                                               DenseMap<int, DenseMap<Value, SmallVector<Value> > > crossCoreBuffers);

    int buildIdxToVarMap(scf::ForOp forOp,
                         const DenseMap<int, DenseMap<Value, SmallVector<Value> > > &intraCoreBuffers,
                         DenseMap<int, Value> &idxToVar);

    int getInputOutputValues(scf::IfOp ifOp, DenseMap<int, DenseMap<Value, SmallVector<Value> > > crossCoreBuffers,
                              DenseMap<int, DenseMap<Value, SmallVector<Value> > > intraCoreBuffers,
                              SmallVector<int> &crossCoreInputValues, SmallVector<int> &crossCoreOutputValues,
                              SmallVector<int> &intraCoreInputValues, SmallVector<int> &intraCoreOutputValues);

    int buildOutputGroups(SmallVector<int> &intraCoreOutputValues,
                          DenseMap<int, DenseMap<Value, SmallVector<Value> > > &intraCoreBuffers,
                          DenseMap<int, Value> &idxToVar, SmallVector<OutputGroupInfo> &outputGroups);

    void collectIntraCoreInputConditions(OpBuilder &builder, Location loc, SmallVector<int> &intraCoreInputValues,
                                         DenseMap<int, Value> &idxToVar, SmallVector<Value> &conditions,
                                         DenseSet<Value> &usedVarsSet,
                                         DenseMap<Value, VarUpdateType> &varUpdateTypes);

    int collectIntraCoreOutputConditions(OpBuilder &builder, Location loc,
                                         DenseMap<int, DenseMap<Value, SmallVector<Value> > > &intraCoreBuffers,
                                         SmallVector<int> &intraCoreOutputValues, DenseMap<int, Value> &idxToVar,
                                         SmallVector<Value> &conditions, DenseSet<Value> &usedVarsSet,
                                         DenseMap<Value, VarUpdateType> &varUpdateTypes);

    SmallVector<Type> buildNewIfResultTypes(scf::IfOp oldIfOp, bool hasCounter, Value counter);

    void collectYieldOperands(Block &block, Operation *&yieldOp, SmallVector<Value> &yieldOperands);

    void populateNewThenBlock(scf::IfOp newIfOp, Block &oldThenBlock, Operation *oldThenYieldOp,
                              ArrayRef<Value> oldYieldOperands, DenseMap<Value, VarUpdateType> &varUpdateTypes,
                              bool hasCounter, Value counter, Value step);

    void populateNewElseBlock(scf::IfOp newIfOp, scf::IfOp oldIfOp, bool needsYield, bool oldHasElse,
                              bool hasCounter, Value counter);

    scf::IfOp createNewIfOpWithBlocks(scf::IfOp oldIfOp, Value combinedCond,
                                      DenseMap<Value, VarUpdateType> &varUpdateTypes, bool hasCounter, Value counter,
                                      Value step);

    int setIntraCoreCondition(ModuleOp module, scf::IfOp ifOp,
                              DenseMap<int, DenseMap<Value, SmallVector<Value> > > &intraCoreBuffers,
                              SmallVector<int> &intraCoreInputIndices, SmallVector<int> &intraCoreOutputIndices,
                              DenseMap<int, Value> &idxToVar, DenseMap<Value, VarUpdateType> &varUpdateTypes,
                              Value &intraCoreCond);

    void updateControlVarToLatestValue(scf::IfOp newIfOp, scf::IfOp oldIfOp, bool hasCounter, Value counter);

    int updateForOpYield(scf::ForOp forOp);

    int combineConditions(ModuleOp module, Value crossCoreCond, Value intraCoreCond, scf::IfOp ifOp, scf::ForOp forOp,
                          size_t &usedCounterNum, DenseMap<Value, VarUpdateType> &varUpdateTypes);

    int setCrossCoreCondition(SmallVector<int> crossCoreInputValues, SmallVector<int> crossCoreOutputValues,
                               DenseMap<int, DenseMap<Value, SmallVector<Value> > > &crossCoreBuffers, scf::IfOp ifOp,
                               SmallVector<SmallVector<Value> > ssbufferPtrs, Value &crossCoreCond);

    // Helper function to get pointer based on core type
    Value getSSBufferPtr(bool isAIC, int groupIdx, int ptrSetIdx,
                         DenseMap<int, Value> &precomputedPtrs,
                         SmallVector<SmallVector<Value> > ssbufferPtrs);

    // Compute pointers for VECTOR core SSBuffer
    DenseMap<int, Value> computeVectorSSBufferPtrs(OpBuilder &builder, Location loc,
                                                   Operation *scopeOp,
                                                   SmallVector<int> crossCoreInputValues,
                                                   SmallVector<int> crossCoreOutputValues);

    // Part 2: Add cross-core conditions
    Value addCrossCoreConditions(OpBuilder &builder, Location loc,
                                 SmallVector<int> crossCoreInputValues, SmallVector<int> crossCoreOutputValues,
                                 DenseMap<int, DenseMap<Value, SmallVector<Value> > > &crossCoreBuffers,
                                 bool isAIC, Value zeroConst,
                                 DenseMap<int, Value> &precomputedPtrs,
                                 SmallVector<SmallVector<Value> > ssbufferPtrs);

    // Part 3: Update control variables in then block
    void updateCrossCoreControlVars(OpBuilder &builder, Location loc,
                                    scf::IfOp ifOp, SmallVector<int> crossCoreInputValues,
                                    SmallVector<int> crossCoreOutputValues,
                                    bool isAIC, Value oneConst,
                                    DenseMap<int, Value> &precomputedPtrs,
                                    SmallVector<SmallVector<Value> > ssbufferPtrs);

    DenseMap<Value, Value> controlVarToLatestValue;
    SmallVector<Value> currentUsedVars;
    ControlFlowConditionInfo *info = nullptr;
};

std::unique_ptr<OperationPass<ModuleOp> > createUpdateConditionInfoPass();
} // namespace triton
} // namespace mlir
#endif // TRITON_ADAPTER_UPDATE_CONDITION_INFO_H
