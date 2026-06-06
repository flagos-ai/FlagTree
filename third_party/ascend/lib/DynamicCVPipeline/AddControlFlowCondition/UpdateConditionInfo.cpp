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
#include "third_party/ascend/include/DynamicCVPipeline/AddControlFlowCondition/UpdateConditionInfo.h"
#include "ascend/include/DynamicCVPipeline/AddControlFlowCondition.h"
#include "bishengir/Dialect/HIVM/IR/HIVM.h"
#include "bishengir/Dialect/HIVM/IR/HIVMImpl.h"
#include "bishengir/Dialect/HIVM/IR/HIVMInterfaces.h"
#include "bishengir/Dialect/HIVM/Transforms/Passes.h"
#include "bishengir/Dialect/HIVM/Utils/Utils.h"
#include "bishengir/Dialect/Scope/IR/Scope.h"
#include "llvm/ADT/APFloat.h"
#include "llvm/Support/Debug.h"

static constexpr const char *DEBUG_TYPE = "UpdateConditionInfoPass";
static constexpr const char *SSBUFFER_Main_LOOP = "ssbuffer.main_loop";
static constexpr const char *SSBUFFER_IF = "ssbuffer.if";
static constexpr int SSBUF_ADDR_SPACE = 11;
static constexpr int ADDR_INT_TYPE = 64;
static constexpr int CONST_INT_TYPE = 32;
static constexpr int VECTOR_SSBUF_OFFSET = 1024;
static constexpr int VALUE_SSBUF_OFFSET = 4;
static constexpr int UPDATE_CONDITION_INFO_SUCCESS = 0;
static constexpr int UPDATE_CONDITION_INFO_FAILED = -1;

#define DBGS() (llvm::dbgs() << '[' << DEBUG_TYPE << "] ")
#define LDBG(X) LLVM_DEBUG(DBGS() << (X) << "\n")
using namespace mlir;
using namespace triton;
using namespace hivm;

// Allocate the SSBuffer pointer
SmallVector<SmallVector<Value>> UpdateConditionInfoPass::allocSSBuffer(ModuleOp module)
{
  OpBuilder builder(module.getContext());
  auto i64Type = builder.getIntegerType(ADDR_INT_TYPE);
  auto i32Type = builder.getIntegerType(CONST_INT_TYPE);
  auto ptrType = mlir::LLVM::LLVMPointerType::get(builder.getContext(), SSBUF_ADDR_SPACE);

  // alloc 2 group of ssbuffer pointers:
  // Core Vector 0: allocate ssbuffer address: 0, 4, 8, ...
  // Core Vector 1: allocate ssbuffer address: 1024, 1028, 1032, ...
  SmallVector<SmallVector<Value>> ssbufferPtrs;
  SmallVector<Value> ssbufferVec0Ptrs;
  SmallVector<Value> ssbufferVec1Ptrs;
  int numBuffers = info->crossCoreDependentMap.size();
  if (numBuffers == 0) {
      LDBG("crossCoreDependentMap is empty!");
      return ssbufferPtrs;
  }

  module->walk([&](Operation *op) {
    if (auto scopeOp = dyn_cast<scope::ScopeOp>(op)) {
      builder.setInsertionPoint(scopeOp);
      auto zeroConst =
          builder.create<mlir::LLVM::ConstantOp>(scopeOp->getLoc(), i32Type, builder.getIntegerAttr(i32Type, 0));

      for (int i = 0; i < numBuffers; i++) {
        auto addr0Attr = builder.getIntegerAttr(i64Type, i * VALUE_SSBUF_OFFSET);
        auto addr1Attr = builder.getIntegerAttr(i64Type, VECTOR_SSBUF_OFFSET + i * VALUE_SSBUF_OFFSET);

        auto addr0Const = builder.create<mlir::LLVM::ConstantOp>(scopeOp->getLoc(), i64Type, addr0Attr);
        auto addr1Const = builder.create<mlir::LLVM::ConstantOp>(scopeOp->getLoc(), i64Type, addr1Attr);

        auto ptr0 = builder.create<mlir::LLVM::IntToPtrOp>(scopeOp->getLoc(), ptrType, addr0Const.getResult());
        auto ptr1 = builder.create<mlir::LLVM::IntToPtrOp>(scopeOp->getLoc(), ptrType, addr1Const.getResult());

        builder.create<LLVM::StoreOp>(scopeOp->getLoc(), zeroConst, ptr0);
        builder.create<LLVM::StoreOp>(scopeOp->getLoc(), zeroConst, ptr1);

        ssbufferVec0Ptrs.push_back(ptr0.getResult());
        ssbufferVec1Ptrs.push_back(ptr1.getResult());
      }
      return mlir::WalkResult::interrupt();
    }
    return mlir::WalkResult::advance();
  });

  ssbufferPtrs.push_back(ssbufferVec0Ptrs);
  ssbufferPtrs.push_back(ssbufferVec1Ptrs);
  return ssbufferPtrs;
}

// Collect dependency buffer
void UpdateConditionInfoPass::collectDependencyBuffers(
    scf::ForOp forOp, DenseMap<int, DenseMap<Value, SmallVector<Value>>> &crossCoreBuffers,
    DenseMap<int, DenseMap<Value, SmallVector<Value>>> &intraCoreBuffers)
{
  int crossCoreIdx = 0;
  for (auto &entry : info->crossCoreDependentMap) {
    crossCoreBuffers[crossCoreIdx][entry.first] = entry.second;
    crossCoreIdx++;
  }

  if (info->intraCoreDependentMap.count(forOp)) {
    auto &forOpDeps = info->intraCoreDependentMap[forOp];
    int intraCoreIdx = 0;
    for (auto &entry : forOpDeps) {
      intraCoreBuffers[intraCoreIdx][entry.first] = entry.second;
      intraCoreIdx++;
    }
  }
}

// Helper: Find the tcb group id that contains value v
// Returns the group id if found, -1 otherwise
static int findTcbGroupId(Value v, DenseMap<int, SmallVector<Value>> &tightlyCoupledBufferGroups)
{
  for (auto &tcbEntry : tightlyCoupledBufferGroups) {
    if (llvm::is_contained(tcbEntry.second, v)) {
      return tcbEntry.first;
    }
  }
  return UPDATE_CONDITION_INFO_FAILED;
}

// Helper: Add all equivalent values from tcbValues to values (excluding v itself)
int addEquivalentValues(Value v, SmallVector<Value> &tcbValues, SmallVector<Value> &values)
{
  int ret = -1;
  for (Value equivValue : tcbValues) {
    if (equivValue != v && !llvm::is_contained(values, equivValue)) {
      ret = 0;
      values.push_back(equivValue);
    }
  }
  return ret;
}

// Buffers in cube/vector scope are different value
// scope {
//   %alloc_5 = memref.alloc()
//   annotation.mark %alloc_5 {hivm.tightly_coupled_buffer = #hivm.tightly_coupled_buffer<1>}
//   fixpipe ins() outs(alloc_5)
// } {CUBE}
// scope {
//   %alloc_6 = memref.alloc()
//   bufferization.to_tensor %alloc_6
//   annotation.mark %alloc_6 {hivm.tightly_coupled_buffer = #hivm.tightly_coupled_buffer<1>}
// } {VECTOR}
// % alloc_5 and alloc_6 have the same tightly_coupled_buffer id indicates they are the same buffer
// crossCoreBuffers only include producer buffers in one scope: {cosumer: {%alloc_6}, ...}
// This function is to extend crossCoreBuffers to include producer buffers in another scope: {cosumer: {%alloc_6, %alloc_5}, ...}
DenseMap<int, DenseMap<Value, SmallVector<Value>>> UpdateConditionInfoPass::extendCrossCoreBuffersWithEquivalentValues(
    ModuleOp module, DenseMap<int, DenseMap<Value, SmallVector<Value>>> crossCoreBuffers)
{
  // Error map to return when error occurs
  DenseMap<int, DenseMap<Value, SmallVector<Value>>> errorMap;
  errorMap[-1] = DenseMap<Value, SmallVector<Value>>();

  // copy from crossCoreBuffers
  DenseMap<int, DenseMap<Value, SmallVector<Value>>> extendedCrossCoreBuffers;
  for (auto &entry : crossCoreBuffers) {
    int groupIdx = entry.first;
    for (auto &entry2 : entry.second) {
      extendedCrossCoreBuffers[groupIdx][entry2.first] = entry2.second;
    }
  }

  // Get the buffers have the same tightly_coupled_buffer id
  int ret = 0;
  DenseMap<int, SmallVector<Value>> tightlyCoupledBufferGroups;
  WalkResult walkResult = module.walk([&](Operation *op) -> WalkResult {
    if (isa<annotation::MarkOp>(op)) {
      if (auto tcbAttr = op->getAttrOfType<hivm::HIVMTightlyCoupledBufferAttr>("hivm.tightly_coupled_buffer")) {
        auto id = tcbAttr.getId();
        if (id.has_value()) {
          int tcb = id.value();
          Value markedValue = op->getOperand(0);
          tightlyCoupledBufferGroups[tcb].push_back(markedValue);
        } else {
          ret = -1;
          LDBG("hivm.tightly_coupled_buffer Attribute has no id!");
          return WalkResult::interrupt();
        }
      }
    }
    return WalkResult::advance();
  });
  if (ret == -1) {
    return errorMap;
  }

  // Extend crossCoreBuffers to include producer buffers in another scope
  for (auto &entry : extendedCrossCoreBuffers) {
    int groupIdx = entry.first;
    for (auto &deps : entry.second) {
      SmallVector<Value> &producers = deps.second;
      for (Value buffer : producers) {
        int tcbGroupId = findTcbGroupId(buffer, tightlyCoupledBufferGroups);
        if (tcbGroupId == -1) {
          LDBG("Can not find tightly_coupled_buffer id");
          return errorMap;
        }
        if (addEquivalentValues(buffer, tightlyCoupledBufferGroups[tcbGroupId], producers) == -1) {
          LDBG("Can not find the crossCore Buffer from another scope");
          return errorMap;
        }
      }
    }
  }
  return extendedCrossCoreBuffers;
}

int UpdateConditionInfoPass::buildIdxToVarMap(scf::ForOp forOp,
                                              const DenseMap<int, DenseMap<Value, SmallVector<Value> > > &
                                              intraCoreBuffers,
                                              DenseMap<int, Value> &idxToVar)
{
  int varIdx = 0;
  int iterArgNum = static_cast<int>(forOp.getNumRegionIterArgs());

  const auto &innerDepIndices = info->innerDepConds[forOp];
  if (innerDepIndices.size() < intraCoreBuffers.size()) {
    LLVM_DEBUG(llvm::dbgs() << "Not enough inner dependency condition indices: assigned "
               << innerDepIndices.size() << ", expected " << intraCoreBuffers.size() << "\n");
    return UPDATE_CONDITION_INFO_FAILED;
  }

  for (const auto &entry : intraCoreBuffers) {
    int idx = entry.first;

    int argIdx = innerDepIndices[varIdx];
    if (argIdx < 0 || argIdx >= iterArgNum) {
      LLVM_DEBUG(llvm::dbgs() << "Invalid inner dependency arg index: " << argIdx
                 << ", iter args " << iterArgNum << "\n");
      return UPDATE_CONDITION_INFO_FAILED;
    }

    idxToVar[idx] = forOp.getRegionIterArgs()[argIdx];
    LLVM_DEBUG(llvm::dbgs() << "Assign intraCore buffer group " << idx << " to iter arg index " << argIdx << "\n");
    varIdx++;
  }

  LLVM_DEBUG(llvm::dbgs() << "Assigned " << idxToVar.size() << " intraCore condition variables.\n");
  return UPDATE_CONDITION_INFO_SUCCESS;
}

// getInputOutputValues - Analyze the input/output buffer groups used in a single ifOp
// This function traverses all operations within ifOp, identifies FixpipeOp, CopyOp and
// MaterializeInDestinationOp, and extracts cross-core and intra-core buffer group indices
// from their operands and yield values.

// Data structure description
//   - crossCoreBuffers: {groupIdx -> (consumer -> [producers])}
//   - intraCoreBuffers: {groupIdx -> (consumer -> [producers])}
//   - crossCoreInputValues: list of cross-core group indices read by this ifOp
//   - crossCoreOutputValues: list of cross-core group indices written by this ifOp
//   - intraCoreInputValues: list of intra-core group indices read by this ifOp
//   - intraCoreOutputValues: list of intra-core group indices written by this ifOp

// Processing flow
//   1. Build reverse mapping table from Value to groupIdx
//   2. Traverse Fixpipe/Copy/MaterializeInDestination ops in ifOp, collect input/output groups
//   3. Traverse ifOp.thenYield() operands, collect yield output groups
//   4. Deduplicate and output four groups of index values
int UpdateConditionInfoPass::getInputOutputValues(
    scf::IfOp ifOp, DenseMap<int, DenseMap<Value, SmallVector<Value>>> crossCoreBuffers,
    DenseMap<int, DenseMap<Value, SmallVector<Value>>> intraCoreBuffers, SmallVector<int> &crossCoreInputValues,
    SmallVector<int> &crossCoreOutputValues, SmallVector<int> &intraCoreInputValues,
    SmallVector<int> &intraCoreOutputValues)
{
  DenseSet<int> crossCoreInputSet;
  DenseSet<int> crossCoreOutputSet;
  DenseSet<int> intraCoreInputSet;
  DenseSet<int> intraCoreOutputSet;

  // Get the crossCoreBufferToGroup: {bufferValue, groupIdx}
  DenseMap<Value, int> crossCoreBufferToGroup;
  DenseMap<Value, int> intraCoreInputToGroup;
  DenseMap<Value, SmallVector<int>> intraCoreOutputToGroups;

  for (auto &entry : crossCoreBuffers) {
    int groupIdx = entry.first;
    for (auto &entry2 : entry.second) {
      for (Value v : entry2.second) {
        crossCoreBufferToGroup[v] = groupIdx;
      }
    }
  }

  // Get the intraCoreInputToGroup: {bufferValue, groupIdx}
  for (auto &entry : intraCoreBuffers) {
    int groupIdx = entry.first;
    for (auto &entry2 : entry.second) {
      Value input = entry2.first;
      SmallVector<Value> outputs = entry2.second;
      intraCoreInputToGroup[input] = groupIdx;
      for (Value output : outputs) {
        intraCoreOutputToGroups[output].push_back(groupIdx);
      }
    }
  }

  // collect input/output groups
  ifOp.walk([&](Operation *op) {
    if (op == ifOp)
      return WalkResult::advance();

    bool isFixpipeOrCopy = dyn_cast<hivm::FixpipeOp>(op) || dyn_cast<hivm::CopyOp>(op);
    bool isBufferizationWrite = dyn_cast<bufferization::MaterializeInDestinationOp>(op);
    // Op is FixpipeOp/CopyOp/BufferizationWriteOp
    // they have two operand, operand 0(ins) is input, operand 1(outs) is output
    if (isFixpipeOrCopy || isBufferizationWrite) {
      Value insVal = op->getOperands()[0];
      if (crossCoreBufferToGroup.count(insVal)) {
        crossCoreInputSet.insert(crossCoreBufferToGroup[insVal]);
      } else if (intraCoreInputToGroup.count(insVal)) {
        intraCoreInputSet.insert(intraCoreInputToGroup[insVal]);
      }

      Value outsVal = op->getOperands()[1];
      if (crossCoreBufferToGroup.count(outsVal)) {
        crossCoreOutputSet.insert(crossCoreBufferToGroup[outsVal]);
      } else if (intraCoreOutputToGroups.count(outsVal)) {
        for (int idx : intraCoreOutputToGroups[outsVal]) {
          intraCoreOutputSet.insert(idx);
        }
      }
      return WalkResult::advance();
    } else {
      // Op is normal, operand is input
      for (Value operand : op->getOperands()) {
        if (crossCoreBufferToGroup.count(operand))
          crossCoreInputSet.insert(crossCoreBufferToGroup[operand]);
        if (intraCoreInputToGroup.count(operand))
          intraCoreInputSet.insert(intraCoreInputToGroup[operand]);
      }
    }
    return WalkResult::advance();
  });

  // operands in yield op are output
  scf::YieldOp thenYield = ifOp.thenYield();
  for (Value yieldVal : thenYield.getOperands()) {
    if (crossCoreBufferToGroup.count(yieldVal))
      crossCoreOutputSet.insert(crossCoreBufferToGroup[yieldVal]);
    if (intraCoreOutputToGroups.count(yieldVal)) {
      for (int idx : intraCoreOutputToGroups[yieldVal]) {
        intraCoreOutputSet.insert(idx);
      }
    }
  }

  crossCoreInputValues.assign(crossCoreInputSet.begin(), crossCoreInputSet.end());
  crossCoreOutputValues.assign(crossCoreOutputSet.begin(), crossCoreOutputSet.end());
  intraCoreInputValues.assign(intraCoreInputSet.begin(), intraCoreInputSet.end());
  intraCoreOutputValues.assign(intraCoreOutputSet.begin(), intraCoreOutputSet.end());
  LDBG("==== Cross Core & Intra Core Values ====");
  LLVM_DEBUG(llvm::dbgs() << "crossCoreInputValues: ");
  for (int val : crossCoreInputValues) {
      LLVM_DEBUG(llvm::dbgs() << val << " ");
  }
  LLVM_DEBUG(llvm::dbgs() << "\n");

  LLVM_DEBUG(llvm::dbgs() << "crossCoreOutputValues: ");
  for (int val : crossCoreOutputValues) {
      LLVM_DEBUG(llvm::dbgs() << val << " ");
  }
  LLVM_DEBUG(llvm::dbgs() << "\n");

  LLVM_DEBUG(llvm::dbgs() << "intraCoreInputValues: ");
  for (int val : intraCoreInputValues) {
      LLVM_DEBUG(llvm::dbgs() << val << " ");
  }
  LLVM_DEBUG(llvm::dbgs() << "\n");

  LLVM_DEBUG(llvm::dbgs() << "intraCoreOutputValues: ");
  for (int val : intraCoreOutputValues) {
      LLVM_DEBUG(llvm::dbgs() << val << " ");
  }
  LLVM_DEBUG(llvm::dbgs() << "\n");

  return UPDATE_CONDITION_INFO_SUCCESS;
}

Value UpdateConditionInfoPass::getVarValue(scf::ForOp forOp, int varIndex)
{
  if (!info->innerDepConds.count(forOp))
    return Value();
  SmallVector<int> &innerDepIndices = info->innerDepConds[forOp];
  if (varIndex < (int)innerDepIndices.size()) {
    int argIdx = innerDepIndices[varIndex];
    return forOp.getRegionIterArgs()[argIdx];
  }
  return Value();
}

// Build the information of the producer group.
int UpdateConditionInfoPass::buildOutputGroups(
    SmallVector<int> &intraCoreOutputValues, DenseMap<int, DenseMap<Value, SmallVector<Value>>> &intraCoreBuffers,
    DenseMap<int, Value> &idxToVar, SmallVector<OutputGroupInfo> &outputGroups)
{
  outputGroups.clear();

  for (int idx : intraCoreOutputValues) {
    auto bufferIt = intraCoreBuffers.find(idx);
    if (bufferIt == intraCoreBuffers.end()) {
      LLVM_DEBUG(llvm::dbgs() << "Failed to build output groups: no buffer entry for intraCore output group "
                               << idx << ".\n");
      return UPDATE_CONDITION_INFO_FAILED;
    }

    auto varIt = idxToVar.find(idx);
    if (varIt == idxToVar.end()) {
      LLVM_DEBUG(llvm::dbgs() << "Failed to build output groups: no control variable for intraCore output group "
                               << idx << ".\n");
      return UPDATE_CONDITION_INFO_FAILED;
    }
    Value var = varIt->second;

    for (auto &entry : bufferIt->second) {
      SmallVector<Value> &outputs = entry.second;
      if (outputs.empty())
        continue;

      bool flag = true;
      for (auto &outputGroup : outputGroups) {
        if (outputGroup.outputs == outputs) {
          outputGroup.inputVars.push_back(var);
          flag = false;
          break;
        }
      }
      if (flag) {
        OutputGroupInfo groupInfo;
        groupInfo.outputs = outputs;
        groupInfo.inputVars.push_back(var);
        outputGroups.push_back(groupInfo);
      }
    }
  }

  LLVM_DEBUG(llvm::dbgs() << "Built " << outputGroups.size() << " intraCore output groups.\n");
  for (size_t i = 0; i < outputGroups.size(); ++i) {
      auto &group = outputGroups[i];
      LLVM_DEBUG(llvm::dbgs() << "buildOutputGroups: Input Vars (Consumer): ");
      for (Value var : group.inputVars) {
          LLVM_DEBUG(llvm::dbgs() << var << " ");
      }
      LLVM_DEBUG(llvm::dbgs() << "\n");
      LLVM_DEBUG(llvm::dbgs() << "buildOutputGroups: Output Vars (Producer): ");

      for (Value output : group.outputs) {
          LLVM_DEBUG(llvm::dbgs() << output << " ");
      }

      LLVM_DEBUG(llvm::dbgs() << "\n");
  }
  return UPDATE_CONDITION_INFO_SUCCESS;
}

// Select corresponding SSBuffer ptr based on ifblock running on vector/cube core
Value UpdateConditionInfoPass::getSSBufferPtr(bool isAIC, int groupIdx, int ptrSetIdx,
                                              DenseMap<int, Value> &VectorSSBufferPtrs,
                                              SmallVector<SmallVector<Value>> ssbufferPtrs)
{
  if (isAIC) {
    return ssbufferPtrs[ptrSetIdx][groupIdx];
  } else {
    return VectorSSBufferPtrs[groupIdx];
  }
}

// Compute pointers for VECTOR core SSBuffer
DenseMap<int, Value> UpdateConditionInfoPass::computeVectorSSBufferPtrs(
    OpBuilder &builder, Location loc,
    Operation *scopeOp,
    SmallVector<int> crossCoreInputValues,
    SmallVector<int> crossCoreOutputValues)
{
  // Collect all unique group indices
  SmallVector<int> allGroupIndices;
  DenseSet<int> uniqueIndices;
  for (int idx : crossCoreInputValues) {
    if (uniqueIndices.insert(idx).second) {
      allGroupIndices.push_back(idx);
    }
  }
  for (int idx : crossCoreOutputValues) {
    if (uniqueIndices.insert(idx).second) {
      allGroupIndices.push_back(idx);
    }
  }

  DenseMap<int, Value> vectorSSBufferPtrs;

  builder.setInsertionPointToStart(&scopeOp->getRegion(0).front());
  int vec1Offset = 1024;
  Value vec1OffsetValue = builder.create<arith::ConstantIntOp>(loc, VECTOR_SSBUF_OFFSET, ADDR_INT_TYPE);
  auto subIdOp = builder.create<GetSubBlockIdxOp>(loc, builder.getIntegerType(ADDR_INT_TYPE));
  Value ssbAddrOffset = builder.create<arith::MulIOp>(loc, subIdOp, vec1OffsetValue);

  for (int groupIdx : allGroupIndices) {
    auto ssbBaseAddr = builder.create<arith::ConstantIntOp>(loc, groupIdx * VALUE_SSBUF_OFFSET, ADDR_INT_TYPE);
    auto ssbAddr = builder.create<arith::AddIOp>(loc, ssbBaseAddr, ssbAddrOffset);
    Value ptr = builder.create<LLVM::IntToPtrOp>(
        loc, LLVM::LLVMPointerType::get(builder.getContext(), SSBUF_ADDR_SPACE),
        ssbAddr.getResult());
    vectorSSBufferPtrs[groupIdx] = ptr;
  }

  return vectorSSBufferPtrs;
}

// Part 2: Add cross-core conditions
Value UpdateConditionInfoPass::addCrossCoreConditions(
    OpBuilder &builder, Location loc,
    SmallVector<int> crossCoreInputValues, SmallVector<int> crossCoreOutputValues,
    DenseMap<int, DenseMap<Value, SmallVector<Value>>> &crossCoreBuffers,
    bool isAIC, Value zeroConst,
    DenseMap<int, Value> &VectorSSBufferPtrs,
    SmallVector<SmallVector<Value>> ssbufferPtrs)
{
  Value conditions = nullptr;

  auto combineCondition = [&](Value newCond) {
    if (conditions) {
      conditions = builder.create<arith::AndIOp>(loc, conditions, newCond);
    } else {
      conditions = newCond;
    }
  };

  for (int inputGroupIdx : crossCoreInputValues) {
    Value cond = nullptr;
    if (isAIC) {
      Value vec0Value = builder.create<LLVM::LoadOp>(loc, builder.getI32Type(),
          getSSBufferPtr(isAIC, inputGroupIdx, 0, VectorSSBufferPtrs, ssbufferPtrs));
      Value vec1Value = builder.create<LLVM::LoadOp>(loc, builder.getI32Type(),
          getSSBufferPtr(isAIC, inputGroupIdx, 1, VectorSSBufferPtrs, ssbufferPtrs));
      Value vec0Cond = builder.create<arith::CmpIOp>(loc, arith::CmpIPredicate::sgt, vec0Value, zeroConst);
      Value vec1Cond = builder.create<arith::CmpIOp>(loc, arith::CmpIPredicate::sgt, vec1Value, zeroConst);
      cond = builder.create<arith::AndIOp>(loc, vec0Cond, vec1Cond);
    } else {
      Value value = builder.create<LLVM::LoadOp>(loc, builder.getI32Type(),
          getSSBufferPtr(isAIC, inputGroupIdx, 0, VectorSSBufferPtrs, ssbufferPtrs));
      cond = builder.create<arith::CmpIOp>(loc, arith::CmpIPredicate::sgt, value, zeroConst);
    }
    combineCondition(cond);
  }

  for (int outputGroupIdx : crossCoreOutputValues) {
    int outputCount = 0;
    for (auto &entry : crossCoreBuffers[outputGroupIdx]) {
      outputCount += entry.second.size();
    }
    Value bufferNum = builder.create<arith::ConstantIntOp>(loc, outputCount, CONST_INT_TYPE);
    Value cond = nullptr;
    if (isAIC) {
      Value vec0Value = builder.create<LLVM::LoadOp>(loc, builder.getI32Type(),
          getSSBufferPtr(isAIC, outputGroupIdx, 0, VectorSSBufferPtrs, ssbufferPtrs));
      Value vec1Value = builder.create<LLVM::LoadOp>(loc, builder.getI32Type(),
          getSSBufferPtr(isAIC, outputGroupIdx, 1, VectorSSBufferPtrs, ssbufferPtrs));
      Value vec0Cond = builder.create<arith::CmpIOp>(loc, arith::CmpIPredicate::slt, vec0Value, bufferNum);
      Value vec1Cond = builder.create<arith::CmpIOp>(loc, arith::CmpIPredicate::slt, vec1Value, bufferNum);
      cond = builder.create<arith::AndIOp>(loc, vec0Cond, vec1Cond);
    } else {
      Value value = builder.create<LLVM::LoadOp>(loc, builder.getI32Type(),
          getSSBufferPtr(isAIC, outputGroupIdx, 0, VectorSSBufferPtrs, ssbufferPtrs));
      cond = builder.create<arith::CmpIOp>(loc, arith::CmpIPredicate::slt, value, bufferNum);
    }
    combineCondition(cond);
  }

  return conditions;
}

// Part 3: Update control variables in then block
void UpdateConditionInfoPass::updateCrossCoreControlVars(
    OpBuilder &builder, Location loc,
    scf::IfOp ifOp, SmallVector<int> crossCoreInputValues,
    SmallVector<int> crossCoreOutputValues,
    bool isAIC, Value oneConst,
    DenseMap<int, Value> &VectorSSBufferPtrs,
    SmallVector<SmallVector<Value>> ssbufferPtrs)
{
  Block *thenBlock = &ifOp.getThenRegion().front();
  auto yieldOp = cast<scf::YieldOp>(thenBlock->getTerminator());
  builder.setInsertionPoint(yieldOp);

  for (int inputGroupIdx : crossCoreInputValues) {
    if (isAIC) {
      Value vec0Value = builder.create<LLVM::LoadOp>(loc, builder.getI32Type(),
          getSSBufferPtr(isAIC, inputGroupIdx, 0, VectorSSBufferPtrs, ssbufferPtrs));
      Value vec1Value = builder.create<LLVM::LoadOp>(loc, builder.getI32Type(),
          getSSBufferPtr(isAIC, inputGroupIdx, 1, VectorSSBufferPtrs, ssbufferPtrs));
      Value vec0NewValue = builder.create<arith::SubIOp>(loc, vec0Value, oneConst);
      Value vec1NewValue = builder.create<arith::SubIOp>(loc, vec1Value, oneConst);
      builder.create<LLVM::StoreOp>(loc, vec0NewValue,
          getSSBufferPtr(isAIC, inputGroupIdx, 0, VectorSSBufferPtrs, ssbufferPtrs));
      builder.create<LLVM::StoreOp>(loc, vec1NewValue,
          getSSBufferPtr(isAIC, inputGroupIdx, 1, VectorSSBufferPtrs, ssbufferPtrs));
    } else {
      Value value = builder.create<LLVM::LoadOp>(loc, builder.getI32Type(),
          getSSBufferPtr(isAIC, inputGroupIdx, 0, VectorSSBufferPtrs, ssbufferPtrs));
      Value newValue = builder.create<arith::SubIOp>(loc, value, oneConst);
      builder.create<LLVM::StoreOp>(loc, newValue,
          getSSBufferPtr(isAIC, inputGroupIdx, 0, VectorSSBufferPtrs, ssbufferPtrs));
    }
  }

  for (int outputGroupIdx : crossCoreOutputValues) {
    if (isAIC) {
      Value vec0Value = builder.create<LLVM::LoadOp>(loc, builder.getI32Type(),
          getSSBufferPtr(isAIC, outputGroupIdx, 0, VectorSSBufferPtrs, ssbufferPtrs));
      Value vec1Value = builder.create<LLVM::LoadOp>(loc, builder.getI32Type(),
          getSSBufferPtr(isAIC, outputGroupIdx, 1, VectorSSBufferPtrs, ssbufferPtrs));
      Value vec0NewValue = builder.create<arith::AddIOp>(loc, vec0Value, oneConst);
      Value vec1NewValue = builder.create<arith::AddIOp>(loc, vec1Value, oneConst);
      builder.create<LLVM::StoreOp>(loc, vec0NewValue,
          getSSBufferPtr(isAIC, outputGroupIdx, 0, VectorSSBufferPtrs, ssbufferPtrs));
      builder.create<LLVM::StoreOp>(loc, vec1NewValue,
          getSSBufferPtr(isAIC, outputGroupIdx, 1, VectorSSBufferPtrs, ssbufferPtrs));
    } else {
      Value value = builder.create<LLVM::LoadOp>(loc, builder.getI32Type(),
          getSSBufferPtr(isAIC, outputGroupIdx, 0, VectorSSBufferPtrs, ssbufferPtrs));
      Value newValue = builder.create<arith::AddIOp>(loc, value, oneConst);
      builder.create<LLVM::StoreOp>(loc, newValue,
          getSSBufferPtr(isAIC, outputGroupIdx, 0, VectorSSBufferPtrs, ssbufferPtrs));
    }
  }
}

// Set the crossCore condition
int UpdateConditionInfoPass::setCrossCoreCondition(
    SmallVector<int> crossCoreInputValues, SmallVector<int> crossCoreOutputValues,
    DenseMap<int, DenseMap<Value, SmallVector<Value>>> &crossCoreBuffers, scf::IfOp ifOp,
    SmallVector<SmallVector<Value>> ssbufferPtrs, Value &crossCoreCond)
{
  OpBuilder builder(ifOp);
  Location loc = ifOp.getLoc();

  // ========== Part 1: Preparation ==========
  // Determine whether the current ifblock is on cube or vector core
  auto aiCAttr = hivm::TCoreTypeAttr::get(builder.getContext(), hivm::TCoreType::CUBE);
  auto aivAttr = hivm::TCoreTypeAttr::get(builder.getContext(), hivm::TCoreType::VECTOR);
  bool isAIC = false;
  bool isAIV = false;
  mlir::Operation *parentOp = ifOp->getParentOp();
  mlir::Operation *scopeOp = nullptr;
  while (parentOp) {
    if (dyn_cast<scope::ScopeOp>(parentOp)) {
      scopeOp = parentOp;
      break;
    }
    parentOp = parentOp->getParentOp();
  }
  if (scopeOp && scopeOp->hasAttr("hivm.tcore_type")) {
    auto attr = scopeOp->getAttr("hivm.tcore_type");
    if (attr == aiCAttr) {
      isAIC = true;
    } else if (attr == aivAttr) {
      isAIV = true;
    } else {
      LDBG("scope block has invalid tcore_type attribute");
      return UPDATE_CONDITION_INFO_FAILED;
    }
  } else {
    LDBG("ifblock not in a correct scope block");
    return UPDATE_CONDITION_INFO_FAILED;
  }

  Value zeroConst = builder.create<arith::ConstantIntOp>(loc, 0, CONST_INT_TYPE);
  Value oneConst = builder.create<arith::ConstantIntOp>(loc, 1, CONST_INT_TYPE);

  // If ifblock is on vector core, compute the required SSBuffer ptrs for vector side
  DenseMap<int, Value> VectorSSBufferPtrs;
  if (!isAIC) {
    VectorSSBufferPtrs = computeVectorSSBufferPtrs(builder, loc, scopeOp, crossCoreInputValues, crossCoreOutputValues);
  }

  builder.setInsertionPoint(ifOp);

  // ========== Part 2: Add cross-core conditions ==========
  crossCoreCond = addCrossCoreConditions(builder, loc, crossCoreInputValues, crossCoreOutputValues,
                                         crossCoreBuffers, isAIC, zeroConst,
                                         VectorSSBufferPtrs, ssbufferPtrs);

  // ========== Part 3: Update control variables ==========
  updateCrossCoreControlVars(builder, loc, ifOp, crossCoreInputValues, crossCoreOutputValues,
                             isAIC, oneConst, VectorSSBufferPtrs, ssbufferPtrs);

  return UPDATE_CONDITION_INFO_SUCCESS;
}

// Collect the conditions for intra-core consumer values.
void UpdateConditionInfoPass::collectIntraCoreInputConditions(
    OpBuilder &builder, Location loc, SmallVector<int> &intraCoreInputValues, DenseMap<int, Value> &idxToVar,
    SmallVector<Value> &conditions, DenseSet<Value> &usedVarsSet,
    DenseMap<Value, VarUpdateType> &varUpdateTypes)
{
  if (intraCoreInputValues.empty()) {
    LLVM_DEBUG(llvm::dbgs() << "No intraCore input conditions to collect.\n");
    return;
  }

  size_t beforeConditionNum = conditions.size();
  Value zeroConst = builder.create<arith::ConstantIntOp>(loc, 0, CONST_INT_TYPE);
  for (int idx : intraCoreInputValues) {
    auto varIt = idxToVar.find(idx);
    if (varIt == idxToVar.end()) {
      LLVM_DEBUG(llvm::dbgs() << "Skip intraCore input group " << idx << ": no control variable.\n");
      continue;
    }

    Value var = varIt->second;
    Value varToUse = var;
    auto latestIt = controlVarToLatestValue.find(var);
    if (latestIt != controlVarToLatestValue.end()) {
      varToUse = latestIt->second;
    }

    Value cond = builder.create<arith::CmpIOp>(loc, arith::CmpIPredicate::sgt, varToUse, zeroConst);
    conditions.push_back(cond);
    usedVarsSet.insert(var);
    varUpdateTypes[var] = VarUpdateType::DEC;
    LLVM_DEBUG(llvm::dbgs() << "Add intraCore input condition for group " << idx << ".\n");
  }
  LLVM_DEBUG(llvm::dbgs() << "Collected " << (conditions.size() - beforeConditionNum)
                           << " intraCore input conditions.\n");
}

// Collect the conditions for intra-core producer values.
int UpdateConditionInfoPass::collectIntraCoreOutputConditions(
    OpBuilder &builder, Location loc, DenseMap<int, DenseMap<Value, SmallVector<Value>>> &intraCoreBuffers,
    SmallVector<int> &intraCoreOutputValues, DenseMap<int, Value> &idxToVar, SmallVector<Value> &conditions,
    DenseSet<Value> &usedVarsSet, DenseMap<Value, VarUpdateType> &varUpdateTypes)
{
  if (intraCoreOutputValues.empty()) {
    LLVM_DEBUG(llvm::dbgs() << "No intraCore output conditions to collect.\n");
    return UPDATE_CONDITION_INFO_SUCCESS;
  }

  size_t beforeConditionNum = conditions.size();
  SmallVector<OutputGroupInfo> outputGroups;
  if (buildOutputGroups(intraCoreOutputValues, intraCoreBuffers, idxToVar, outputGroups) ==
      UPDATE_CONDITION_INFO_FAILED) {
    return UPDATE_CONDITION_INFO_FAILED;
  }
  for (auto &group : outputGroups) {
    int size = group.outputs.size();
    Value limitVal = builder.create<arith::ConstantIntOp>(loc, size, CONST_INT_TYPE);
    for (Value var : group.inputVars) {
      Value varToUse = var;
      auto latestIt = controlVarToLatestValue.find(var);
      if (latestIt != controlVarToLatestValue.end()) {
        varToUse = latestIt->second;
      }

      Value cond = builder.create<arith::CmpIOp>(loc, arith::CmpIPredicate::slt, varToUse, limitVal);
      conditions.push_back(cond);
      usedVarsSet.insert(var);
      varUpdateTypes[var] = VarUpdateType::INC;
      LLVM_DEBUG(llvm::dbgs() << "Add intraCore output condition with producer limit " << size << ".\n");
    }
  }
  LLVM_DEBUG(llvm::dbgs() << "Collected " << (conditions.size() - beforeConditionNum)
                           << " intraCore output conditions.\n");
  return UPDATE_CONDITION_INFO_SUCCESS;
}

// Set the intraCore condition.
int UpdateConditionInfoPass::setIntraCoreCondition(
    ModuleOp module, scf::IfOp ifOp, DenseMap<int, DenseMap<Value, SmallVector<Value>>> &intraCoreBuffers,
    SmallVector<int> &intraCoreInputValues, SmallVector<int> &intraCoreOutputValues, DenseMap<int, Value> &idxToVar,
    DenseMap<Value, VarUpdateType> &varUpdateTypes, Value &intraCoreCond)
{
  LDBG("Enter set intraCore condition.");
  intraCoreCond = Value();
  OpBuilder builder(ifOp.getContext());
  builder.setInsertionPoint(ifOp);
  Location loc = ifOp.getLoc();

  SmallVector<Value> conditions;
  DenseSet<Value> usedVarsSet;
  LLVM_DEBUG(llvm::dbgs() << "Collect intraCore conditions: inputs " << intraCoreInputValues.size()
                           << ", outputs " << intraCoreOutputValues.size() << "\n");
  // Collect the conditions for intra-core consumer values.
  collectIntraCoreInputConditions(builder, loc, intraCoreInputValues, idxToVar, conditions, usedVarsSet,
                                  varUpdateTypes);
  // Collect the conditions for intra-core producer values.
  if (collectIntraCoreOutputConditions(builder, loc, intraCoreBuffers, intraCoreOutputValues, idxToVar, conditions,
                                       usedVarsSet, varUpdateTypes) == UPDATE_CONDITION_INFO_FAILED) {
    return UPDATE_CONDITION_INFO_FAILED;
  }

  if (!conditions.empty()) {
    intraCoreCond = conditions[0];
    for (size_t i = 1; i < conditions.size(); ++i) {
      intraCoreCond = builder.create<arith::AndIOp>(loc, intraCoreCond, conditions[i]);
    }
  }

  currentUsedVars.clear();
  for (Value var : usedVarsSet) {
    currentUsedVars.push_back(var);
  }
  LLVM_DEBUG(llvm::dbgs() << "Built " << conditions.size() << " intraCore conditions using "
                           << currentUsedVars.size() << " control variables.\n");

  LDBG("Exit set intraCore condition.");
  return UPDATE_CONDITION_INFO_SUCCESS;
}

// Update the mapping of control variables to their latest values
void UpdateConditionInfoPass::updateControlVarToLatestValue(scf::IfOp newIfOp, scf::IfOp oldIfOp, bool hasCounter,
                                                            Value counter)
{
  if (currentUsedVars.empty() && !hasCounter) {
    LLVM_DEBUG(llvm::dbgs() << "No control variable latest values to update.\n");
    return;
  }

  size_t origResultCount = oldIfOp.getNumResults();

  for (size_t i = 0; i < currentUsedVars.size(); ++i) {
    Value var = currentUsedVars[i];
    Value newValue = newIfOp.getResult(origResultCount + i);
    controlVarToLatestValue[var] = newValue;
    LLVM_DEBUG(llvm::dbgs() << "Record latest intraCore control value at result index "
                             << (origResultCount + i) << ".\n");
  }

  if (hasCounter) {
    size_t counterResultIdx = origResultCount + currentUsedVars.size();
    Value newCounterValue = newIfOp.getResult(counterResultIdx);
    controlVarToLatestValue[counter] = newCounterValue;
    LLVM_DEBUG(llvm::dbgs() << "Record latest counter value at result index " << counterResultIdx << ".\n");
  }
  LLVM_DEBUG(llvm::dbgs() << "[DEBUG] controlVarToLatestValue size: " << controlVarToLatestValue.size() << "\n");

  for (auto &entry : controlVarToLatestValue) {
    LLVM_DEBUG(llvm::dbgs() << "[DEBUG]   key = " << entry.first << "  -->  new value = " << entry.second << "\n");
  }
}

// Update the yield in the forOp
int UpdateConditionInfoPass::updateForOpYield(scf::ForOp forOp)
{
  LDBG("Enter update forOp yield ");
  if (controlVarToLatestValue.empty()) {
    LLVM_DEBUG(llvm::dbgs() << "Failed to update forOp yield: no latest control variable values.\n");
    return UPDATE_CONDITION_INFO_FAILED;
  }

  Location loc = forOp.getLoc();
  Block *forBody = forOp.getBody();
  auto yieldOp = dyn_cast<scf::YieldOp>(forBody->getTerminator());
  if (!yieldOp) {
    LLVM_DEBUG(llvm::dbgs() << "Failed to update forOp yield: terminator is not scf.yield.\n");
    return UPDATE_CONDITION_INFO_FAILED;
  }

  SmallVector<Value> newYieldOperands(yieldOp.getOperands().begin(), yieldOp.getOperands().end());
  if (newYieldOperands.size() != forOp.getNumRegionIterArgs()) {
    LLVM_DEBUG(llvm::dbgs() << "Failed to update forOp yield: yield operands " << newYieldOperands.size()
                             << ", iter args " << forOp.getNumRegionIterArgs() << "\n");
    return UPDATE_CONDITION_INFO_FAILED;
  }

  DenseMap<Value, unsigned> iterArgToIndex;
  for (unsigned j = 0; j < forOp.getNumRegionIterArgs(); ++j) {
    iterArgToIndex[forOp.getRegionIterArgs()[j]] = j;
  }

  for (auto &entry : controlVarToLatestValue) {
    Value origVar = entry.first;
    Value latestValue = entry.second;
    auto it = iterArgToIndex.find(origVar);
    if (it == iterArgToIndex.end()) {
      LLVM_DEBUG(llvm::dbgs() << "Failed to update forOp yield: control variable is not a region iter arg.\n");
      return UPDATE_CONDITION_INFO_FAILED;
    }
    newYieldOperands[it->second] = latestValue;
    LLVM_DEBUG(llvm::dbgs() << "Update forOp yield operand index " << it->second << "\n");
  }

  OpBuilder yieldBuilder(yieldOp);
  yieldBuilder.create<scf::YieldOp>(loc, newYieldOperands);
  yieldOp.erase();
  LLVM_DEBUG(llvm::dbgs() << "Updated forOp yield with " << controlVarToLatestValue.size()
                           << " latest control values.\n");
  LDBG("Exit update forOp yield ");
  return UPDATE_CONDITION_INFO_SUCCESS;
}

SmallVector<Type> UpdateConditionInfoPass::buildNewIfResultTypes(scf::IfOp oldIfOp, bool hasCounter, Value counter)
{
  SmallVector<Type> resultTypes;
  for (Value result : oldIfOp.getResults()) {
    resultTypes.push_back(result.getType());
  }
  for (Value var : currentUsedVars) {
    resultTypes.push_back(var.getType());
  }
  if (hasCounter) {
    resultTypes.push_back(counter.getType());
  }
  LLVM_DEBUG(llvm::dbgs() << "Build new if result types: old results " << oldIfOp.getNumResults()
                           << ", control vars " << currentUsedVars.size()
                           << ", has counter " << hasCounter << ".\n");
  return resultTypes;
}

void UpdateConditionInfoPass::collectYieldOperands(Block &block, Operation *&yieldOp,
                                                   SmallVector<Value> &yieldOperands)
{
  yieldOp = nullptr;
  yieldOperands.clear();
  if (block.empty()) {
    LLVM_DEBUG(llvm::dbgs() << "Collect yield operands: block is empty.\n");
    return;
  }

  Operation *lastOp = &block.back();
  if (!isa<scf::YieldOp>(lastOp)) {
    LLVM_DEBUG(llvm::dbgs() << "Collect yield operands: block terminator is not scf.yield.\n");
    return;
  }

  yieldOp = lastOp;
  auto scfYieldOp = cast<scf::YieldOp>(lastOp);
  yieldOperands.assign(scfYieldOp.getOperands().begin(), scfYieldOp.getOperands().end());
  LLVM_DEBUG(llvm::dbgs() << "Collected " << yieldOperands.size() << " yield operands.\n");
}

void UpdateConditionInfoPass::populateNewThenBlock(
    scf::IfOp newIfOp, Block &oldThenBlock, Operation *oldThenYieldOp, ArrayRef<Value> oldYieldOperands,
    DenseMap<Value, VarUpdateType> &varUpdateTypes, bool hasCounter, Value counter, Value step)
{
  Location loc = newIfOp.getLoc();
  Block &newThenBlock = newIfOp.getThenRegion().front();
  for (Operation &op : llvm::make_early_inc_range(oldThenBlock)) {
    if (&op != oldThenYieldOp) {
      op.moveBefore(&newThenBlock, newThenBlock.end());
    }
  }
  LLVM_DEBUG(llvm::dbgs() << "Populate new then block with " << oldYieldOperands.size()
                           << " original yield operands.\n");

  OpBuilder thenBuilder(&newThenBlock, newThenBlock.end());
  SmallVector<Value> thenYieldOperands(oldYieldOperands.begin(), oldYieldOperands.end());
  if (!currentUsedVars.empty()) {
    Value one = thenBuilder.create<arith::ConstantIntOp>(loc, 1, CONST_INT_TYPE);
    for (Value var : currentUsedVars) {
      Value varToUse = var;
      auto latestIt = controlVarToLatestValue.find(var);
      if (latestIt != controlVarToLatestValue.end()) {
        varToUse = latestIt->second;
      }

      Value yieldVal = varToUse;
      auto it = varUpdateTypes.find(var);
      if (it != varUpdateTypes.end()) {
        if (it->second == VarUpdateType::DEC) {
          yieldVal = thenBuilder.create<arith::SubIOp>(loc, varToUse, one);
        } else if (it->second == VarUpdateType::INC) {
          yieldVal = thenBuilder.create<arith::AddIOp>(loc, varToUse, one);
        }
      }
      thenYieldOperands.push_back(yieldVal);
    }
  }

  if (hasCounter) {
    Value newCounter = thenBuilder.create<arith::AddIOp>(loc, counter, step);
    thenYieldOperands.push_back(newCounter);
    LLVM_DEBUG(llvm::dbgs() << "Append updated counter to then yield.\n");
  }

  LLVM_DEBUG(llvm::dbgs() << "Create then yield with " << thenYieldOperands.size() << " operands.\n");
  thenBuilder.create<scf::YieldOp>(loc, thenYieldOperands);
}

void UpdateConditionInfoPass::populateNewElseBlock(scf::IfOp newIfOp, scf::IfOp oldIfOp, bool needsYield,
                                                   bool oldHasElse, bool hasCounter, Value counter)
{
  if (!needsYield && !oldHasElse) {
    LLVM_DEBUG(llvm::dbgs() << "Skip populating else block: no yield needed and old if has no else.\n");
    return;
  }

  Location loc = newIfOp.getLoc();
  Block &newElseBlock = newIfOp.getElseRegion().front();
  SmallVector<Value> oldElseYieldOperands;
  Operation *oldElseYieldOp = nullptr;

  if (oldHasElse) {
    Block &oldElseBlock = oldIfOp.getElseRegion().front();
    collectYieldOperands(oldElseBlock, oldElseYieldOp, oldElseYieldOperands);
    for (Operation &op : llvm::make_early_inc_range(oldElseBlock)) {
      if (&op != oldElseYieldOp) {
        op.moveBefore(&newElseBlock, newElseBlock.end());
      }
    }
    LLVM_DEBUG(llvm::dbgs() << "Moved old else block ops and collected " << oldElseYieldOperands.size()
                             << " old else yield operands.\n");
  }

  if (needsYield) {
    OpBuilder elseBuilder(&newElseBlock, newElseBlock.end());
    SmallVector<Value> elseYieldOperands;
    for (Value operand : oldElseYieldOperands) {
      Value newOperand = operand;
      auto it = controlVarToLatestValue.find(operand);
      if (it != controlVarToLatestValue.end()) {
        newOperand = it->second;
      }
      elseYieldOperands.push_back(newOperand);
    }

    for (Value var : currentUsedVars) {
      Value varToUse = var;
      auto it = controlVarToLatestValue.find(var);
      if (it != controlVarToLatestValue.end()) {
        varToUse = it->second;
      }
      elseYieldOperands.push_back(varToUse);
    }

    if (hasCounter) {
      Value counterToUse = counter;
      auto it = controlVarToLatestValue.find(counter);
      if (it != controlVarToLatestValue.end()) {
        counterToUse = it->second;
      }
      elseYieldOperands.push_back(counterToUse);
    }

    LLVM_DEBUG(llvm::dbgs() << "Create else yield with " << elseYieldOperands.size() << " operands.\n");
    elseBuilder.create<scf::YieldOp>(loc, elseYieldOperands);
  } else if (oldElseYieldOp) {
    oldElseYieldOp->erase();
    LLVM_DEBUG(llvm::dbgs() << "Erase old else yield because new if does not need yield values.\n");
  }
}

// Create new IfOp with new then and else blocks.
scf::IfOp UpdateConditionInfoPass::createNewIfOpWithBlocks(scf::IfOp oldIfOp, Value combinedCond,
                                                           DenseMap<Value, VarUpdateType> &varUpdateTypes,
                                                           bool hasCounter, Value counter, Value step)
{
  Location loc = oldIfOp.getLoc();
  OpBuilder builder(oldIfOp);

  bool needsYield = !currentUsedVars.empty() || hasCounter;
  bool oldHasElse = oldIfOp.getElseRegion().hasOneBlock();
  LLVM_DEBUG(llvm::dbgs() << "Create replacement if op: needs yield " << needsYield
                           << ", old has else " << oldHasElse
                           << ", current used vars " << currentUsedVars.size() << ".\n");

  Block &oldThenBlock = oldIfOp.getThenRegion().front();
  Operation *oldThenYieldOp = nullptr;
  SmallVector<Value> oldYieldOperands;
  collectYieldOperands(oldThenBlock, oldThenYieldOp, oldYieldOperands);
  SmallVector<Type> resultTypes = buildNewIfResultTypes(oldIfOp, hasCounter, counter);
  scf::IfOp newIfOp = builder.create<scf::IfOp>(loc, resultTypes, combinedCond, true);
  LLVM_DEBUG(llvm::dbgs() << "Created replacement if op with " << resultTypes.size() << " results.\n");

  for (auto &attr : oldIfOp->getAttrs()) {
    newIfOp->setAttr(attr.getName(), attr.getValue());
  }

  populateNewThenBlock(newIfOp, oldThenBlock, oldThenYieldOp, oldYieldOperands, varUpdateTypes, hasCounter, counter,
                       step);
  populateNewElseBlock(newIfOp, oldIfOp, needsYield, oldHasElse, hasCounter, counter);

  for (size_t i = 0; i < oldIfOp.getNumResults(); ++i) {
    oldIfOp.getResult(i).replaceAllUsesWith(newIfOp.getResult(i));
  }
  LLVM_DEBUG(llvm::dbgs() << "Replaced " << oldIfOp.getNumResults() << " old if results.\n");

  return newIfOp;
}

// Combine the three conditions: crossCore condition + intraCore condition + counter condition
int UpdateConditionInfoPass::combineConditions(ModuleOp module, Value crossCoreCond, Value intraCoreCond,
                                               scf::IfOp ifOp, scf::ForOp forOp, size_t &usedCounterNum,
                                               DenseMap<Value, VarUpdateType> &varUpdateTypes)
{
  Location loc = ifOp.getLoc();
  SmallVector<Value> validConditions;
  Value counter;
  bool hasCounter = false;

  if (crossCoreCond) {
    validConditions.push_back(crossCoreCond);
  }
  if (intraCoreCond) {
    validConditions.push_back(intraCoreCond);
  }

  if (!info->blockCounters.count(forOp)) {
    LLVM_DEBUG(llvm::dbgs() << "Missing block counters for forOp.\n");
    return UPDATE_CONDITION_INFO_FAILED;
  }

  SmallVector<int> &counterIndices = info->blockCounters[forOp];

  if (info->cntArgs.count(ifOp)) {
    counter = info->cntArgs[ifOp];
    hasCounter = true;
  } else {
    if (usedCounterNum >= counterIndices.size()) {
      LLVM_DEBUG(llvm::dbgs() << "Not enough counters for ssbuffer if ops: used " << usedCounterNum
                               << ", counters " << counterIndices.size() << "\n");
      return UPDATE_CONDITION_INFO_FAILED;
    }

    int argIdx = counterIndices[usedCounterNum];
    int iterArgNum = static_cast<int>(forOp.getNumRegionIterArgs());
    if (argIdx < 0 || argIdx >= iterArgNum) {
      LLVM_DEBUG(llvm::dbgs() << "Invalid counter arg index: " << argIdx << ", iter args " << iterArgNum << "\n");
      return UPDATE_CONDITION_INFO_FAILED;
    }

    counter = forOp.getRegionIterArgs()[argIdx];
    hasCounter = true;
    info->cntArgs[ifOp] = counter;
    usedCounterNum++;
    LLVM_DEBUG(llvm::dbgs() << "Assign counter iter arg index " << argIdx << " to ssbuffer if op.\n");
  }

  LLVM_DEBUG(llvm::dbgs() << "this ifop used counter is: " << counter << "\n");
  if (hasCounter) {
    OpBuilder builder(ifOp);
    Value upperBound = forOp.getUpperBound();
    Value counterToUse = counter;
    auto latestIt = controlVarToLatestValue.find(counter);
    if (latestIt != controlVarToLatestValue.end()) {
      counterToUse = latestIt->second;
    }
    Value counterCond = builder.create<arith::CmpIOp>(loc, arith::CmpIPredicate::slt, counterToUse, upperBound);
    validConditions.push_back(counterCond);
  }

  if (validConditions.empty()) {
    LLVM_DEBUG(llvm::dbgs() << "Failed to build any condition for ssbuffer if op.\n");
    return UPDATE_CONDITION_INFO_FAILED;
  }

  LLVM_DEBUG(llvm::dbgs() << "Combine " << validConditions.size() << " conditions for ssbuffer if op.\n");
  OpBuilder builder(ifOp);
  Value combinedCond = validConditions[0];
  for (size_t i = 1; i < validConditions.size(); ++i) {
    combinedCond = builder.create<arith::AndIOp>(loc, combinedCond, validConditions[i]);
  }

  scf::IfOp newIfOp = createNewIfOpWithBlocks(ifOp, combinedCond, varUpdateTypes, hasCounter, counter, forOp.getStep());

  if (hasCounter) {
    info->cntArgs.erase(ifOp);
    info->cntArgs[newIfOp] = counter;
  }

  updateControlVarToLatestValue(newIfOp, ifOp, hasCounter, counter);

  ifOp.erase();
  return UPDATE_CONDITION_INFO_SUCCESS;
}

// Update the conditions of ifOp.
int UpdateConditionInfoPass::updateIfConds(ModuleOp module, SmallVector<SmallVector<Value> > ssbufferPtrs)
{
  // Walk the forOp in the module to update the conditions of ifOp
  SmallVector<scf::ForOp> mainLoopForOps;
  WalkResult walkResult = module.walk([&](Operation *op) -> WalkResult {
    if (!op->hasAttr(SSBUFFER_Main_LOOP)) {
      return WalkResult::advance();
    }

    auto forOp = dyn_cast<scf::ForOp>(op);
    if (!forOp) {
      LLVM_DEBUG(llvm::dbgs() << "Found unsupported main loop op: "
                 << op->getName() << "\n");
      return WalkResult::interrupt();
    }
    mainLoopForOps.push_back(forOp);
    return WalkResult::advance();
  });
  if (walkResult.wasInterrupted()) {
    return UPDATE_CONDITION_INFO_FAILED;
  }
  for (scf::ForOp forOp : mainLoopForOps) {
    controlVarToLatestValue.clear();

    DenseMap<int, DenseMap<Value, SmallVector<Value> > > crossCoreBuffers;
    DenseMap<int, DenseMap<Value, SmallVector<Value> > > intraCoreBuffers;
    // Step1:Collect the dependency buffer info of this forOp
    collectDependencyBuffers(forOp, crossCoreBuffers, intraCoreBuffers);
    if (crossCoreBuffers.empty() && intraCoreBuffers.empty()) {
      LDBG("crossCoreBuffers and intraCoreBuffers are both empty!");
      return UPDATE_CONDITION_INFO_FAILED;
    }

    DenseMap<int, DenseMap<Value, SmallVector<Value> > > extendedCrossCoreBuffers =
        extendCrossCoreBuffersWithEquivalentValues(module, crossCoreBuffers);
    if (extendedCrossCoreBuffers.count(-1)) {
      LDBG("extendCrossCoreBuffersWithEquivalentValues failed!");
      return UPDATE_CONDITION_INFO_FAILED;
    }
    // Step2:Assign a variable to each inputValue of this forOp
    DenseMap<int, Value> idxToVar;
    if (buildIdxToVarMap(forOp, intraCoreBuffers, idxToVar) == UPDATE_CONDITION_INFO_FAILED) {
      return UPDATE_CONDITION_INFO_FAILED;
    }
    size_t usedCounterNum = 0;
    SmallVector<scf::IfOp> ifOps;
    WalkResult ifWalkResult = forOp.walk([&](Operation *op) -> WalkResult {
      if (!op->hasAttr(SSBUFFER_IF)) {
        return WalkResult::advance();
      }

      auto ifOp = dyn_cast<scf::IfOp>(op);
      if (!ifOp) {
        LLVM_DEBUG(llvm::dbgs() << "Found unsupported ssbuffer if op: " << op->getName() << "\n");
        return WalkResult::interrupt();
      }

      ifOps.push_back(ifOp);
      return WalkResult::advance();
    });
    if (ifWalkResult.wasInterrupted()) {
      return UPDATE_CONDITION_INFO_FAILED;
    }
    auto counterIt = info->blockCounters.find(forOp);
    if (counterIt == info->blockCounters.end()) {
      LLVM_DEBUG(llvm::dbgs() << "Failed to assign counters for ssbuffer if ops: no counters for forOp.\n");
      return UPDATE_CONDITION_INFO_FAILED;
    }

    size_t counterNum = counterIt->second.size();
    if (ifOps.size() > counterNum) {
      LLVM_DEBUG(llvm::dbgs() << "Failed to assign counters for all ssbuffer if ops: if ops "
                               << ifOps.size() << ", counters " << counterNum << "\n");
      return UPDATE_CONDITION_INFO_FAILED;
    }
    // Update the conditions of ifOp in this forOp.
    for (scf::IfOp ifOp : ifOps) {
      // Walk the ifOp in this forOp to update the conditions of ifOp
      SmallVector<int> crossCoreInputValues;
      SmallVector<int> crossCoreOutputValues;
      SmallVector<int> intraCoreInputValues;
      SmallVector<int> intraCoreOutputValues;

      if (getInputOutputValues(ifOp, extendedCrossCoreBuffers, intraCoreBuffers, crossCoreInputValues,
                               crossCoreOutputValues, intraCoreInputValues, intraCoreOutputValues) != 0) {
        LDBG("getInputOutputValues failed!");
        return UPDATE_CONDITION_INFO_FAILED;
      }

      // Step3:Set the crossCore condition
      Value crossCoreCond;
      if (setCrossCoreCondition(crossCoreInputValues, crossCoreOutputValues, crossCoreBuffers, ifOp,
                                ssbufferPtrs, crossCoreCond) != 0) {
        LDBG("setCrossCoreCondition failed!");
        return UPDATE_CONDITION_INFO_FAILED;
      }
      // Step4:Set the intraCore condition
      DenseMap<Value, VarUpdateType> varUpdateTypes;
      Value intraCoreCond;
      if (setIntraCoreCondition(module, ifOp, intraCoreBuffers, intraCoreInputValues, intraCoreOutputValues, idxToVar,
                                varUpdateTypes, intraCoreCond) == UPDATE_CONDITION_INFO_FAILED) {
        return UPDATE_CONDITION_INFO_FAILED;
      }
      // Step5:Combine the three conditions: crossCore condition + intraCore condition + counter condition
      if (combineConditions(module, crossCoreCond, intraCoreCond, ifOp, forOp, usedCounterNum,
                            varUpdateTypes) == UPDATE_CONDITION_INFO_FAILED) {
        return UPDATE_CONDITION_INFO_FAILED;
      }
    }
    // Step6:Update the yield variable of the forOp
    if (updateForOpYield(forOp) == UPDATE_CONDITION_INFO_FAILED) {
      return UPDATE_CONDITION_INFO_FAILED;
    }
  }
  return UPDATE_CONDITION_INFO_SUCCESS;
}

void UpdateConditionInfoPass::runOnOperation()
{
  ModuleOp module = getOperation();

  LDBG("Enter UpdateConditionInfo pass.\n");
  // Step1:Init the ssbufferPtrs
  SmallVector<SmallVector<Value> > ssbufferPtrs = allocSSBuffer(module);

  // Step2:Update the conditions of ifOp based on the intraCoreDependentMap and crossCoreDependentMap
  int updateResult = updateIfConds(module, ssbufferPtrs);
  if (updateResult == UPDATE_CONDITION_INFO_FAILED) {
    signalPassFailure();
    return;
  }
  LDBG("Exit UpdateConditionInfo pass.\n");
}

namespace mlir {
namespace triton {
std::unique_ptr<OperationPass<ModuleOp> > createUpdateConditionInfoPass()
{
  return std::make_unique<UpdateConditionInfoPass>();
}
} // namespace triton
} // namespace mlir
