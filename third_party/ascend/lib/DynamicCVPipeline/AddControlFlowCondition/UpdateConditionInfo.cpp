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

#include <functional>
#include <optional>
#include <string>
#include <type_traits>

#include "llvm/ADT/APFloat.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/CodeGen/MachinePassManager.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/raw_ostream.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/Location.h"
#include "mlir/IR/ValueRange.h"

#include "ascend/include/DynamicCVPipeline/AddControlFlowCondition.h"
#include "ascend/include/DynamicCVPipeline/Common/SSBufferManager.h"
#include "ascend/include/DynamicCVPipeline/Common/Utils.h"

#include "bishengir/Dialect/Annotation/IR/Annotation.h"
#include "bishengir/Dialect/HIVM/IR/HIVM.h"
#include "bishengir/Dialect/HIVM/IR/HIVMInterfaces.h"
#include "bishengir/Dialect/Scope/IR/Scope.h"
#include "mlir/Pass/PassManager.h"
#include "third_party/ascend/include/DynamicCVPipeline/AddControlFlowCondition/UpdateConditionInfo.h"

static constexpr const char *DEBUG_TYPE = "UpdateConditionInfoPass";
static constexpr int VECTOR_SSBUF_OFFSET = 1024;
static constexpr int VALUE_SSBUF_OFFSET = 4;
static constexpr int UPDATE_CONDITION_INFO_SUCCESS = 0;
static constexpr int UPDATE_CONDITION_INFO_FAILED = -1;
static constexpr size_t kVectorCount = 2;

#define DBGS() (llvm::dbgs() << '[' << DEBUG_TYPE << "] ")
#define LDBG(...)                                                              \
  LLVM_DEBUG({                                                                 \
    DBGS();                                                                    \
    llvm::dbgs() << __VA_ARGS__;                                               \
  })
using namespace mlir;
using namespace triton;
using namespace hivm;
using namespace CVPipeline;

static void logConditionGroupIndices(llvm::StringRef label,
                                     llvm::ArrayRef<int> groupIndices) {
  std::string message;
  llvm::raw_string_ostream os(message);
  os << label;
  for (int groupIndex : groupIndices) {
    os << groupIndex << " ";
  }
  os << "\n";
  LDBG(os.str());
}

static void logOutputGroupValues(llvm::StringRef label,
                                 llvm::ArrayRef<Value> values) {
  std::string message;
  llvm::raw_string_ostream os(message);
  os << label;
  for (Value value : values) {
    os << value << " ";
  }
  os << "\n";
  LDBG(os.str());
}

// Read block id from ssbuffer.if on ifOp. Missing attr is unexpected.
static int getIfBlockId(scf::IfOp ifOp, int &outBlockId) {
  auto attr = ifOp->getAttrOfType<IntegerAttr>(kIf);
  if (!attr) {
    LDBG("ssbuffer.if missing block id on ifOp: " << ifOp << "\n");
    return UPDATE_CONDITION_INFO_FAILED;
  }
  outBlockId = static_cast<int>(attr.getInt());
  return UPDATE_CONDITION_INFO_SUCCESS;
}

// for: body region iter args; while-do: after-region args.
static int getLoopRegionIterArgs(Operation *loopOp,
                                 MutableArrayRef<BlockArgument> &outArgs) {
  if (auto forOp = dyn_cast<scf::ForOp>(loopOp)) {
    outArgs = forOp.getRegionIterArgs();
    return UPDATE_CONDITION_INFO_SUCCESS;
  }
  if (auto whileOp = dyn_cast<scf::WhileOp>(loopOp)) {
    outArgs = whileOp.getAfterArguments();
    return UPDATE_CONDITION_INFO_SUCCESS;
  }
  LDBG("getLoopRegionIterArgs expects scf.for or scf.while, got " << *loopOp
                                                                  << "\n");
  return UPDATE_CONDITION_INFO_FAILED;
}

static int getLoopRegionIterArg(Operation *loopOp, int argIdx, Value &outArg) {
  MutableArrayRef<BlockArgument> args;
  if (getLoopRegionIterArgs(loopOp, args) == UPDATE_CONDITION_INFO_FAILED)
    return UPDATE_CONDITION_INFO_FAILED;
  if (argIdx < 0 || argIdx >= static_cast<int>(args.size())) {
    LDBG("Invalid loop region iter arg index: "
         << argIdx << ", iter args " << args.size() << ", loopOp=" << *loopOp
         << "\n");
    return UPDATE_CONDITION_INFO_FAILED;
  }
  outArg = args[argIdx];
  return UPDATE_CONDITION_INFO_SUCCESS;
}

// Clone only the SSA def-chain that produces `condition` when it lives in
// beforeRegion.
static int cloneConditionDefChain(Value value, Region &beforeRegion,
                                  IRMapping &mapping, OpBuilder &builder,
                                  Value &outValue) {
  if (Value mapped = mapping.lookupOrNull(value)) {
    outValue = mapped;
    return UPDATE_CONDITION_INFO_SUCCESS;
  }

  if (auto blockArg = dyn_cast<BlockArgument>(value)) {
    if (blockArg.getParentRegion() == &beforeRegion) {
      LDBG("Before-region arg is not in whileBlockArgMap mapping: " << value
                                                                    << "\n");
      return UPDATE_CONDITION_INFO_FAILED;
    }
    outValue = value;
    return UPDATE_CONDITION_INFO_SUCCESS;
  }

  Operation *defOp = value.getDefiningOp();
  if (!defOp) {
    outValue = value;
    return UPDATE_CONDITION_INFO_SUCCESS;
  }
  // Defined outside before-region: reuse as-is (constants, outer values).
  if (defOp->getParentRegion() != &beforeRegion) {
    outValue = value;
    return UPDATE_CONDITION_INFO_SUCCESS;
  }

  for (Value operand : defOp->getOperands()) {
    Value remappedOperand;
    if (cloneConditionDefChain(operand, beforeRegion, mapping, builder,
                               remappedOperand) == UPDATE_CONDITION_INFO_FAILED)
      return UPDATE_CONDITION_INFO_FAILED;
    if (!mapping.lookupOrNull(operand))
      mapping.map(operand, remappedOperand);
  }

  Operation *cloned = builder.clone(*defOp, mapping);
  outValue = cloned->getResult(cast<OpResult>(value).getResultNumber());
  return UPDATE_CONDITION_INFO_SUCCESS;
}

// Remap scf.condition(x): clone only ops that produce x into the after region,
// remapping before args via whileBlockArgMap[while][blockId]:
// {new_arg_idx(after) -> old_arg_idx(before)}.
static int buildWhileCounterCondition(
    scf::WhileOp whileOp, scf::IfOp ifOp, ControlFlowConditionInfo *info,
    OpBuilder &builder, const DenseMap<Value, Value> &controlVarToLatestValue,
    Value &outCond) {
  int blockId;
  if (getIfBlockId(ifOp, blockId) == UPDATE_CONDITION_INFO_FAILED)
    return UPDATE_CONDITION_INFO_FAILED;

  auto whileIt = info->whileBlockArgMap.find(whileOp);
  if (whileIt == info->whileBlockArgMap.end()) {
    LDBG("whileBlockArgMap has no entry for whileOp=" << whileOp << ", ifOp="
                                                      << ifOp << "\n");
    return UPDATE_CONDITION_INFO_FAILED;
  }
  auto blockIt = whileIt->second.find(blockId);
  if (blockIt == whileIt->second.end()) {
    LDBG("whileBlockArgMap has no entry for blockId "
         << blockId << ", whileOp=" << whileOp << ", ifOp=" << ifOp << "\n");
    return UPDATE_CONDITION_INFO_FAILED;
  }

  const DenseMap<int, int> &argIdxMap = blockIt->second;
  auto beforeArgs = whileOp.getBeforeArguments();
  auto afterArgs = whileOp.getAfterArguments();

  IRMapping mapping;
  for (auto [newArgIdx, oldArgIdx] : argIdxMap) {
    if (oldArgIdx < 0 || oldArgIdx >= static_cast<int>(beforeArgs.size()) ||
        newArgIdx < 0 || newArgIdx >= static_cast<int>(afterArgs.size())) {
      LDBG("Invalid whileBlockArgMap entry (new="
           << newArgIdx << ", old=" << oldArgIdx << "), whileOp=" << whileOp
           << ", ifOp=" << ifOp << "\n");
      return UPDATE_CONDITION_INFO_FAILED;
    }
    Value afterArg = afterArgs[newArgIdx];
    auto latestIt = controlVarToLatestValue.find(afterArg);
    if (latestIt != controlVarToLatestValue.end())
      afterArg = latestIt->second;
    mapping.map(beforeArgs[oldArgIdx], afterArg);
  }

  // Only take condition value x from scf.condition(x);
  scf::ConditionOp condOp = whileOp.getConditionOp();
  Value beforeCond = condOp.getCondition();
  if (cloneConditionDefChain(beforeCond, whileOp.getBefore(), mapping, builder,
                             outCond) == UPDATE_CONDITION_INFO_FAILED) {
    LDBG("Failed to remap while before-region condition expression, whileOp="
         << whileOp << ", ifOp=" << ifOp << "\n");
    return UPDATE_CONDITION_INFO_FAILED;
  }
  return UPDATE_CONDITION_INFO_SUCCESS;
}

// Allocate the SSBuffer pointer
SmallVector<SmallVector<Value>>
UpdateConditionInfoPass::allocSSBuffer(ModuleOp module) {
  OpBuilder builder(module.getContext());
  auto i64Type = builder.getIntegerType(ADDR_INT_TYPE);
  auto i32Type = builder.getIntegerType(CONST_INT_TYPE);
  auto memrefType = getSsbufMemrefType(builder);

  // alloc 2 group of ssbuffer pointers:
  // Core Vector 0: allocate ssbuffer address: 0, 4, 8, ...
  // Core Vector 1: allocate ssbuffer address: 1024, 1028, 1032, ...
  SmallVector<SmallVector<Value>> ssbufferMemrefs;
  SmallVector<Value> ssbufferVec0Memrefs;
  SmallVector<Value> ssbufferVec1Memrefs;
  int numBuffers = countProducerGroups(info->crossCoreDependentMap);
  if (numBuffers == 0) {
    LDBG("crossCoreDependentMap is empty!" << "\n");
    return ssbufferMemrefs;
  }

  module->walk([&](Operation *op) {
    if (auto scopeOp = dyn_cast<scope::ScopeOp>(op)) {
      builder.setInsertionPoint(scopeOp);
      auto loc = scopeOp.getLoc();
      auto zeroConst = builder.create<arith::ConstantOp>(
          scopeOp->getLoc(), i32Type, builder.getIntegerAttr(i32Type, 0));

      for (int i = 0; i < numBuffers; i++) {
        auto memref0 =
            createPointerCastOp(builder, loc, i * VALUE_SSBUF_OFFSET);
        auto memref1 = createPointerCastOp(
            builder, loc, VECTOR_SSBUF_OFFSET + i * VALUE_SSBUF_OFFSET);

        builder.create<memref::StoreOp>(scopeOp->getLoc(), zeroConst, memref0);
        builder.create<memref::StoreOp>(scopeOp->getLoc(), zeroConst, memref1);

        ssbufferVec0Memrefs.push_back(memref0.getResult());
        ssbufferVec1Memrefs.push_back(memref1.getResult());
      }
      return mlir::WalkResult::interrupt();
    }
    return mlir::WalkResult::advance();
  });

  ssbufferMemrefs.push_back(ssbufferVec0Memrefs);
  ssbufferMemrefs.push_back(ssbufferVec1Memrefs);
  return ssbufferMemrefs;
}

// Collect dependency buffer
void UpdateConditionInfoPass::collectDependencyBuffers(
    ModuleOp module, SmallVector<Operation *> &mainLoopOps,
    DenseMap<int, DenseMap<Operation *, SmallVector<Operation *>>>
        &crossCoreBuffers,
    DenseMap<Operation *,
             DenseMap<int, DenseMap<Operation *, SmallVector<Operation *>>>>
        &intraCoreBuffersMap) {
  // Collect crossCoreBuffers by traversing module in deterministic order
  int crossCoreIdx = 0;
  module.walk([&](Operation *op) {
    // Collect crossCoreBuffers for this op
    auto it = info->crossCoreDependentMap.find(op);
    if (it != info->crossCoreDependentMap.end()) {
      for (SmallVector<Operation *> &producers : it->second) {
        crossCoreBuffers[crossCoreIdx][op] = producers;
        crossCoreIdx++;
      }
    }

    return WalkResult::advance();
  });

  // Collect intraCoreBuffers for all main_loop for/while ops
  for (Operation *loopOp : mainLoopOps) {
    if (info->intraCoreDependentMap.count(loopOp)) {
      auto &loopDeps = info->intraCoreDependentMap[loopOp];
      DenseMap<int, DenseMap<Operation *, SmallVector<Operation *>>>
          intraCoreBuffers;
      int intraCoreIdx = 0;
      for (auto &entry : loopDeps) {
        intraCoreBuffers[intraCoreIdx][entry.first] = entry.second;
        intraCoreIdx++;
      }
      intraCoreBuffersMap[loopOp] = intraCoreBuffers;
    }
  }
}

// Helper: Find the tcb group id that contains op
// Returns the group id if found, -1 otherwise
static int findTcbGroupId(
    Operation *op,
    DenseMap<int, SmallVector<Operation *>> &tightlyCoupledBufferGroups) {
  for (auto &tcbEntry : tightlyCoupledBufferGroups) {
    if (llvm::is_contained(tcbEntry.second, op)) {
      return tcbEntry.first;
    }
  }
  return UPDATE_CONDITION_INFO_FAILED;
}

// Helper: Add all equivalent ops from tcbOps to ops (excluding op itself)
int addEquivalentOps(Operation *op, SmallVector<Operation *> &tcbOps,
                     SmallVector<Operation *> &ops) {
  int ret = -1;
  for (Operation *equivOp : tcbOps) {
    if (equivOp != op && !llvm::is_contained(ops, equivOp)) {
      ret = 0;
      ops.push_back(equivOp);
    }
  }
  return ret;
}

int UpdateConditionInfoPass::buildIdxToVarMap(
    Operation *loopOp,
    const DenseMap<int, DenseMap<Operation *, SmallVector<Operation *>>>
        &intraCoreBuffers,
    DenseMap<int, Value> &idxToVar) {
  int varIdx = 0;
  MutableArrayRef<BlockArgument> regionIterArgs;
  if (getLoopRegionIterArgs(loopOp, regionIterArgs) ==
      UPDATE_CONDITION_INFO_FAILED)
    return UPDATE_CONDITION_INFO_FAILED;
  int iterArgNum = static_cast<int>(regionIterArgs.size());

  const auto &innerDepIndices = info->innerDepConds[loopOp];
  if (innerDepIndices.size() < intraCoreBuffers.size()) {
    LDBG("Not enough inner dependency condition indices: assigned "
         << innerDepIndices.size() << ", expected " << intraCoreBuffers.size()
         << ", loopOp=" << *loopOp << "\n");
    return UPDATE_CONDITION_INFO_FAILED;
  }

  for (const auto &entry : intraCoreBuffers) {
    int idx = entry.first;

    int argIdx = innerDepIndices[varIdx];
    if (argIdx < 0 || argIdx >= iterArgNum) {
      LDBG("Invalid inner dependency arg index: "
           << argIdx << ", iter args " << iterArgNum << ", loopOp=" << *loopOp
           << "\n");
      return UPDATE_CONDITION_INFO_FAILED;
    }

    idxToVar[idx] = regionIterArgs[argIdx];
    LDBG("Assign intraCore buffer group " << idx << " to iter arg index "
                                          << argIdx << "\n");
    varIdx++;
  }

  LDBG("Assigned " << idxToVar.size() << " intraCore condition variables."
                   << "\n");
  return UPDATE_CONDITION_INFO_SUCCESS;
}

// Helper function to build buffer dependency mappings (fully Operation* based)
// Outputs two reverse lookup tables for O(1) lookup during IR walk:
//   - consumerToGroups: consumer Op -> [groupIdx] (one op may consume several
//     groups)
//   - producerToGroups: producer Op -> [groupIdx]
static int buildBufferDependencyMappings(
    DenseMap<int, DenseMap<Operation *, SmallVector<Operation *>>> &buffers,
    DenseMap<Operation *, SmallVector<int>> &consumerToGroups,
    DenseMap<Operation *, SmallVector<int>> &outputToGroups) {
  for (auto &[groupIdx, deps] : buffers) {
    for (auto &[consumer, producers] : deps) {
      consumerToGroups[consumer].push_back(groupIdx);

      for (Operation *producer : producers) {
        outputToGroups[producer].push_back(groupIdx);
      }
    }
  }
  return UPDATE_CONDITION_INFO_SUCCESS;
}

// getInputOutputValues - Analyze the input/output buffer groups used in a
// single ifOp This function traverses all operations within ifOp, identifies
// FixpipeOp, CopyOp and MaterializeInDestinationOp, and extracts cross-core and
// intra-core buffer group indices from their operands and yield values.

// Data structure description
//   - crossCoreBuffers: {groupIdx -> (consumer -> [producers])}
//   - intraCoreBuffers: {groupIdx -> (consumer -> [producers])}
//   - crossCoreInputValues: list of cross-core group indices read by this ifOp
//   - crossCoreOutputValues: list of cross-core group indices written by this
//   ifOp
//   - intraCoreInputValues: list of intra-core group indices read by this ifOp
//   - intraCoreOutputValues: list of intra-core group indices written by this
//   ifOp

// Processing flow
//   1. Build reverse mapping table from Value to groupIdx
//   2. Traverse Fixpipe/Copy/MaterializeInDestination ops in ifOp, collect
//   input/output groups
//   3. Traverse ifOp.thenYield() operands, collect yield output groups
//   4. Deduplicate and output four groups of index values
int UpdateConditionInfoPass::getInputOutputValues(
    scf::IfOp ifOp,
    DenseMap<int, DenseMap<Operation *, SmallVector<Operation *>>>
        crossCoreBuffers,
    DenseMap<int, DenseMap<Operation *, SmallVector<Operation *>>>
        intraCoreBuffers,
    SmallVector<int> &crossCoreInputValues,
    SmallVector<int> &crossCoreOutputValues,
    SmallVector<int> &intraCoreInputValues,
    SmallVector<int> &intraCoreOutputValues) {
  DenseSet<int> crossCoreInputSet;
  DenseSet<int> crossCoreOutputSet;
  DenseSet<int> intraCoreInputSet;
  DenseSet<int> intraCoreOutputSet;

  // Build output mappings for cross-core and intra-core
  // Same producer/output can be used by multiple consumers/inputs, so we need
  // to track all related groups
  DenseMap<Operation *, SmallVector<int>> crossCoreOutputToGroups;
  DenseMap<Operation *, SmallVector<int>> intraCoreOutputToGroups;

  // Add consumer mappings for input dependency identification
  DenseMap<Operation *, SmallVector<int>> crossCoreConsumerToGroups;
  DenseMap<Operation *, SmallVector<int>> intraCoreConsumerToGroups;

  // Build cross-core mappings
  if (buildBufferDependencyMappings(crossCoreBuffers, crossCoreConsumerToGroups,
                                    crossCoreOutputToGroups) ==
      UPDATE_CONDITION_INFO_FAILED) {
    return UPDATE_CONDITION_INFO_FAILED;
  }

  // Build intra-core mappings
  if (buildBufferDependencyMappings(intraCoreBuffers, intraCoreConsumerToGroups,
                                    intraCoreOutputToGroups) ==
      UPDATE_CONDITION_INFO_FAILED) {
    return UPDATE_CONDITION_INFO_FAILED;
  }

  // collect input/output groups
  ifOp.walk([&](Operation *op) {
    if (op == ifOp)
      return WalkResult::advance();

    // Check if this op is a consumer (for input dependency)
    if (crossCoreConsumerToGroups.count(op)) {
      for (int idx : crossCoreConsumerToGroups[op])
        crossCoreInputSet.insert(idx);
    }
    if (intraCoreConsumerToGroups.count(op)) {
      for (int idx : intraCoreConsumerToGroups[op])
        intraCoreInputSet.insert(idx);
    }

    // Check if this op is a producer (for output dependency)
    if (crossCoreOutputToGroups.count(op)) {
      for (int idx : crossCoreOutputToGroups[op]) {
        crossCoreOutputSet.insert(idx);
      }
    }
    if (intraCoreOutputToGroups.count(op)) {
      for (int idx : intraCoreOutputToGroups[op]) {
        intraCoreOutputSet.insert(idx);
      }
    }
    return WalkResult::advance();
  });

  crossCoreInputValues.assign(crossCoreInputSet.begin(),
                              crossCoreInputSet.end());
  crossCoreOutputValues.assign(crossCoreOutputSet.begin(),
                               crossCoreOutputSet.end());
  intraCoreInputValues.assign(intraCoreInputSet.begin(),
                              intraCoreInputSet.end());
  intraCoreOutputValues.assign(intraCoreOutputSet.begin(),
                               intraCoreOutputSet.end());

  LDBG("==== Cross Core & Intra Core Values ====" << "\n");
  LLVM_DEBUG(
      logConditionGroupIndices("crossCoreInputValues: ", crossCoreInputValues));
  LLVM_DEBUG(logConditionGroupIndices("crossCoreOutputValues: ",
                                      crossCoreOutputValues));
  LLVM_DEBUG(
      logConditionGroupIndices("intraCoreInputValues: ", intraCoreInputValues));
  LLVM_DEBUG(logConditionGroupIndices("intraCoreOutputValues: ",
                                      intraCoreOutputValues));

  return UPDATE_CONDITION_INFO_SUCCESS;
}

Value UpdateConditionInfoPass::getVarValue(Operation *loopOp, int varIndex) {
  if (!info->innerDepConds.count(loopOp))
    return Value();
  SmallVector<int> &innerDepIndices = info->innerDepConds[loopOp];
  if (varIndex < (int)innerDepIndices.size()) {
    int argIdx = innerDepIndices[varIndex];
    Value arg;
    if (getLoopRegionIterArg(loopOp, argIdx, arg) ==
        UPDATE_CONDITION_INFO_FAILED)
      return Value();
    return arg;
  }
  return Value();
}

// Build the information of the producer group.
int UpdateConditionInfoPass::buildOutputGroups(
    SmallVector<int> &intraCoreOutputValues,
    DenseMap<int, DenseMap<Operation *, SmallVector<Operation *>>>
        &intraCoreBuffers,
    DenseMap<int, Value> &idxToVar,
    SmallVector<OutputGroupInfo> &outputGroups) {
  outputGroups.clear();

  for (int idx : intraCoreOutputValues) {
    auto bufferIt = intraCoreBuffers.find(idx);
    if (bufferIt == intraCoreBuffers.end()) {
      LDBG("Failed to build output groups: no buffer entry for intraCore "
           "output group "
           << idx << "." << "\n");
      return UPDATE_CONDITION_INFO_FAILED;
    }

    auto varIt = idxToVar.find(idx);
    if (varIt == idxToVar.end()) {
      LDBG("Failed to build output groups: no control variable for intraCore "
           "output group "
           << idx << "." << "\n");
      return UPDATE_CONDITION_INFO_FAILED;
    }
    Value var = varIt->second;

    for (auto &entry : bufferIt->second) {
      SmallVector<Operation *> outputOps;
      for (Operation *producer : entry.second) {
        outputOps.push_back(producer);
      }
      if (outputOps.empty())
        continue;

      bool flag = true;
      for (auto &outputGroup : outputGroups) {
        if (outputGroup.outputs == outputOps) {
          outputGroup.inputVars.push_back(var);
          flag = false;
          break;
        }
      }
      if (flag) {
        OutputGroupInfo groupInfo;
        groupInfo.outputs = outputOps;
        groupInfo.inputVars.push_back(var);
        outputGroups.push_back(groupInfo);
      }
    }
  }

  LDBG("Built " << outputGroups.size() << " intraCore output groups." << "\n");
  return UPDATE_CONDITION_INFO_SUCCESS;
}

// Select corresponding SSBuffer ptr based on ifblock running on vector/cube
// core
Value UpdateConditionInfoPass::getSSBufferMemref(
    bool isAIC, int groupIdx, int ptrSetIdx,
    DenseMap<int, Value> &VectorSSBufferPtrs,
    ArrayRef<SmallVector<Value>> ssbufferPtrs) {
  if (isAIC) {
    return ssbufferPtrs[ptrSetIdx][groupIdx];
  } else {
    return VectorSSBufferPtrs[groupIdx];
  }
}

// Compute pointers for VECTOR core SSBuffer
std::optional<DenseMap<int, Value>>
UpdateConditionInfoPass::computeVectorSSBufferMemrefs(
    OpBuilder &builder, Location loc, Operation *scopeOp,
    SmallVector<int> crossCoreInputValues,
    SmallVector<int> crossCoreOutputValues) {
  if (!scopeOp) {
    LDBG("Scope Op is null pointer!");
    return std::nullopt;
  }

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

  DenseMap<int, Value> vectorSSBufferMemrefs;

  builder.setInsertionPointToStart(&scopeOp->getRegion(0).front());
  int vec1Offset = 1024;
  Value vec1OffsetValue = builder.create<arith::ConstantIntOp>(
      loc, VECTOR_SSBUF_OFFSET, ADDR_INT_TYPE);
  auto subIdOp = builder.create<GetSubBlockIdxOp>(
      loc, builder.getIntegerType(ADDR_INT_TYPE));
  Value ssbAddrOffset =
      builder.create<arith::MulIOp>(loc, subIdOp, vec1OffsetValue);

  for (int groupIdx : allGroupIndices) {
    auto ssbBaseAddr = builder.create<arith::ConstantIntOp>(
        loc, groupIdx * VALUE_SSBUF_OFFSET, ADDR_INT_TYPE);
    auto ssbAddr =
        builder.create<arith::AddIOp>(loc, ssbBaseAddr, ssbAddrOffset);
    Value memref = builder.create<PointerCastOp>(
        loc, getSsbufMemrefType(builder), ssbAddr.getResult());
    vectorSSBufferMemrefs[groupIdx] = memref;
  }

  return vectorSSBufferMemrefs;
}

template <typename FuncTy>
auto UpdateConditionInfoPass::createSsbufLoads(
    OpBuilder &builder, Location loc, bool isAIC, int groupIdx,
    DenseMap<int, Value> &vectorSSBufferMemrefs,
    llvm::ArrayRef<SmallVector<Value>> ssbufferMemrefs, FuncTy &&pred) {
  using ResTy = std::invoke_result_t<FuncTy, memref::LoadOp, int>;
  SmallVector<ResTy, kVectorCount> res;
  size_t resNum = isAIC ? kVectorCount : 1;
  for (int ptrSetIdx : llvm::seq(resNum)) {
    auto loadOp = builder.create<memref::LoadOp>(
        loc,
        getSSBufferMemref(isAIC, groupIdx, ptrSetIdx, vectorSSBufferMemrefs,
                          ssbufferMemrefs),
        ValueRange{});
    auto markOp = builder.create<annotation::MarkOp>(loc, loadOp);
    markOp->setAttr(kMemrefExtVolatile, builder.getUnitAttr());
    res.push_back(std::invoke(pred, loadOp, ptrSetIdx));
  }
  return res;
}

// Part 2: Add cross-core conditions
Value UpdateConditionInfoPass::addCrossCoreConditions(
    OpBuilder &builder, Location loc, SmallVector<int> crossCoreInputValues,
    SmallVector<int> crossCoreOutputValues,
    DenseMap<int, DenseMap<Operation *, SmallVector<Operation *>>>
        &crossCoreBuffers,
    bool isAIC, Value zeroConst, DenseMap<int, Value> &VectorSSBufferPtrs,
    SmallVector<SmallVector<Value>> ssbufferPtrs) {
  Value conditions = nullptr;

  auto combineCondition = [&](Value newCond) {
    if (conditions) {
      conditions = builder.create<arith::AndIOp>(loc, conditions, newCond);
    } else {
      conditions = newCond;
    }
  };

  for (int inputGroupIdx : crossCoreInputValues) {
    auto conds = createSsbufLoads(
        builder, loc, isAIC, inputGroupIdx, VectorSSBufferPtrs, ssbufferPtrs,
        [&](memref::LoadOp loadOp, auto) {
          return builder.create<arith::CmpIOp>(loc, arith::CmpIPredicate::sgt,
                                               loadOp, zeroConst);
        });
    Value cond = conds[0];
    if (isAIC) {
      cond = builder.create<arith::AndIOp>(loc, conds[0], conds[1]);
    }
    combineCondition(cond);
  }

  for (int outputGroupIdx : crossCoreOutputValues) {
    int outputCount = 0;
    if (crossCoreBuffers.count(outputGroupIdx)) {
      for (auto &entry : crossCoreBuffers[outputGroupIdx]) {
        outputCount += entry.second.size();
      }
    } else {
      LDBG("outputGroupIdx " << outputGroupIdx
                             << " not found in any buffers map!" << "\n");
      return nullptr;
    }
    Value bufferNum =
        builder.create<arith::ConstantIntOp>(loc, outputCount, CONST_INT_TYPE);
    auto conds = createSsbufLoads(
        builder, loc, isAIC, outputGroupIdx, VectorSSBufferPtrs, ssbufferPtrs,
        [&](Value cond, auto) {
          return builder.create<arith::CmpIOp>(loc, arith::CmpIPredicate::slt,
                                               cond, bufferNum);
        });
    Value cond = conds[0];
    if (isAIC) {
      cond = builder.create<arith::AndIOp>(loc, conds[0], conds[1]);
    }
    combineCondition(cond);
  }

  return conditions;
}

// Part 3: Update control variables in then block
void UpdateConditionInfoPass::updateCrossCoreControlVars(
    OpBuilder &builder, Location loc, scf::IfOp ifOp,
    SmallVector<int> crossCoreInputValues,
    SmallVector<int> crossCoreOutputValues, bool isAIC, Value oneConst,
    DenseMap<int, Value> &VectorSSBufferPtrs,
    SmallVector<SmallVector<Value>> ssbufferPtrs) {
  Block *thenBlock = &ifOp.getThenRegion().front();
  auto yieldOp = cast<scf::YieldOp>(thenBlock->getTerminator());
  builder.setInsertionPoint(yieldOp);

  for (int inputGroupIdx : crossCoreInputValues) {
    createSsbufLoads(builder, loc, isAIC, inputGroupIdx, VectorSSBufferPtrs,
                     ssbufferPtrs, [&](Value loadRes, int ptrSetIdx) {
                       Value newValue = builder.create<arith::SubIOp>(
                           loc, loadRes, oneConst);
                       return builder.create<memref::StoreOp>(
                           loc, newValue,
                           getSSBufferMemref(isAIC, inputGroupIdx, ptrSetIdx,
                                             VectorSSBufferPtrs, ssbufferPtrs),
                           ValueRange{});
                     });
  }

  for (int outputGroupIdx : crossCoreOutputValues) {
    createSsbufLoads(builder, loc, isAIC, outputGroupIdx, VectorSSBufferPtrs,
                     ssbufferPtrs, [&](Value loadRes, int ptrSetIdx) {
                       Value newValue = builder.create<arith::AddIOp>(
                           loc, loadRes, oneConst);
                       return builder.create<memref::StoreOp>(
                           loc, newValue,
                           getSSBufferMemref(isAIC, outputGroupIdx, ptrSetIdx,
                                             VectorSSBufferPtrs, ssbufferPtrs),
                           ValueRange{});
                     });
  }
}

// Set the crossCore condition
int UpdateConditionInfoPass::setCrossCoreCondition(
    SmallVector<int> crossCoreInputValues,
    SmallVector<int> crossCoreOutputValues,
    DenseMap<int, DenseMap<Operation *, SmallVector<Operation *>>>
        &crossCoreBuffers,
    scf::IfOp ifOp, SmallVector<SmallVector<Value>> ssbufferPtrs,
    Value &crossCoreCond) {
  OpBuilder builder(ifOp);
  Location loc = ifOp.getLoc();

  // ========== Part 1: Preparation ==========
  // Determine whether the current ifblock is on cube or vector core
  auto aiCAttr =
      hivm::TCoreTypeAttr::get(builder.getContext(), hivm::TCoreType::CUBE);
  auto aivAttr =
      hivm::TCoreTypeAttr::get(builder.getContext(), hivm::TCoreType::VECTOR);
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
      LDBG("scope block has invalid tcore_type attribute" << "\n");
      return UPDATE_CONDITION_INFO_FAILED;
    }
  } else {
    LDBG("ifblock not in a correct scope block" << "\n");
    return UPDATE_CONDITION_INFO_FAILED;
  }

  Value zeroConst =
      builder.create<arith::ConstantIntOp>(loc, 0, CONST_INT_TYPE);
  Value oneConst = builder.create<arith::ConstantIntOp>(loc, 1, CONST_INT_TYPE);

  // If ifblock is on vector core, compute the required SSBuffer ptrs for vector
  // side
  DenseMap<int, Value> VectorSSBufferPtrs;
  if (!isAIC) {
    auto result = computeVectorSSBufferMemrefs(
        builder, loc, scopeOp, crossCoreInputValues, crossCoreOutputValues);
    if (!result) {
      LDBG("computeVectorSSBufferPtrs failed!");
      return UPDATE_CONDITION_INFO_FAILED;
    }
    VectorSSBufferPtrs = std::move(*result);
  }

  builder.setInsertionPoint(ifOp);

  // ========== Part 2: Add cross-core conditions ==========
  crossCoreCond = addCrossCoreConditions(
      builder, loc, crossCoreInputValues, crossCoreOutputValues,
      crossCoreBuffers, isAIC, zeroConst, VectorSSBufferPtrs, ssbufferPtrs);

  // ========== Part 3: Update control variables ==========
  updateCrossCoreControlVars(builder, loc, ifOp, crossCoreInputValues,
                             crossCoreOutputValues, isAIC, oneConst,
                             VectorSSBufferPtrs, ssbufferPtrs);

  return UPDATE_CONDITION_INFO_SUCCESS;
}

// Collect the conditions for intra-core consumer values.
void UpdateConditionInfoPass::collectIntraCoreInputConditions(
    OpBuilder &builder, Location loc, SmallVector<int> &intraCoreInputValues,
    DenseMap<int, Value> &idxToVar, SmallVector<Value> &conditions,
    DenseSet<Value> &usedVarsSet,
    DenseMap<Value, VarUpdateType> &varUpdateTypes) {
  if (intraCoreInputValues.empty()) {
    LDBG("No intraCore input conditions to collect." << "\n");
    return;
  }

  size_t beforeConditionNum = conditions.size();
  Value zeroConst =
      builder.create<arith::ConstantIntOp>(loc, 0, CONST_INT_TYPE);
  for (int idx : intraCoreInputValues) {
    auto varIt = idxToVar.find(idx);
    if (varIt == idxToVar.end()) {
      LDBG("Skip intraCore input group " << idx << ": no control variable."
                                         << "\n");
      continue;
    }

    Value var = varIt->second;
    Value varToUse = var;
    auto latestIt = controlVarToLatestValue.find(var);
    if (latestIt != controlVarToLatestValue.end()) {
      varToUse = latestIt->second;
    }

    Value cond = builder.create<arith::CmpIOp>(loc, arith::CmpIPredicate::sgt,
                                               varToUse, zeroConst);
    conditions.push_back(cond);
    usedVarsSet.insert(var);
    varUpdateTypes[var] = VarUpdateType::DEC;
    LDBG("Add intraCore input condition for group " << idx << "." << "\n");
  }
  LDBG("Collected " << (conditions.size() - beforeConditionNum)
                    << " intraCore input conditions." << "\n");
}

// Collect the conditions for intra-core producer values.
int UpdateConditionInfoPass::collectIntraCoreOutputConditions(
    OpBuilder &builder, Location loc,
    DenseMap<int, DenseMap<Operation *, SmallVector<Operation *>>>
        &intraCoreBuffers,
    SmallVector<int> &intraCoreOutputValues, DenseMap<int, Value> &idxToVar,
    SmallVector<Value> &conditions, DenseSet<Value> &usedVarsSet,
    DenseMap<Value, VarUpdateType> &varUpdateTypes) {
  if (intraCoreOutputValues.empty()) {
    LDBG("No intraCore output conditions to collect." << "\n");
    return UPDATE_CONDITION_INFO_SUCCESS;
  }

  size_t beforeConditionNum = conditions.size();
  SmallVector<OutputGroupInfo> outputGroups;
  if (buildOutputGroups(intraCoreOutputValues, intraCoreBuffers, idxToVar,
                        outputGroups) == UPDATE_CONDITION_INFO_FAILED) {
    return UPDATE_CONDITION_INFO_FAILED;
  }
  for (auto &group : outputGroups) {
    int size = group.outputs.size();
    Value limitVal =
        builder.create<arith::ConstantIntOp>(loc, size, CONST_INT_TYPE);
    for (Value var : group.inputVars) {
      Value varToUse = var;
      auto latestIt = controlVarToLatestValue.find(var);
      if (latestIt != controlVarToLatestValue.end()) {
        varToUse = latestIt->second;
      }

      Value cond = builder.create<arith::CmpIOp>(loc, arith::CmpIPredicate::slt,
                                                 varToUse, limitVal);
      conditions.push_back(cond);
      usedVarsSet.insert(var);
      varUpdateTypes[var] = VarUpdateType::INC;
      LDBG("Add intraCore output condition with producer limit " << size << "."
                                                                 << "\n");
    }
  }
  LDBG("Collected " << (conditions.size() - beforeConditionNum)
                    << " intraCore output conditions." << "\n");
  return UPDATE_CONDITION_INFO_SUCCESS;
}

// Build the ifOp variable mapping for the tensor iter_args
int UpdateConditionInfoPass::buildTensorIterArgIfOpVarMap(Operation *loopOp) {
  // Clear any previous data
  tensorIterArgIfOpVars.clear();

  if (!info->tensorIterArgDepsMap.count(loopOp) ||
      !info->tensorIterArgIndicesMap.count(loopOp)) {
    LDBG("Skip buildTensorIterArgIfOpVarMap: no tensor iter_args info for "
         << *loopOp << "\n");
    return UPDATE_CONDITION_INFO_SUCCESS;
  }

  auto &depsVec = info->tensorIterArgDepsMap[loopOp];
  auto &indicesMap = info->tensorIterArgIndicesMap[loopOp];

  llvm::DenseMap<scf::IfOp, llvm::DenseSet<Value>> producerVars;
  llvm::DenseMap<scf::IfOp, llvm::DenseSet<Value>> consumerVars;

  for (auto &depEntry : depsVec) {
    Value origIterArg = depEntry.iterArg;
    TensorIterArgIfOpRelation &relation = depEntry;

    if (!indicesMap.count(origIterArg)) {
      LDBG("[Error]: origIterArg not found in indicesMap, loopOp=" << *loopOp
                                                                   << "\n");
      return UPDATE_CONDITION_INFO_FAILED;
    }
    SmallVector<int> &argIndices = indicesMap[origIterArg];

    if (relation.consumers.size() != argIndices.size()) {
      LDBG("[Error]: consumers size mismatch: "
           << relation.consumers.size() << " vs " << argIndices.size()
           << ", loopOp=" << *loopOp << "\n");
      return UPDATE_CONDITION_INFO_FAILED;
    }

    // Establish a mapping (one-to-one) from consumers to variables
    // for: region iter args; while-do: after-region args.
    llvm::DenseMap<scf::IfOp, Value> consumerToVar;
    for (size_t i = 0; i < relation.consumers.size(); ++i) {
      scf::IfOp consumer = relation.consumers[i];
      Value var;
      if (getLoopRegionIterArg(loopOp, argIndices[i], var) ==
          UPDATE_CONDITION_INFO_FAILED) {
        LDBG("[Error]: invalid tensor iter_arg index "
             << argIndices[i] << ", loopOp=" << *loopOp
             << ", consumer=" << consumer << "\n");
        return UPDATE_CONDITION_INFO_FAILED;
      }
      consumerToVar[consumer] = var;
    }

    // Add all the variables of the consumers that depend on the producer
    if (relation.producer) {
      for (auto &[consumer, var] : consumerToVar) {
        producerVars[relation.producer].insert(var);
      }
    }

    // Add all the variables of the consumers that depend on each producer
    for (auto &[consumer, var] : consumerToVar) {
      consumerVars[consumer].insert(var);
    }
  }

  // Convert the temporary data structure to tensorIfOpVarMap
  for (auto &[producer, vars] : producerVars) {
    auto &ifOpVars = tensorIterArgIfOpVars[producer];
    for (Value var : vars) {
      ifOpVars.producerVars.push_back(var);
    }
  }

  for (auto &[consumer, vars] : consumerVars) {
    auto &ifOpVars = tensorIterArgIfOpVars[consumer];
    for (Value var : vars) {
      ifOpVars.consumerVars.push_back(var);
    }
  }
  return UPDATE_CONDITION_INFO_SUCCESS;
}

// Collect the conditions for tensor iter arg consumer values.
void UpdateConditionInfoPass::collectTensorIterArgInputConditions(
    OpBuilder &builder, Location loc, scf::IfOp ifOp,
    SmallVector<Value> &conditions, DenseSet<Value> &usedVarsSet,
    DenseMap<Value, VarUpdateType> &varUpdateTypes) {
  if (!tensorIterArgIfOpVars.count(ifOp)) {
    return;
  }

  auto &ifOpVars = tensorIterArgIfOpVars[ifOp];
  for (Value var : ifOpVars.consumerVars) {
    Value varToUse = var;
    auto latestIt = controlVarToLatestValue.find(var);
    if (latestIt != controlVarToLatestValue.end()) {
      varToUse = latestIt->second;
    }

    Value oneConst =
        builder.create<arith::ConstantIntOp>(loc, 1, CONST_INT_TYPE);
    Value cond = builder.create<arith::CmpIOp>(loc, arith::CmpIPredicate::eq,
                                               varToUse, oneConst);
    conditions.push_back(cond);
    usedVarsSet.insert(var);
    varUpdateTypes[var] = VarUpdateType::DEC;
    LDBG("Add tensor iter arg consumer condition for var.\n");
  }
}

// Collect the conditions for tensor iter arg producer values.
void UpdateConditionInfoPass::collectTensorIterArgOutputConditions(
    OpBuilder &builder, Location loc, scf::IfOp ifOp,
    SmallVector<Value> &conditions, DenseSet<Value> &usedVarsSet,
    DenseMap<Value, VarUpdateType> &varUpdateTypes) {
  if (!tensorIterArgIfOpVars.count(ifOp)) {
    return;
  }

  auto &ifOpVars = tensorIterArgIfOpVars[ifOp];
  for (Value var : ifOpVars.producerVars) {
    Value varToUse = var;
    auto latestIt = controlVarToLatestValue.find(var);
    if (latestIt != controlVarToLatestValue.end()) {
      varToUse = latestIt->second;
    }

    Value zeroConst =
        builder.create<arith::ConstantIntOp>(loc, 0, CONST_INT_TYPE);
    Value cond = builder.create<arith::CmpIOp>(loc, arith::CmpIPredicate::eq,
                                               varToUse, zeroConst);
    conditions.push_back(cond);
    usedVarsSet.insert(var);
    varUpdateTypes[var] = VarUpdateType::INC;
    LDBG("Add tensor iter arg producer condition (var == 0) and +1 update for "
         "var.\n");
  }
}

// Set the intraCore condition.
int UpdateConditionInfoPass::setIntraCoreCondition(
    ModuleOp module, scf::IfOp ifOp,
    DenseMap<int, DenseMap<Operation *, SmallVector<Operation *>>>
        &intraCoreBuffers,
    SmallVector<int> &intraCoreInputValues,
    SmallVector<int> &intraCoreOutputValues, DenseMap<int, Value> &idxToVar,
    DenseMap<Value, VarUpdateType> &varUpdateTypes, Value &intraCoreCond) {
  LDBG("Enter set intraCore condition." << "\n");
  intraCoreCond = Value();
  OpBuilder builder(ifOp.getContext());
  builder.setInsertionPoint(ifOp);
  Location loc = ifOp.getLoc();

  SmallVector<Value> conditions;
  DenseSet<Value> usedVarsSet;
  LDBG("Collect intraCore conditions: inputs "
       << intraCoreInputValues.size() << ", outputs "
       << intraCoreOutputValues.size() << "\n");
  // Collect the conditions for intra-core consumer values.
  collectIntraCoreInputConditions(builder, loc, intraCoreInputValues, idxToVar,
                                  conditions, usedVarsSet, varUpdateTypes);
  // Collect the conditions for intra-core producer values.
  if (collectIntraCoreOutputConditions(
          builder, loc, intraCoreBuffers, intraCoreOutputValues, idxToVar,
          conditions, usedVarsSet,
          varUpdateTypes) == UPDATE_CONDITION_INFO_FAILED) {
    return UPDATE_CONDITION_INFO_FAILED;
  }
  // Collect tensor iter_args conditions
  collectTensorIterArgInputConditions(builder, loc, ifOp, conditions,
                                      usedVarsSet, varUpdateTypes);
  collectTensorIterArgOutputConditions(builder, loc, ifOp, conditions,
                                       usedVarsSet, varUpdateTypes);

  if (!conditions.empty()) {
    intraCoreCond = conditions[0];
    for (size_t i = 1; i < conditions.size(); ++i) {
      intraCoreCond =
          builder.create<arith::AndIOp>(loc, intraCoreCond, conditions[i]);
    }
  }

  currentUsedVars.clear();
  for (Value var : usedVarsSet) {
    currentUsedVars.push_back(var);
  }
  LDBG("Built " << conditions.size() << " intraCore conditions using "
                << currentUsedVars.size() << " control variables." << "\n");

  LDBG("Exit set intraCore condition." << "\n");
  return UPDATE_CONDITION_INFO_SUCCESS;
}

// Set the FlowOpt extra condition for the third if block in the DAG
int UpdateConditionInfoPass::setFlowOptCondition(scf::IfOp currentIfOp,
                                                 Operation *loopOp,
                                                 Value &flowOptCond) {
  auto forOp = dyn_cast<scf::ForOp>(loopOp);
  if (!forOp) {
    flowOptCond = nullptr;
    return UPDATE_CONDITION_INFO_SUCCESS;
  }

  // Check if current ifOp is a target node (third node) in flowOptIfOpPairs
  if (!info->flowOptIfOpPairs.count(currentIfOp)) {
    LDBG("Current ifOp is not a flowOpt target node, skip.");
    flowOptCond = nullptr;
    return UPDATE_CONDITION_INFO_SUCCESS;
  }

  // Check if flowOpt condition is needed based on buffer counts
  // - Cross-core buffer count > CROSS_CORE_BUFFER_COUNT_THRESHOLD
  // - Intra-core buffer count > INTRA_CORE_BUFFER_COUNT_THRESHOLD
  if (info->crossCoreBufferCount <= CROSS_CORE_BUFFER_COUNT_THRESHOLD ||
      info->intraCoreBufferCount <= INTRA_CORE_BUFFER_COUNT_THRESHOLD) {
    flowOptCond = nullptr;
    return UPDATE_CONDITION_INFO_SUCCESS;
  }

  // Get the start node
  scf::IfOp sourceIfOp = info->flowOptIfOpPairs[currentIfOp];
  if (!info->cntArgs.count(sourceIfOp)) {
    LDBG("[Error] Start node has no counter in cntArgs, cannot build flowOpt "
         "condition. currentIfOp="
         << currentIfOp << ", sourceIfOp=" << sourceIfOp << ", forOp=" << forOp
         << "\n");
    return UPDATE_CONDITION_INFO_FAILED;
  }

  // Create builder and location
  OpBuilder builder(currentIfOp);
  Location loc = currentIfOp.getLoc();

  Value counter = info->cntArgs[sourceIfOp];
  Value lowerBound = forOp.getLowerBound();
  Value upperBound = forOp.getUpperBound();
  Value step = forOp.getStep();

  // Build condition: counter >= upperBound
  Value cond1 = builder.create<arith::CmpIOp>(loc, arith::CmpIPredicate::sge,
                                              counter, upperBound);

  // Build condition: counter >= lowerBound + step * opt_num
  // opt_num = min(intraCoreBufferCount - 1, crossCoreBufferCount)
  int optInt =
      std::min(info->intraCoreBufferCount - 1, info->crossCoreBufferCount);

  auto stepIntType = dyn_cast<IntegerType>(step.getType());
  if (!stepIntType) {
    LDBG("[Error] forOp step is expected to be an integer type, got "
         << step.getType() << ", forOp=" << forOp << "\n");
    return UPDATE_CONDITION_INFO_FAILED;
  }
  Value optNum =
      builder.create<arith::ConstantIntOp>(loc, optInt, stepIntType.getWidth());
  Value optOffset = builder.create<arith::MulIOp>(loc, step, optNum);
  Value lowerPlusOffset =
      builder.create<arith::AddIOp>(loc, lowerBound, optOffset);
  Value cond2 = builder.create<arith::CmpIOp>(loc, arith::CmpIPredicate::sge,
                                              counter, lowerPlusOffset);

  // Combine: cond1 OR cond2
  flowOptCond = builder.create<arith::OrIOp>(loc, cond1, cond2);

  return UPDATE_CONDITION_INFO_SUCCESS;
}

// Update DAG nodes after ifOp replacement
void UpdateConditionInfoPass::updateDAGAfterIfOpReplacement(scf::IfOp oldIfOp,
                                                            scf::IfOp newIfOp) {
  // 1. Update ifBlockDAG
  if (info->ifBlockDAG.count(oldIfOp)) {
    auto consumers = info->ifBlockDAG[oldIfOp];
    info->ifBlockDAG.erase(oldIfOp);
    info->ifBlockDAG[newIfOp] = consumers;
  }

  for (auto &entry : info->ifBlockDAG) {
    for (auto &edge : entry.second) {
      if (edge.first == oldIfOp) {
        edge.first = newIfOp;
      }
    }
  }

  // 2. Update flowOptIfOpPairs
  if (info->flowOptIfOpPairs.count(oldIfOp)) {
    auto source = info->flowOptIfOpPairs[oldIfOp];
    info->flowOptIfOpPairs.erase(oldIfOp);
    info->flowOptIfOpPairs[newIfOp] = source;
  }

  for (auto &entry : info->flowOptIfOpPairs) {
    if (entry.second == oldIfOp) {
      entry.second = newIfOp;
    }
  }
}

// Update the mapping of control variables to their latest values
void UpdateConditionInfoPass::updateControlVarToLatestValue(scf::IfOp newIfOp,
                                                            scf::IfOp oldIfOp,
                                                            bool hasCounter,
                                                            Value counter) {
  if (currentUsedVars.empty() && !hasCounter) {
    LDBG("No control variable latest values to update." << "\n");
    return;
  }

  size_t origResultCount = oldIfOp.getNumResults();

  for (size_t i = 0; i < currentUsedVars.size(); ++i) {
    Value var = currentUsedVars[i];
    Value newValue = newIfOp.getResult(origResultCount + i);
    controlVarToLatestValue[var] = newValue;
    LDBG("Record latest intraCore control value at result index "
         << (origResultCount + i) << "." << "\n");
  }

  if (hasCounter) {
    size_t counterResultIdx = origResultCount + currentUsedVars.size();
    Value newCounterValue = newIfOp.getResult(counterResultIdx);
    controlVarToLatestValue[counter] = newCounterValue;
    LDBG("Record latest counter value at result index " << counterResultIdx
                                                        << "." << "\n");
  }
  LDBG("[DEBUG] controlVarToLatestValue size: "
       << controlVarToLatestValue.size() << "\n");

  for (auto &entry : controlVarToLatestValue) {
    LDBG("[DEBUG]   key = " << entry.first
                            << "  -->  new value = " << entry.second << "\n");
  }
}

// Update the yield in the forOp
int UpdateConditionInfoPass::updateForOpYield(scf::ForOp forOp) {
  if (controlVarToLatestValue.empty()) {
    LDBG("Failed to update forOp yield: no latest control variable values, "
         << "forOp=" << forOp << "\n");
    return UPDATE_CONDITION_INFO_FAILED;
  }

  Location loc = forOp.getLoc();
  Block *forBody = forOp.getBody();
  auto yieldOp = dyn_cast<scf::YieldOp>(forBody->getTerminator());
  if (!yieldOp) {
    LDBG("Failed to update forOp yield: terminator is not scf.yield, forOp="
         << forOp << "\n");
    return UPDATE_CONDITION_INFO_FAILED;
  }

  SmallVector<Value> newYieldOperands(yieldOp.getOperands().begin(),
                                      yieldOp.getOperands().end());
  if (newYieldOperands.size() != forOp.getNumRegionIterArgs()) {
    LDBG("Failed to update forOp yield: yield operands "
         << newYieldOperands.size() << ", iter args "
         << forOp.getNumRegionIterArgs() << ", forOp=" << forOp << "\n");
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
      LDBG("Failed to update forOp yield: control variable is not a region "
           "iter arg, forOp="
           << forOp << ", var=" << origVar << "\n");
      return UPDATE_CONDITION_INFO_FAILED;
    }
    newYieldOperands[it->second] = latestValue;
    LDBG("Update forOp yield operand index " << it->second << "\n");
  }

  OpBuilder yieldBuilder(yieldOp);
  yieldBuilder.create<scf::YieldOp>(loc, newYieldOperands);
  yieldOp.erase();
  LDBG("Updated forOp yield with " << controlVarToLatestValue.size()
                                   << " latest control values." << "\n");
  return UPDATE_CONDITION_INFO_SUCCESS;
}

int UpdateConditionInfoPass::updateWhileOpYield(scf::WhileOp whileOp) {
  LDBG("Enter update whileOp yield " << "\n");
  if (controlVarToLatestValue.empty()) {
    LDBG("Skip update whileOp yield: no latest control variable values, "
         << "whileOp=" << whileOp << "\n");
    return UPDATE_CONDITION_INFO_SUCCESS;
  }

  Location loc = whileOp.getLoc();
  Block *afterBody = whileOp.getAfterBody();
  auto yieldOp = dyn_cast<scf::YieldOp>(afterBody->getTerminator());
  if (!yieldOp) {
    LDBG("Failed to update whileOp yield: terminator is not scf.yield, "
         << "whileOp=" << whileOp << "\n");
    return UPDATE_CONDITION_INFO_FAILED;
  }

  SmallVector<Value> newYieldOperands(yieldOp.getOperands().begin(),
                                      yieldOp.getOperands().end());
  auto afterArgs = whileOp.getAfterArguments();
  if (newYieldOperands.size() != afterArgs.size()) {
    LDBG("Failed to update whileOp yield: yield operands "
         << newYieldOperands.size() << ", after args " << afterArgs.size()
         << ", whileOp=" << whileOp << "\n");
    return UPDATE_CONDITION_INFO_FAILED;
  }

  DenseMap<Value, unsigned> iterArgToIndex;
  for (auto [idx, arg] : llvm::enumerate(afterArgs))
    iterArgToIndex[arg] = idx;

  for (auto &entry : controlVarToLatestValue) {
    Value origVar = entry.first;
    Value latestValue = entry.second;
    auto it = iterArgToIndex.find(origVar);
    if (it == iterArgToIndex.end()) {
      LDBG("Failed to update whileOp yield: control variable is not an after "
           "region iter arg, whileOp="
           << whileOp << ", var=" << origVar << "\n");
      return UPDATE_CONDITION_INFO_FAILED;
    }
    newYieldOperands[it->second] = latestValue;
    LDBG("Update whileOp yield operand index " << it->second << "\n");
  }

  OpBuilder yieldBuilder(yieldOp);
  yieldBuilder.create<scf::YieldOp>(loc, newYieldOperands);
  yieldOp.erase();
  LDBG("Updated whileOp yield with " << controlVarToLatestValue.size()
                                     << " latest control values." << "\n");
  LDBG("Exit update whileOp yield " << "\n");
  return UPDATE_CONDITION_INFO_SUCCESS;
}

int UpdateConditionInfoPass::updateLoopYield(Operation *loopOp) {
  if (auto forOp = dyn_cast<scf::ForOp>(loopOp))
    return updateForOpYield(forOp);
  if (auto whileOp = dyn_cast<scf::WhileOp>(loopOp))
    return updateWhileOpYield(whileOp);
  LDBG("updateLoopYield expects scf.for or scf.while, got " << *loopOp << "\n");
  return UPDATE_CONDITION_INFO_FAILED;
}

int UpdateConditionInfoPass::collectSSBufferIfOps(
    Operation *loopOp, SmallVector<scf::IfOp> &ifOps) {
  auto collectIfOps = [&](Operation *op) -> WalkResult {
    if (!op->hasAttr(kIf)) {
      return WalkResult::advance();
    }

    auto ifOp = dyn_cast<scf::IfOp>(op);
    if (!ifOp) {
      LDBG("Found unsupported ssbuffer if op: "
           << op->getName() << " under loopOp=" << *loopOp << "\n");
      return WalkResult::interrupt();
    }

    ifOps.push_back(ifOp);
    return WalkResult::advance();
  };

  WalkResult ifWalkResult;
  if (auto forOp = dyn_cast<scf::ForOp>(loopOp)) {
    ifWalkResult = forOp.walk(collectIfOps);
  } else if (auto whileOp = dyn_cast<scf::WhileOp>(loopOp)) {
    // while-do: only after-region.
    ifWalkResult = whileOp.getAfterBody()->walk(collectIfOps);
  } else {
    LDBG("collectSSBufferIfOps expects scf.for or scf.while, got " << *loopOp
                                                                   << "\n");
    return UPDATE_CONDITION_INFO_FAILED;
  }

  if (ifWalkResult.wasInterrupted()) {
    return UPDATE_CONDITION_INFO_FAILED;
  }
  return UPDATE_CONDITION_INFO_SUCCESS;
}

int UpdateConditionInfoPass::validateBlockCounters(Operation *loopOp,
                                                   size_t ifOpCount) {
  auto forOp = dyn_cast<scf::ForOp>(loopOp);
  if (!forOp) {
    // blockCounters / cntArgs are for-only.
    return UPDATE_CONDITION_INFO_SUCCESS;
  }

  auto counterIt = info->blockCounters.find(forOp);
  if (counterIt == info->blockCounters.end()) {
    LDBG("Failed to assign counters for ssbuffer if ops: no counters for "
         << "forOp=" << forOp << "\n");
    return UPDATE_CONDITION_INFO_FAILED;
  }

  size_t counterNum = counterIt->second.size();
  if (ifOpCount > counterNum) {
    LDBG("Failed to assign counters for all ssbuffer if ops: if ops "
         << ifOpCount << ", counters " << counterNum << ", forOp=" << forOp
         << "\n");
    return UPDATE_CONDITION_INFO_FAILED;
  }
  return UPDATE_CONDITION_INFO_SUCCESS;
}

SmallVector<Type>
UpdateConditionInfoPass::buildNewIfResultTypes(scf::IfOp oldIfOp,
                                               bool hasCounter, Value counter) {
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
  LDBG("Build new if result types: old results "
       << oldIfOp.getNumResults() << ", control vars " << currentUsedVars.size()
       << ", has counter " << hasCounter << "." << "\n");
  return resultTypes;
}

void UpdateConditionInfoPass::collectYieldOperands(
    Block &block, Operation *&yieldOp, SmallVector<Value> &yieldOperands) {
  yieldOp = nullptr;
  yieldOperands.clear();
  if (block.empty()) {
    LDBG("Collect yield operands: block is empty." << "\n");
    return;
  }

  Operation *lastOp = &block.back();
  if (!isa<scf::YieldOp>(lastOp)) {
    LDBG("Collect yield operands: block terminator is not scf.yield." << "\n");
    return;
  }

  yieldOp = lastOp;
  auto scfYieldOp = cast<scf::YieldOp>(lastOp);
  yieldOperands.assign(scfYieldOp.getOperands().begin(),
                       scfYieldOp.getOperands().end());
  LDBG("Collected " << yieldOperands.size() << " yield operands." << "\n");
}

void UpdateConditionInfoPass::populateNewThenBlock(
    scf::IfOp newIfOp, Block &oldThenBlock, Operation *oldThenYieldOp,
    ArrayRef<Value> oldYieldOperands,
    DenseMap<Value, VarUpdateType> &varUpdateTypes, bool hasCounter,
    Value counter, Value step) {
  Location loc = newIfOp.getLoc();
  Block &newThenBlock = newIfOp.getThenRegion().front();
  for (Operation &op : llvm::make_early_inc_range(oldThenBlock)) {
    if (&op != oldThenYieldOp) {
      op.moveBefore(&newThenBlock, newThenBlock.end());
    }
  }
  LDBG("Populate new then block with " << oldYieldOperands.size()
                                       << " original yield operands." << "\n");

  OpBuilder thenBuilder(&newThenBlock, newThenBlock.end());
  SmallVector<Value> thenYieldOperands(oldYieldOperands.begin(),
                                       oldYieldOperands.end());
  if (!currentUsedVars.empty()) {
    Value one =
        thenBuilder.create<arith::ConstantIntOp>(loc, 1, CONST_INT_TYPE);
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
    LDBG("Append updated counter to then yield." << "\n");
  }

  LDBG("Create then yield with " << thenYieldOperands.size() << " operands."
                                 << "\n");
  thenBuilder.create<scf::YieldOp>(loc, thenYieldOperands);
}

void UpdateConditionInfoPass::populateNewElseBlock(scf::IfOp newIfOp,
                                                   scf::IfOp oldIfOp,
                                                   bool oldHasElse,
                                                   bool hasCounter,
                                                   Value counter) {
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
    LDBG("Moved old else block ops and collected "
         << oldElseYieldOperands.size() << " old else yield operands." << "\n");
  }

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

  LDBG("Create else yield with " << elseYieldOperands.size() << " operands."
                                 << "\n");
  elseBuilder.create<scf::YieldOp>(loc, elseYieldOperands);
  if (oldElseYieldOp) {
    oldElseYieldOp->erase();
  }
}

// Create new IfOp with new then and else blocks.
scf::IfOp UpdateConditionInfoPass::createNewIfOpWithBlocks(
    scf::IfOp oldIfOp, Value combinedCond,
    DenseMap<Value, VarUpdateType> &varUpdateTypes, bool hasCounter,
    Value counter, Value step) {
  Location loc = oldIfOp.getLoc();
  OpBuilder builder(oldIfOp);

  bool needsYield = !currentUsedVars.empty() || hasCounter;
  bool oldHasElse = oldIfOp.getElseRegion().hasOneBlock();
  // Only create else when we will populate it;
  bool withElse = needsYield || oldHasElse;
  LDBG("Create replacement if op: needs yield "
       << needsYield << ", old has else " << oldHasElse << ", with else "
       << withElse << ", current used vars " << currentUsedVars.size() << "."
       << "\n");

  Block &oldThenBlock = oldIfOp.getThenRegion().front();
  Operation *oldThenYieldOp = nullptr;
  SmallVector<Value> oldYieldOperands;
  collectYieldOperands(oldThenBlock, oldThenYieldOp, oldYieldOperands);
  SmallVector<Type> resultTypes =
      buildNewIfResultTypes(oldIfOp, hasCounter, counter);
  scf::IfOp newIfOp =
      builder.create<scf::IfOp>(loc, resultTypes, combinedCond, withElse);
  LDBG("Created replacement if op with " << resultTypes.size() << " results."
                                         << "\n");

  for (auto &attr : oldIfOp->getAttrs()) {
    newIfOp->setAttr(attr.getName(), attr.getValue());
  }

  populateNewThenBlock(newIfOp, oldThenBlock, oldThenYieldOp, oldYieldOperands,
                       varUpdateTypes, hasCounter, counter, step);
  if (withElse) {
    populateNewElseBlock(newIfOp, oldIfOp, oldHasElse, hasCounter, counter);
  }

  for (size_t i = 0; i < oldIfOp.getNumResults(); ++i) {
    oldIfOp.getResult(i).replaceAllUsesWith(newIfOp.getResult(i));
  }
  LDBG("Replaced " << oldIfOp.getNumResults() << " old if results." << "\n");

  return newIfOp;
}

// Combine the conditions: crossCore condition + intraCore condition + counter
// condition + flowOpt condition
int UpdateConditionInfoPass::combineConditions(
    ModuleOp module, Value crossCoreCond, Value intraCoreCond,
    Value flowOptCond, scf::IfOp ifOp, Operation *loopOp,
    size_t &usedCounterNum, DenseMap<Value, VarUpdateType> &varUpdateTypes) {
  Location loc = ifOp.getLoc();
  SmallVector<Value> validConditions;
  Value counter;
  // Only for updates counter args inside if then  (+step).
  bool updateCounterArg = false;

  if (crossCoreCond) {
    validConditions.push_back(crossCoreCond);
  }
  if (intraCoreCond) {
    validConditions.push_back(intraCoreCond);
  }
  if (flowOptCond) {
    validConditions.push_back(flowOptCond);
  }

  auto forOp = dyn_cast<scf::ForOp>(loopOp);
  auto whileOp = dyn_cast<scf::WhileOp>(loopOp);
  OpBuilder condBuilder(ifOp);

  if (forOp) {
    if (!info->blockCounters.count(forOp)) {
      LDBG("Missing block counters for forOp=" << forOp << ", ifOp=" << ifOp
                                               << "\n");
      return UPDATE_CONDITION_INFO_FAILED;
    }

    SmallVector<int> &counterIndices = info->blockCounters[forOp];

    if (info->cntArgs.count(ifOp)) {
      counter = info->cntArgs[ifOp];
      updateCounterArg = true;
    } else {
      if (usedCounterNum >= counterIndices.size()) {
        LDBG("Not enough counters for ssbuffer if ops: used "
             << usedCounterNum << ", counters " << counterIndices.size()
             << ", forOp=" << forOp << ", ifOp=" << ifOp << "\n");
        return UPDATE_CONDITION_INFO_FAILED;
      }

      int argIdx = counterIndices[usedCounterNum];
      int iterArgNum = static_cast<int>(forOp.getNumRegionIterArgs());
      if (argIdx < 0 || argIdx >= iterArgNum) {
        LDBG("Invalid counter arg index: " << argIdx << ", iter args "
                                           << iterArgNum << ", forOp=" << forOp
                                           << ", ifOp=" << ifOp << "\n");
        return UPDATE_CONDITION_INFO_FAILED;
      }

      counter = forOp.getRegionIterArgs()[argIdx];
      updateCounterArg = true;
      info->cntArgs[ifOp] = counter;
      usedCounterNum++;
      LDBG("Assign counter iter arg index " << argIdx << " to ssbuffer if op."
                                            << "\n");
    }

    LDBG("this ifop used counter is: " << counter << "\n");
    Value upperBound = forOp.getUpperBound();
    Value counterToUse = counter;
    auto latestIt = controlVarToLatestValue.find(counter);
    if (latestIt != controlVarToLatestValue.end()) {
      counterToUse = latestIt->second;
    }
    Value counterCond = condBuilder.create<arith::CmpIOp>(
        loc, arith::CmpIPredicate::slt, counterToUse, upperBound);
    validConditions.push_back(counterCond);
  } else if (whileOp) {
    // Only add remapped scf.condition(x) expression;
    Value counterCond;
    if (buildWhileCounterCondition(whileOp, ifOp, info, condBuilder,
                                   controlVarToLatestValue,
                                   counterCond) == UPDATE_CONDITION_INFO_FAILED)
      return UPDATE_CONDITION_INFO_FAILED;
    validConditions.push_back(counterCond);
  } else {
    LDBG("Unsupported loop op for counter condition, loopOp="
         << *loopOp << ", ifOp=" << ifOp << "\n");
    return UPDATE_CONDITION_INFO_FAILED;
  }

  if (validConditions.empty()) {
    LDBG("Failed to build any condition for ssbuffer if op, ifOp="
         << ifOp << ", loopOp=" << *loopOp << "\n");
    return UPDATE_CONDITION_INFO_FAILED;
  }

  LDBG("Combine " << validConditions.size() << " conditions for ssbuffer if op."
                  << "\n");
  OpBuilder builder(ifOp);
  Value combinedCond = validConditions[0];
  for (size_t i = 1; i < validConditions.size(); ++i) {
    combinedCond =
        builder.create<arith::AndIOp>(loc, combinedCond, validConditions[i]);
  }

  Value step = updateCounterArg ? forOp.getStep() : Value();
  scf::IfOp newIfOp = createNewIfOpWithBlocks(
      ifOp, combinedCond, varUpdateTypes, updateCounterArg, counter, step);

  // Update DAG nodes
  updateDAGAfterIfOpReplacement(ifOp, newIfOp);

  if (updateCounterArg) {
    info->cntArgs.erase(ifOp);
    info->cntArgs[newIfOp] = counter;
  }

  // Update mappings that refer to the old ifOp BEFORE erasing it
  // Update the tensorIterArgIfOpVars mapping
  if (tensorIterArgIfOpVars.count(ifOp)) {
    auto ifOpVars = tensorIterArgIfOpVars[ifOp];
    tensorIterArgIfOpVars.erase(ifOp);
    tensorIterArgIfOpVars[newIfOp] = ifOpVars;
  }

  // Update tensorIterArgDepsMap with new ifOp
  if (info->tensorIterArgDepsMap.count(loopOp)) {
    auto &depsVec = info->tensorIterArgDepsMap[loopOp];
    for (auto &relation : depsVec) {
      // Update producer ifOp - use pointer comparison
      if (relation.producer.getOperation() == ifOp.getOperation()) {
        relation.producer = newIfOp;
      }
      // Update consumer ifOps
      for (auto &consumerIfOp : relation.consumers) {
        if (consumerIfOp.getOperation() == ifOp.getOperation()) {
          consumerIfOp = newIfOp;
        }
      }
    }
  }

  updateControlVarToLatestValue(newIfOp, ifOp, updateCounterArg, counter);

  ifOp.erase();
  return UPDATE_CONDITION_INFO_SUCCESS;
}

// Update the conditions of ifOp.
int UpdateConditionInfoPass::updateIfConds(
    ModuleOp module, SmallVector<SmallVector<Value>> ssbufferPtrs) {
  // Walk main_loop for/while ops to update the conditions of ifOp
  SmallVector<Operation *> mainLoopOps;
  WalkResult walkResult = module.walk([&](Operation *op) -> WalkResult {
    if (CVPipeline::isMainLoopOp(op)) {
      mainLoopOps.push_back(op);
      return WalkResult::advance();
    }

    if (op->hasAttr(kMainLoop)) {
      LDBG("Found unsupported main loop op: " << *op << "\n");
      return WalkResult::interrupt();
    }
    return WalkResult::advance();
  });
  if (walkResult.wasInterrupted()) {
    return UPDATE_CONDITION_INFO_FAILED;
  }

  // Step0: Collect dependency buffers once outside the loop
  DenseMap<int, DenseMap<Operation *, SmallVector<Operation *>>>
      crossCoreBuffers;
  DenseMap<Operation *,
           DenseMap<int, DenseMap<Operation *, SmallVector<Operation *>>>>
      intraCoreBuffersMap;
  collectDependencyBuffers(module, mainLoopOps, crossCoreBuffers,
                           intraCoreBuffersMap);

  for (Operation *loopOp : mainLoopOps) {
    controlVarToLatestValue.clear();

    // Step1: Get intraCoreBuffers from pre-collected map
    DenseMap<int, DenseMap<Operation *, SmallVector<Operation *>>>
        intraCoreBuffers;
    if (intraCoreBuffersMap.count(loopOp)) {
      intraCoreBuffers = intraCoreBuffersMap[loopOp];
    }

    // for/while requires at least one of cross/intra deps;
    if (crossCoreBuffers.empty() && intraCoreBuffers.empty()) {
      LDBG("crossCoreBuffers and intraCoreBuffers are all empty! loopOp="
           << *loopOp << "\n");
      return UPDATE_CONDITION_INFO_FAILED;
    }

    // Step2: Assign control variables (intraCore inputs / tensor iter_args)
    DenseMap<int, Value> idxToVar;
    if (buildIdxToVarMap(loopOp, intraCoreBuffers, idxToVar) ==
        UPDATE_CONDITION_INFO_FAILED) {
      return UPDATE_CONDITION_INFO_FAILED;
    }
    if (buildTensorIterArgIfOpVarMap(loopOp) == UPDATE_CONDITION_INFO_FAILED) {
      return UPDATE_CONDITION_INFO_FAILED;
    }

    size_t usedCounterNum = 0;
    SmallVector<scf::IfOp> ifOps;
    if (collectSSBufferIfOps(loopOp, ifOps) == UPDATE_CONDITION_INFO_FAILED) {
      return UPDATE_CONDITION_INFO_FAILED;
    }
    if (validateBlockCounters(loopOp, ifOps.size()) ==
        UPDATE_CONDITION_INFO_FAILED) {
      return UPDATE_CONDITION_INFO_FAILED;
    }

    // Update the conditions of ifOp in this loop op.
    for (scf::IfOp ifOp : ifOps) {
      SmallVector<int> crossCoreInputValues;
      SmallVector<int> crossCoreOutputValues;
      SmallVector<int> intraCoreInputValues;
      SmallVector<int> intraCoreOutputValues;

      if (getInputOutputValues(ifOp, crossCoreBuffers, intraCoreBuffers,
                               crossCoreInputValues, crossCoreOutputValues,
                               intraCoreInputValues,
                               intraCoreOutputValues) != 0) {
        LDBG("getInputOutputValues failed! ifOp=" << ifOp << ", loopOp="
                                                  << *loopOp << "\n");
        return UPDATE_CONDITION_INFO_FAILED;
      }

      // Step3:Set the crossCore condition
      Value crossCoreCond;
      if (setCrossCoreCondition(crossCoreInputValues, crossCoreOutputValues,
                                crossCoreBuffers, ifOp, ssbufferPtrs,
                                crossCoreCond) != 0) {
        LDBG("setCrossCoreCondition failed! ifOp=" << ifOp << ", loopOp="
                                                   << *loopOp << "\n");
        return UPDATE_CONDITION_INFO_FAILED;
      }
      // Step4:Set the intraCore condition
      DenseMap<Value, VarUpdateType> varUpdateTypes;
      Value intraCoreCond;
      if (setIntraCoreCondition(module, ifOp, intraCoreBuffers,
                                intraCoreInputValues, intraCoreOutputValues,
                                idxToVar, varUpdateTypes, intraCoreCond) ==
          UPDATE_CONDITION_INFO_FAILED) {
        return UPDATE_CONDITION_INFO_FAILED;
      }
      // Step5:Set the flowOpt condition (for only; needs lb/ub/step)
      Value flowOptCond;
      if (setFlowOptCondition(ifOp, loopOp, flowOptCond) ==
          UPDATE_CONDITION_INFO_FAILED) {
        return UPDATE_CONDITION_INFO_FAILED;
      }
      // Step6:Combine the conditions: crossCore + intraCore + counter +
      // flowOpt
      if (combineConditions(module, crossCoreCond, intraCoreCond, flowOptCond,
                            ifOp, loopOp, usedCounterNum,
                            varUpdateTypes) == UPDATE_CONDITION_INFO_FAILED) {
        return UPDATE_CONDITION_INFO_FAILED;
      }
    }
    // Step7: Update loop yield with latest control values
    if (updateLoopYield(loopOp) == UPDATE_CONDITION_INFO_FAILED) {
      return UPDATE_CONDITION_INFO_FAILED;
    }
  }
  return UPDATE_CONDITION_INFO_SUCCESS;
}

void UpdateConditionInfoPass::runOnOperation() {
  ModuleOp module = getOperation();

  if (CVPipeline::hasFallbackAttr(module)) {
    return;
  }

  LDBG("Enter UpdateConditionInfo pass." << "\n");
  // Step1:Init the ssbufferPtrs
  SmallVector<SmallVector<Value>> ssbufferPtrs = allocSSBuffer(module);

  // Step2:Update the conditions of ifOp based on the intraCoreDependentMap and
  // crossCoreDependentMap
  int updateResult = updateIfConds(module, ssbufferPtrs);

  if (updateResult != UPDATE_CONDITION_INFO_SUCCESS) {
    LDBG("updateIfConds failed!");
    CVPipeline::setFallbackAttr(module, CVPipeline::ERRCODE_FAILED);
  }

  LDBG("Exit UpdateConditionInfo pass." << "\n");
}

namespace mlir {
namespace triton {
std::unique_ptr<OperationPass<ModuleOp>> createUpdateConditionInfoPass() {
  return std::make_unique<UpdateConditionInfoPass>();
}
} // namespace triton
} // namespace mlir
