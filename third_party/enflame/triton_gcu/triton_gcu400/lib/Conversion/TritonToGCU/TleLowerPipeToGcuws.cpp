/**
 * Copyright 2025-2026 Enflame. All Rights Reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *  http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include <map>
#include <string>
#include <utility>

#include "Conversion/TritonToGCU/TritonToGCUPass.h"
#include "Dialect/GCUWS/IR/Dialect.h"
#include "Utils/TritonVersionCompat.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/Pass/Pass.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/Support/Debug.h"

#ifdef ENABLE_TLE
#include "tle/dialect/include/IR/Dialect.h"
#endif

namespace mlir {
#define GEN_PASS_DEF_TLELOWERPIPETOGCUWS
#include "Conversion/Passes.h.inc"
} // namespace mlir

using namespace mlir;

#define DEBUG_TYPE "tle-lower-pipe-to-gcuws"
#define DBGS() (llvm::dbgs() << "[" DEBUG_TYPE "]: ")
#define LDBG(X) LLVM_DEBUG(DBGS() << X << "\n")

namespace tt = mlir::triton;
namespace ttg = mlir::triton::gpu;
namespace ttgcuws = mlir::triton::gcuws;

#ifdef ENABLE_TLE
namespace tle = mlir::triton::tle;
#endif

namespace {

struct PipeState {
  Value pipeline;
  int32_t capacity = 0;
  bool oneShot = false;
};

#ifdef ENABLE_TLE
static bool containsPipeLifecycleOp(tt::FuncOp func) {
  bool found = false;
  func.walk([&](Operation *op) {
    if (isa<tle::PipeCreateOp, tle::PipeWriterAcquireOp,
            tle::PipeWriterCommitOp, tle::PipeWriterCloseOp,
            tle::PipeReaderWaitOp, tle::PipeReaderReleaseOp>(op))
      found = true;
  });
  return found;
}

static LogicalResult inlinePipeCall(tt::CallOp call, tt::FuncOp callee) {
  if (callee.isExternal())
    return call.emitOpError("can't inline external callee containing pipe ops");
  Region &body = callee.getBody();
  if (!body.hasOneBlock())
    return call.emitOpError("can't inline multi-block callee containing pipe "
                            "ops before pipe lowering");

  Block &block = body.front();
  auto returnOp = dyn_cast<tt::ReturnOp>(block.getTerminator());
  if (!returnOp)
    return call.emitOpError("callee containing pipe ops must terminate with "
                            "tt.return before pipe lowering");
  if (returnOp.getNumOperands() != call.getNumResults())
    return call.emitOpError("callee return count does not match call results");

  IRMapping mapping;
  for (auto [arg, operand] :
       llvm::zip(block.getArguments(), call.getOperands()))
    mapping.map(arg, operand);

  OpBuilder builder(call);
  for (Operation &op : block.getOperations()) {
    if (&op == returnOp.getOperation())
      continue;
    builder.clone(op, mapping);
  }

  for (auto [result, returned] :
       llvm::zip(call.getResults(), returnOp.getOperands()))
    result.replaceAllUsesWith(mapping.lookupOrDefault(returned));
  call.erase();
  return success();
}

static tt::FuncOp resolveCallee(tt::CallOp call) {
  return dyn_cast_or_null<tt::FuncOp>(
      SymbolTable::lookupNearestSymbolFrom(call, call.getCalleeAttr()));
}

static LogicalResult inlinePipeHelperCalls(ModuleOp module) {
  bool changed = true;
  while (changed) {
    changed = false;
    SmallVector<tt::CallOp> calls;
    module.walk([&](tt::CallOp call) {
      auto callee = resolveCallee(call);
      if (callee && containsPipeLifecycleOp(callee))
        calls.push_back(call);
    });

    for (tt::CallOp call : calls) {
      if (!call->getBlock())
        continue;
      auto callee = resolveCallee(call);
      if (!callee || !containsPipeLifecycleOp(callee))
        continue;
      if (failed(inlinePipeCall(call, callee)))
        return failure();
      changed = true;
    }
  }

  SmallVector<tt::FuncOp> toErase;
  module.walk([&](tt::FuncOp func) {
    if (!containsPipeLifecycleOp(func))
      return;
    if (func.getVisibility() != SymbolTable::Visibility::Public &&
        SymbolTable::symbolKnownUseEmpty(func, module))
      toErase.push_back(func);
  });
  for (auto func : toErase)
    func.erase();

  return success();
}

static std::optional<std::pair<ttg::WarpSpecializeOp, Region *>>
getEnclosingWarpSpecializePartition(Operation *op) {
  for (Region *region = op->getParentRegion(); region;) {
    Operation *parent = region->getParentOp();
    if (!parent)
      break;
    if (auto partitions = dyn_cast<ttg::WarpSpecializePartitionsOp>(parent))
      return std::make_pair(
          cast<ttg::WarpSpecializeOp>(partitions->getParentOp()), region);
    region = parent->getParentRegion();
  }
  return std::nullopt;
}

static bool isDefinedInsideRegion(Value value, Region *region) {
  if (auto blockArg = dyn_cast<BlockArgument>(value))
    return region->isAncestor(blockArg.getOwner()->getParent());
  Operation *def = value.getDefiningOp();
  return def && region->isAncestor(def->getParentRegion());
}

static Value getWarpSpecializeCaptureForUse(Operation *useOp, Value value) {
  auto partition = getEnclosingWarpSpecializePartition(useOp);
  if (!partition)
    return value;

  ttg::WarpSpecializeOp wsOp = partition->first;
  Region *region = partition->second;
  if (isDefinedInsideRegion(value, region))
    return value;

  OperandRange captures = triton_gcu::compat::getWsExplicitCaptures(wsOp);
  for (auto indexed : llvm::enumerate(captures)) {
    if (indexed.value() == value)
      return region->getArgument(indexed.index());
  }

  // New capture is appended at the end; its region-argument index equals the
  // number of captures before insertion. Capture the size first because
  // inserting an operand may invalidate the OperandRange view (and on Triton
  // 3.7 the captures live on the PartitionsOp, not on wsOp's operands).
  unsigned captureIndex = captures.size();
  triton_gcu::compat::insertWsCapture(wsOp, value);
  for (Region *partitionRegion : wsOp.getPartitionRegions())
    partitionRegion->addArgument(value.getType(), value.getLoc());
  return region->getArgument(captureIndex);
}

static OperandRange getPipeFields(Operation *op) {
  if (auto pipeOp = dyn_cast<tle::PipeCreateOp>(op))
    return pipeOp.getFields();
  if (auto pipeOp = dyn_cast<tle::PipeWriterAcquireOp>(op))
    return pipeOp.getFields();
  if (auto pipeOp = dyn_cast<tle::PipeWriterCommitOp>(op))
    return pipeOp.getFields();
  if (auto pipeOp = dyn_cast<tle::PipeWriterCloseOp>(op))
    return pipeOp.getFields();
  if (auto pipeOp = dyn_cast<tle::PipeReaderWaitOp>(op))
    return pipeOp.getFields();
  return cast<tle::PipeReaderReleaseOp>(op).getFields();
}

// Trace a pipe field value back through WS partition block arguments to the
// original capture value defined outside the WarpSpecializeOp. This ensures
// that the same pipe used in different partitions produces the same key.
static Value canonicalizePipeField(Value field) {
  while (auto blockArg = dyn_cast<BlockArgument>(field)) {
    Block *block = blockArg.getOwner();
    auto partitions = dyn_cast_or_null<ttg::WarpSpecializePartitionsOp>(
        block->getParentOp());
    if (partitions) {
      auto wsOp = dyn_cast<ttg::WarpSpecializeOp>(partitions->getParentOp());
      if (!wsOp)
        break;
      unsigned argNo = blockArg.getArgNumber();
      OperandRange captures = triton_gcu::compat::getWsExplicitCaptures(wsOp);
      if (argNo >= captures.size())
        break;
      field = captures[argNo];
      continue;
    }
    break;
  }
  return field;
}

static std::string getPipeKey(Operation *op) {
  std::string key;
  llvm::raw_string_ostream os(key);
  os << op->getAttrOfType<IntegerAttr>("capacity").getInt() << "|";
  op->getAttr("scope").print(os);
  os << "|";
  if (Attribute pipeName = op->getAttr("pipe_name"))
    pipeName.print(os);
  os << "|";
  op->getAttr("field_names").print(os);
  os << "|";
  for (Value field : getPipeFields(op))
    os << canonicalizePipeField(field).getAsOpaquePointer() << ",";
  return key;
}

// Determine the number of warps for the partition that contains `op`.
// Default region uses the kernel's num_warps; worker partitions use their
// per-partition num_warps from the WarpSpecializeOp attribute.
static int getWarpsForPartition(Operation *op, int kernelNumWarps) {
  auto partition = getEnclosingWarpSpecializePartition(op);
  if (!partition) {
    return kernelNumWarps;
  }
  ttg::WarpSpecializeOp wsOp = partition->first;
  Region *region = partition->second;
  unsigned regionIdx = region->getRegionNumber();
  ArrayRef<int32_t> pnw = wsOp.getPartitionNumWarps();
  if (regionIdx < pnw.size())
    return pnw[regionIdx];
  return 1;
}

// Fix memdesc_index slot indices to use a continuous pipe iteration counter.
//
// Problem: When a TLE pipe is used inside a nested loop (outer tile loop +
// inner k loop), the user-provided iteration index (typically the inner loop
// variable k) resets to 0 at each tile boundary. The memdesc_index uses
// stage = k % capacity to select a buffer slot, while the GCUWS pipeline's
// internal counter is continuous across tiles. This desynchronization causes
// data corruption when num_k_tiles is odd and the program processes multiple
// tiles.
//
// Fix: Add a loop-carried counter (pipe_cnt) to the outer loop. Inside the
// inner loop, compute pipe_iter = pipe_cnt + iterIndex and new_stage =
// pipe_iter % capacity. Replace the stage in memdesc_index with new_stage so
// that the buffer slot selection matches the GCUWS pipeline's continuous
// counter.
//
// iterIndex normalizes the inner loop induction variable to a zero-based,
// unit-step index: iterIndex = (k - lb) floordiv step. When lb=0 and step=1,
// iterIndex = k (no extra arith ops needed).
//
// tripCount = ceilDiv(ub - lb, step) gives the number of iterations per tile,
// which is added to pipe_cnt at the end of each outer loop iteration.
//
// Multiple pipes in the same outer loop are handled by adding one pipe_cnt
// iter_arg per pipe (each pipe gets its own independent counter).
static void fixPipeSlotIndex(ModuleOp mod) {
  // Collect all pipe ops that use a stage value.
  struct PipeOpInfo {
    Operation *op;
    Value stage;
  };

  SmallVector<PipeOpInfo> pipeOps;
  mod.walk([&](tle::PipeWriterAcquireOp op) {
    pipeOps.push_back({op, op.getStage()});
  });
  mod.walk([&](tle::PipeReaderWaitOp op) {
    pipeOps.push_back({op, op.getStage()});
  });

  if (pipeOps.empty())
    return;

  // Group pipe ops by their (outerLoop, innerLoop, stage) so that all
  // memdesc_index ops sharing the same stage are fixed together, and each
  // unique (outerLoop, innerLoop) pair gets its own pipe_cnt.
  struct FixGroup {
    scf::ForOp outerFor;
    scf::ForOp innerFor;
    Value stage;
    arith::RemSIOp remOp;
    Value k;
    Value capacity;
    int64_t lbVal;
    int64_t stepVal;
    SmallVector<Operation *> memdescIndexOps;
    Value pipeCnt; // filled during processing
  };

  // Use a DenseMap keyed by the stage Value's opaque pointer to deduplicate
  // groups with the same stage (same pipe used by both acquire and wait).
  DenseMap<void *, FixGroup> stageToGroup;

  for (auto &[op, stage] : pipeOps) {
    auto innerFor = op->getParentOfType<scf::ForOp>();
    if (!innerFor)
      continue;

    auto outerFor = innerFor->getParentOfType<scf::ForOp>();
    if (!outerFor)
      continue;

    auto stageDef = stage.getDefiningOp();
    if (!stageDef)
      continue;

    auto remOp = dyn_cast<arith::RemSIOp>(stageDef);
    if (!remOp)
      continue;

    Value k = remOp.getLhs();
    Value capacity = remOp.getRhs();

    if (k != innerFor.getInductionVar())
      continue;

    auto lbDef = innerFor.getLowerBound().getDefiningOp<arith::ConstantOp>();
    auto stepDef = innerFor.getStep().getDefiningOp<arith::ConstantOp>();
    if (!lbDef || !stepDef)
      continue;

    auto lbAttr = dyn_cast<IntegerAttr>(lbDef.getValueAttr());
    auto stepAttr = dyn_cast<IntegerAttr>(stepDef.getValueAttr());
    if (!lbAttr || !stepAttr)
      continue;

    int64_t lbVal = lbAttr.getInt();
    int64_t stepVal = stepAttr.getInt();
    if (stepVal <= 0)
      continue;

    // Deduplicate by stage value — same stage means same pipe (acquire+wait).
    void *stageKey = stage.getAsOpaquePointer();
    auto it = stageToGroup.find(stageKey);
    if (it != stageToGroup.end()) {
      // Already have a group for this stage; just collect memdesc_index ops.
      continue;
    }

    FixGroup group;
    group.outerFor = outerFor;
    group.innerFor = innerFor;
    group.stage = stage;
    group.remOp = remOp;
    group.k = k;
    group.capacity = capacity;
    group.lbVal = lbVal;
    group.stepVal = stepVal;

    innerFor.walk([&](ttg::MemDescIndexOp indexOp) {
      if (indexOp.getIndex() == stage)
        group.memdescIndexOps.push_back(indexOp);
    });

    if (group.memdescIndexOps.empty())
      continue;

    stageToGroup[stageKey] = std::move(group);
  }

  if (stageToGroup.empty())
    return;

  // Group fix groups by their outer loop so that all pipe_cnt iter_args
  // for the same outer loop are added in a single replacement (avoiding
  // dangling references after erase).
  DenseMap<Operation *, SmallVector<FixGroup *>> outerLoopToGroups;
  for (auto &[stageKey, group] : stageToGroup)
    outerLoopToGroups[group.outerFor.getOperation()].push_back(&group);

  for (auto &[outerOp, groups] : outerLoopToGroups) {
    auto outerFor = scf::ForOp(outerOp);

    auto loc = outerFor.getLoc();
    auto type = groups[0]->stage.getType();

    OpBuilder builder(outerFor);

    // Create one zero constant per group (all share the same type).
    Value zeroVal = builder.create<arith::ConstantOp>(
        loc, type, builder.getIntegerAttr(type, 0));

    // Add one pipe_cnt block argument per group to the old outer loop body.
    for (auto *group : groups) {
      outerFor.getBody()->addArgument(type, loc);
      group->pipeCnt = outerFor.getBody()->getArguments().back();
    }

    // Create new outer loop with extended init args.
    SmallVector<Value> newInitArgs;
    for (auto initArg : outerFor.getInitArgs())
      newInitArgs.push_back(initArg);
    for (size_t i = 0; i < groups.size(); ++i)
      newInitArgs.push_back(zeroVal);

    auto newOuterFor = builder.create<scf::ForOp>(
        loc, outerFor.getLowerBound(), outerFor.getUpperBound(),
        outerFor.getStep(), newInitArgs);

    newOuterFor.getRegion().takeBody(outerFor.getRegion());

    // For each group, compute tripCount and pipe_cnt_next, and fix memdesc_index.
    auto yieldOp = cast<scf::YieldOp>(newOuterFor.getBody()->getTerminator());
    builder.setInsertionPoint(yieldOp);

    for (auto *group : groups) {
      auto innerFor = group->innerFor;
      Value pipeCnt = group->pipeCnt;
      int64_t lbVal = group->lbVal;
      int64_t stepVal = group->stepVal;

      // Compute tripCount = ceilDiv(ub - lb, step).
      Value tripCount;
      if (lbVal == 0 && stepVal == 1) {
        tripCount = innerFor.getUpperBound();
      } else {
        Value ubVal = innerFor.getUpperBound();
        Value lbConst = builder.create<arith::ConstantOp>(
            loc, type, builder.getIntegerAttr(type, lbVal));
        Value stepConst = builder.create<arith::ConstantOp>(
            loc, type, builder.getIntegerAttr(type, stepVal));
        Value range = builder.create<arith::SubIOp>(loc, ubVal, lbConst);
        Value divResult =
            builder.create<arith::DivSIOp>(loc, range, stepConst);
        Value remResult =
            builder.create<arith::RemSIOp>(loc, range, stepConst);
        Value hasRem = builder.create<arith::CmpIOp>(
            loc, arith::CmpIPredicate::ne, remResult, zeroVal);
        Value one = builder.create<arith::ConstantOp>(
            loc, type, builder.getIntegerAttr(type, 1));
        Value extra =
            builder.create<arith::SelectOp>(loc, hasRem, one, zeroVal);
        tripCount = builder.create<arith::AddIOp>(loc, divResult, extra);
      }

      Value pipeCntNext =
          builder.create<arith::AddIOp>(loc, pipeCnt, tripCount);
      yieldOp->insertOperands(yieldOp.getNumOperands(), {pipeCntNext});
    }

    // Replace old outer loop with new one.
    for (unsigned i = 0; i < outerFor.getNumResults(); ++i)
      outerFor.getResult(i).replaceAllUsesWith(newOuterFor.getResult(i));
    outerFor.erase();

    // For each group, fix memdesc_index ops inside the inner loop.
    for (auto *group : groups) {
      auto remOp = group->remOp;
      Value k = group->k;
      Value capacity = group->capacity;
      Value pipeCnt = group->pipeCnt;
      int64_t lbVal = group->lbVal;
      int64_t stepVal = group->stepVal;
      auto &memdescIndexOps = group->memdescIndexOps;

      // Compute iterIndex = (k - lb) floordiv step.
      Value iterIndex;
      builder.setInsertionPoint(remOp);
      if (lbVal == 0 && stepVal == 1) {
        iterIndex = k;
      } else {
        Value lbConst = builder.create<arith::ConstantOp>(
            loc, type, builder.getIntegerAttr(type, lbVal));
        Value stepConst = builder.create<arith::ConstantOp>(
            loc, type, builder.getIntegerAttr(type, stepVal));
        Value offset = builder.create<arith::SubIOp>(loc, k, lbConst);
        iterIndex = builder.create<arith::DivSIOp>(loc, offset, stepConst);
      }

      Value pipeIter = builder.create<arith::AddIOp>(loc, pipeCnt, iterIndex);
      Value newStage =
          builder.create<arith::RemSIOp>(loc, pipeIter, capacity);

      for (auto *memdescIndexOp : memdescIndexOps)
        memdescIndexOp->setOperand(1, newStage);

      LDBG("fixPipeSlotIndex: replaced stage in "
           << memdescIndexOps.size()
           << " memdesc_index ops with continuous pipe_iter");
    }
  }
}
#endif // ENABLE_TLE

class TleLowerPipeToGcuwsPass
    : public impl::TleLowerPipeToGcuwsBase<TleLowerPipeToGcuwsPass> {
public:
  using impl::TleLowerPipeToGcuwsBase<
      TleLowerPipeToGcuwsPass>::TleLowerPipeToGcuwsBase;

  void runOnOperation() override {
#ifdef ENABLE_TLE
    ModuleOp mod = getOperation();

    fixPipeSlotIndex(mod);

    if (failed(inlinePipeHelperCalls(mod))) {
      signalPassFailure();
      return;
    }

    std::map<std::string, PipeState> pipeStates;
    SmallVector<Operation *> opsToErase;

    int numWarps = triton::gpu::lookupNumWarps(mod);

    struct PipeWarpInfo {
      int producerWarps = 0;
      int consumerWarps = 0;
    };
    std::map<std::string, PipeWarpInfo> pipeWarpInfos;

    // Track unique (pipeKey, partitionRegion) pairs to avoid double-counting
    // the same partition when it has multiple pipe ops in a loop.
    // Key: pipeKey + ":" + wsOp ptr + ":" + regionIdx + ":" + isProducer
    struct PartitionKey {
      std::string pipeKey;
      Operation *wsOp;    // enclosing WarpSpecializeOp
      unsigned regionIdx; // partition region number (0 = default)
      bool isProducer;
    };
    struct PartitionKeyInfo {
      static inline PartitionKey getEmptyKey() {
        return {std::string(), nullptr, ~0u, false};
      }
      static inline PartitionKey getTombstoneKey() {
        return {std::string(), nullptr, ~1u, false};
      }
      static inline unsigned getHashValue(const PartitionKey &k) {
        return llvm::hash_combine(k.pipeKey, k.wsOp, k.regionIdx,
                                  static_cast<int>(k.isProducer));
      }
      static inline bool isEqual(const PartitionKey &a, const PartitionKey &b) {
        return a.pipeKey == b.pipeKey && a.wsOp == b.wsOp &&
               a.regionIdx == b.regionIdx && a.isProducer == b.isProducer;
      }
    };
    llvm::DenseSet<PartitionKey, PartitionKeyInfo> seenPartitions;

    auto getPartitionKey = [&](Operation *op, bool isProducer) -> PartitionKey {
      auto partition = getEnclosingWarpSpecializePartition(op);
      auto wsOp = op->getParentOfType<ttg::WarpSpecializeOp>();
      assert(wsOp && "pipe op must be inside a WarpSpecializeOp");
      PartitionKey pk;
      pk.pipeKey = getPipeKey(op);
      pk.wsOp = wsOp.getOperation();
      pk.isProducer = isProducer;
      if (partition) {
        pk.regionIdx = partition->second->getRegionNumber();
      } else {
        pk.regionIdx = 0; // default region
      }
      return pk;
    };

    mod.walk([&](tle::PipeWriterAcquireOp op) {
      PartitionKey pk = getPartitionKey(op, /*isProducer=*/true);
      if (seenPartitions.insert(pk).second)
        pipeWarpInfos[pk.pipeKey].producerWarps +=
            getWarpsForPartition(op, numWarps);
    });
    mod.walk([&](tle::PipeWriterCommitOp op) {
      PartitionKey pk = getPartitionKey(op, /*isProducer=*/true);
      if (seenPartitions.insert(pk).second)
        pipeWarpInfos[pk.pipeKey].producerWarps +=
            getWarpsForPartition(op, numWarps);
    });
    mod.walk([&](tle::PipeReaderWaitOp op) {
      PartitionKey pk = getPartitionKey(op, /*isProducer=*/false);
      if (seenPartitions.insert(pk).second)
        pipeWarpInfos[pk.pipeKey].consumerWarps +=
            getWarpsForPartition(op, numWarps);
    });
    mod.walk([&](tle::PipeReaderReleaseOp op) {
      PartitionKey pk = getPartitionKey(op, /*isProducer=*/false);
      if (seenPartitions.insert(pk).second)
        pipeWarpInfos[pk.pipeKey].consumerWarps +=
            getWarpsForPartition(op, numWarps);
    });

    // Reject is_closed usage — GCU400 fast_pipeline requires strict
    // producer/consumer pairing. is_closed breaks this by allowing
    // consumer to skip wait/release based on a runtime flag.
    bool hasIsClosedError = false;
    mod.walk([&](tle::PipeReaderWaitOp op) {
      if (!op.getIsClosed().use_empty()) {
        op.emitError("is_closed is not supported on GCU400. The "
                     "fast_pipeline requires producer and consumer to "
                     "execute the same number of iterations.");
        hasIsClosedError = true;
      }
    });
    if (hasIsClosedError) {
      signalPassFailure();
      return;
    }

    mod.walk([&](tle::PipeCreateOp createOp) {
      std::string key = getPipeKey(createOp);
      if (pipeStates.count(key)) {
        LDBG("Skipping duplicate pipe.create for key=" << key);
        opsToErase.push_back(createOp);
        return;
      }
      int32_t capacity = createOp.getCapacity();
      bool oneShot = createOp.getOneShot().value_or(false);

      auto &warpInfo = pipeWarpInfos[key];
      int producerCount =
          warpInfo.producerWarps > 0 ? warpInfo.producerWarps : numWarps;
      int consumerCount =
          warpInfo.consumerWarps > 0 ? warpInfo.consumerWarps : 1;
      LDBG("Pipe key=" << key << " producerCount=" << producerCount
                       << " consumerCount=" << consumerCount);

      OpBuilder builder(createOp);
      auto pipelineType =
          ttgcuws::PipelineType::get(builder.getContext(),
                                     /*stage_count=*/capacity,
                                     /*producer_count=*/producerCount,
                                     /*consumer_count=*/consumerCount,
                                     /*inner_barrier=*/true);
      auto initOp = builder.create<ttgcuws::InitPipelineOp>(createOp.getLoc(),
                                                            pipelineType);
      PipeState state;
      state.pipeline = initOp.getResult();
      state.capacity = capacity;
      state.oneShot = oneShot;
      pipeStates[key] = state;

      LDBG("Created gcuws.init_pipeline for pipe key=" << key << " capacity="
                                                       << capacity);
      opsToErase.push_back(createOp);
    });

    if (pipeStates.empty())
      return;

    mod.walk([&](ttg::WarpSpecializeOp wsOp) {
      wsOp->setAttr("tle.warp_specialize", UnitAttr::get(wsOp.getContext()));
    });

    auto lookupState = [&](Operation *op) -> PipeState * {
      std::string key = getPipeKey(op);
      auto it = pipeStates.find(key);
      return it != pipeStates.end() ? &it->second : nullptr;
    };

    mod.walk([&](tle::PipeWriterAcquireOp acquireOp) {
      auto *state = lookupState(acquireOp);
      if (!state) {
        acquireOp.emitError("pipe.writer_acquire without matching pipe.create");
        signalPassFailure();
        return;
      }
      if (state->oneShot) {
        opsToErase.push_back(acquireOp);
        return;
      }
      OpBuilder builder(acquireOp);
      Value pipeline =
          getWarpSpecializeCaptureForUse(acquireOp, state->pipeline);
      builder.create<ttgcuws::ProducerAcquireOp>(acquireOp.getLoc(), pipeline);
      opsToErase.push_back(acquireOp);
    });

    mod.walk([&](tle::PipeWriterCommitOp commitOp) {
      auto *state = lookupState(commitOp);
      if (!state) {
        commitOp.emitError("pipe.writer_commit without matching pipe.create");
        signalPassFailure();
        return;
      }
      OpBuilder builder(commitOp);
      Value pipeline =
          getWarpSpecializeCaptureForUse(commitOp, state->pipeline);
      builder.create<ttgcuws::ProducerCommitOp>(commitOp.getLoc(), pipeline);
      opsToErase.push_back(commitOp);
    });

    mod.walk([&](tle::PipeReaderWaitOp waitOp) {
      auto *state = lookupState(waitOp);
      if (!state) {
        waitOp.emitError("pipe.reader_wait without matching pipe.create");
        signalPassFailure();
        return;
      }
      OpBuilder builder(waitOp);
      Value pipeline = getWarpSpecializeCaptureForUse(waitOp, state->pipeline);
      builder.create<ttgcuws::ConsumerWaitOp>(waitOp.getLoc(), pipeline);
      assert(!(waitOp.getIsClosed() && !waitOp.getIsClosed().use_empty()) &&
             "is_closed should have been rejected earlier");
      opsToErase.push_back(waitOp);
    });

    mod.walk([&](tle::PipeReaderReleaseOp releaseOp) {
      auto *state = lookupState(releaseOp);
      if (!state) {
        releaseOp.emitError("pipe.reader_release without matching pipe.create");
        signalPassFailure();
        return;
      }
      if (state->oneShot) {
        opsToErase.push_back(releaseOp);
        return;
      }
      OpBuilder builder(releaseOp);
      Value pipeline =
          getWarpSpecializeCaptureForUse(releaseOp, state->pipeline);
      builder.create<ttgcuws::ConsumerReleaseOp>(releaseOp.getLoc(), pipeline);
      opsToErase.push_back(releaseOp);
    });

    mod.walk([&](tle::PipeWriterCloseOp closeOp) {
      auto *state = lookupState(closeOp);
      if (!state) {
        closeOp.emitError("pipe.writer_close without matching pipe.create");
        signalPassFailure();
        return;
      }
      OpBuilder builder(closeOp);
      Value pipeline = getWarpSpecializeCaptureForUse(closeOp, state->pipeline);
      builder.create<ttgcuws::DestroyPipelineOp>(closeOp.getLoc(), pipeline);
      opsToErase.push_back(closeOp);
    });

    for (auto it = opsToErase.rbegin(); it != opsToErase.rend(); ++it)
      (*it)->erase();

    LDBG("Lowered " << pipeStates.size() << " pipe(s) to GCUWS");
#endif // ENABLE_TLE
  }
};

} // namespace
