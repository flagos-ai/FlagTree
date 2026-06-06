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

#ifndef TRITON_ADAPTER_ADD_MULTI_BUFFER_TO_GMLOAD_INTERNAL_H
#define TRITON_ADAPTER_ADD_MULTI_BUFFER_TO_GMLOAD_INTERNAL_H

#include "ascend/include/DynamicCVPipeline/SeparateMemoryFromCompute/AddMultiBufferToGMLoadTypes.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/Operation.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "mlir/Support/LogicalResult.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Debug.h"

using namespace mlir;

// ============================================================================
// Constants
// ============================================================================

static constexpr const char *DEBUG_TYPE = "add-multi-buffer-to-gm-load";
#define LOG_DEBUG(...) LLVM_DEBUG(llvm::dbgs() << " [" << DEBUG_TYPE << "] " << __VA_ARGS__)

static constexpr llvm::StringLiteral kBufferableMarker("gm_load_bufferable");
static constexpr llvm::StringLiteral kBlockIdAttr("ssbuffer.block_id");
static constexpr llvm::StringLiteral kLoadStoreAttr("ssbuffer.load_store");
static constexpr llvm::StringLiteral kGMLoadMultiBufferStage("gm-load multi-buffer");
static constexpr unsigned kInitialRefCapacity = 8; // Tuned for typical load chain depth
static constexpr unsigned kForInductionVarArgIndex = 0;
static constexpr unsigned kForBodyIterArgOffset = 1;
// Each load group carries `depth` slot flags plus producer/consumer counters.
// This count is independent of the configured buffer depth.
static constexpr int kLoadGroupCounterIterArgCount = 2;
static constexpr int kLoadGroupProducerCounterOffset = 0;
static constexpr int kLoadGroupConsumerCounterOffset = 1;

// ============================================================================
// Data structures (debug/utility only, types are in AddMultiBufferToGMLoadTypes.h)
// ============================================================================

namespace gmload {

/// Reuses existing arith.constant ops found in the IR rather than creating
/// duplicates. Pre-populated by scanning a block; get-or-create semantics.
/// Suitable for untagged constants only (no ssbuffer.block_id attribution).
struct ConstantCache {
    llvm::DenseMap<mlir::Attribute, Value> table;

    void scan(Block *block)
    {
        for (auto &op : *block)
            if (auto cst = dyn_cast<arith::ConstantOp>(op))
                table.try_emplace(cst.getValue(), cst.getResult());
    }

    Value get(OpBuilder &builder, Location loc, mlir::TypedAttr attr)
    {
        auto [cacheIt, inserted] = table.try_emplace(attr, Value {});
        if (!inserted)
            return cacheIt->second;
        cacheIt->second = builder.create<arith::ConstantOp>(loc, attr).getResult();
        return cacheIt->second;
    }

    Value getFalse(OpBuilder &builder, Location loc) { return get(builder, loc, builder.getBoolAttr(false)); }

    Value getIndex(OpBuilder &builder, Location loc, int64_t value)
    {
        return get(builder, loc, cast<mlir::TypedAttr>(builder.getIndexAttr(value)));
    }
};

// ============================================================================
// Function declarations: Dependency analysis
// ============================================================================

Operation *getAncestorInBlock(Operation *op, Block *scopeBlock);
void backwardTrace(Operation *op, Block *scopeBlock, llvm::SetVector<Operation *> &visited,
                   SmallVectorImpl<Operation *> &worklist);
void backwardTraceFromWorklist(llvm::SetVector<Operation *> &visited, SmallVectorImpl<Operation *> &worklist,
                               Block *scopeBlock);
SmallVector<Operation *> forwardTraceFromAllocs(llvm::SetVector<Operation *> &visited, Block *scopeBlock);
void captureRegionFreeVars(llvm::SetVector<Operation *> &visited, SmallVectorImpl<Operation *> &worklist,
                           Block *scopeBlock);
SmallVector<Operation *> computeLoadChain(Operation *markedOp, Block *scopeBlock);
memref::AllocOp findBackingAlloc(Operation *markedOp, ArrayRef<Operation *> chain);
SmallVector<MarkedLoad> collectMarkedOps(ModuleOp module);
std::optional<int64_t> getConstantTripCount(scf::ForOp forOp);
void groupByEnclosingForOp(SmallVector<MarkedLoad> &markedOps, SmallVectorImpl<ForBufferCtx> &contexts);
void collectChainAndMarkedOps(const SmallVector<MarkedLoad> &loads, llvm::DenseSet<Operation *> &allChainOps,
                              llvm::DenseSet<Operation *> &markedOps);
bool isResultUsedExternally(Value result, Operation *yieldOp, const llvm::DenseSet<Operation *> &allChainOps,
                            Block *forBody);
SmallVector<Value, kInitialRefCapacity> collectOperandsIncludingRegions(Operation *op);
llvm::DenseSet<Operation *> computeSkipInConsumer(const SmallVector<MarkedLoad> &loads, Block *forBody);

// ============================================================================
// Function declarations: Loop transformation utilities
// ============================================================================

bool isLoopInvariant(Value value, scf::ForOp forOp);
bool getLinearIterArgDelta(Value iterArg, Value yieldVal, scf::ForOp forOp, Value &delta);
Value castIndexTo(OpBuilder &builder, Location loc, Value value, Type targetType);
Value castToIndex(OpBuilder &builder, Location loc, Value value);
void tagWithBlockId(Value value, IntegerAttr blockId);
void allocateBufferSlots(OpBuilder &builder, Location loc, scf::ForOp forOp, LoadGroup &group);
ExtendedForInfo buildExtendedFor(OpBuilder &builder, Location loc, scf::ForOp forOp, ArrayRef<LoadGroup> groups,
                                 ConstantCache &cache);
bool isDeadIterArg(scf::ForOp forOp, unsigned iterArgIdx);
FailureOr<scf::ForOp> pruneDeadIterArgs(OpBuilder &builder, scf::ForOp forOp, unsigned candidateCount);
void eraseDeadBodyOps(Block *body);
bool isAvailableForClone(Value value, Block *oldBody, const IRMapping &mapping);
bool areOperandsAvailableForClone(Operation *op, Block *oldBody, const IRMapping &mapping);
std::optional<int32_t> getBlockId(Operation *op);
bool sameRunBlockId(const BodyRun &run, std::optional<int32_t> bid);
SmallVector<BodyRun> collectBodyRuns(Block *body);
int findRunContainingOp(ArrayRef<BodyRun> runs, Operation *target);
int findFirstRunWithBlockId(ArrayRef<BodyRun> runs, int32_t blockId);
void logRepeatedTopLevelBlockRuns(Block *body, llvm::StringRef stage);
bool isAncestorBlock(Block *ancestor, Block *descendant);
void deduplicateConstants(ModuleOp module);
LogicalResult applyMultiBufferToForLoop(ForBufferCtx &context, const llvm::DenseSet<Operation *> &allCtxForOps);

// ============================================================================
// Function declarations: Consumer/Producer emission
// ============================================================================

Value emitLoadSlotSelection(OpBuilder &builder, Location loc, const MarkedLoad &load, ArrayRef<Value> slotBufs,
                            Value target, int depth, IRMapping &mapping, ArrayRef<Value> slotConsts);
Value emitSlotFlagClear(OpBuilder &builder, Location loc, Value target, Value slotConst, Value flagNext, Value falseVal,
                        IntegerAttr blockId);
FailureOr<std::pair<Value, Value>> emitProducerFillForSlot(OpBuilder &builder, Location loc, Block *oldBody,
                                                           Block *newBody, const LoadGroup &group, int slotIdx,
                                                           Value flagArg, Value currentProducerCounter,
                                                           Value tripCount, Value falseVal, Value loopLowerBound,
                                                           Value loopStep, Value trueFlag, Value indexOne,
                                                           const SmallVector<Value> &iterArgDeltas,
                                                           scf::ForOp origForOp);
LogicalResult emitMultiBufferLoopBody(OpBuilder &builder, Location loc, SmallVectorImpl<LoadGroup> &groups,
                                      ExtendedForInfo &info, const SmallVector<Value> &groupIndexOneVals,
                                      const SmallVector<SmallVector<Value>> &allGroupSlotConsts,
                                      const SmallVector<Value> &allGroupDepthConsts,
                                      const llvm::DenseSet<Operation *> &allCtxForOps, Value tripCount,
                                      Value loopLowerBound, Value loopStep, const SmallVector<Value> &groupTrueFlagVals,
                                      const SmallVector<Value> &iterArgDeltas, scf::ForOp origForOp);

} // namespace gmload

#endif // TRITON_ADAPTER_ADD_MULTI_BUFFER_TO_GMLOAD_INTERNAL_H
