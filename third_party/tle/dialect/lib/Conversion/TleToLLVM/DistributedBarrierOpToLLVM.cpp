/*
 * Copyright 2025-     FlagOS Contributors
 *
 * Permission is hereby granted, free of charge, to any person obtaining
 * a copy of this software and associated documentation files
 * (the "Software"), to deal in the Software without restriction,
 * including without limitation the rights to use, copy, modify, merge,
 * publish, distribute, sublicense, and/or sell copies of the Software,
 * and to permit persons to whom the Software is furnished to do so,
 * subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be
 * included in all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
 * EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
 * MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
 * IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY
 * CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,
 * TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE
 * SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
 */

#include "tle/dialect/include/Conversion/TleToLLVM/DistributedBarrierOpToLLVM.h"
#include "tle/dialect/include/Tools/FlagcxUtils.h"

#include "mlir/Conversion/LLVMCommon/Pattern.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/LLVMIR/NVVMDialect.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "nvidia/include/Dialect/NVGPU/IR/Dialect.h"
#include "third_party/nvidia/include/TritonNVIDIAGPUToLLVM/PTXAsmFormat.h"
#include "tle/dialect/include/IR/Dialect.h"
#include "tle/dialect/include/Transforms/GridBarrierScratch.h"
#include "triton/Conversion/TritonGPUToLLVM/Utility.h"
#include "llvm/Support/MathExtras.h"
#include <algorithm>
#include <limits>
#include <optional>

namespace {

using namespace mlir;
using namespace mlir::triton;

constexpr llvm::StringLiteral kSpaceAttr = "space";
constexpr llvm::StringLiteral kOrderAttr = "order";
constexpr llvm::StringLiteral kIndexAttr = "barrier_index";
constexpr llvm::StringLiteral kContextIdAttr = "context_id";
constexpr llvm::StringLiteral kMemoryScopeAttr = "memory_scope";
constexpr llvm::StringLiteral kGroupKindAttr = "group_kind";
constexpr llvm::StringLiteral kGroupShapeAttr = "group_shape";
constexpr llvm::StringLiteral kGroupAxesAttr = "group_axes";
constexpr llvm::StringLiteral kGroupMaskAttr = "group_mask";
constexpr llvm::StringLiteral kGroupDomainShapeAttr = "group_domain_shape";
constexpr llvm::StringLiteral kTTGSharedAttr = "ttg.shared";
constexpr llvm::StringLiteral kSubmeshScratchOffsetAttr =
    "tle.submesh_barrier_scratch_offset";
constexpr int32_t kSubmeshScratchAlignment = 16;
constexpr int32_t kSubmeshScratchBytes = 8;
constexpr int32_t kSubmeshCounterOffsetBytes = 0;
constexpr int32_t kSubmeshPhaseOffsetBytes = 4;
constexpr int32_t kGridScratchBytes = 4;
constexpr int32_t kGridArrivedOffsetBytes = 0;

Value getDistDevicePtr(tle::DistributedBarrierOp op,
                       SmallVector<Value> &srcElems) {
  if (!srcElems.empty())
    return srcElems[0];
  else {
    auto func = op->getParentOfType<LLVM::LLVMFuncOp>();
    return func.getArgument(1);
  }
}

FailureOr<int32_t> getOrCreateSubmeshScratchOffset(ModuleOp mod) {
  if (auto existing =
          mod->getAttrOfType<IntegerAttr>(kSubmeshScratchOffsetAttr)) {
    int64_t value = existing.getInt();
    if (value < 0 || value > std::numeric_limits<int32_t>::max())
      return failure();
    return static_cast<int32_t>(value);
  }

  auto sharedAttr = mod->getAttrOfType<IntegerAttr>(kTTGSharedAttr);
  if (!sharedAttr)
    return failure();

  int64_t currentShared = sharedAttr.getInt();
  if (currentShared < 0)
    return failure();

  int64_t offset =
      llvm::alignTo(currentShared, int64_t{kSubmeshScratchAlignment});
  int64_t newShared = offset + kSubmeshScratchBytes;
  if (newShared > std::numeric_limits<int32_t>::max())
    return failure();

  auto i32Ty = IntegerType::get(mod.getContext(), 32);
  mod->setAttr(kTTGSharedAttr, IntegerAttr::get(i32Ty, newShared));
  mod->setAttr(kSubmeshScratchOffsetAttr, IntegerAttr::get(i32Ty, offset));
  return static_cast<int32_t>(offset);
}

struct DistributedBarrierOpConversion
    : public ConvertOpToLLVMPattern<tle::DistributedBarrierOp> {
  DistributedBarrierOpConversion(LLVMTypeConverter &typeConverter,
                                 PatternBenefit benefit,
                                 bool supportsAcquireFence)
      : ConvertOpToLLVMPattern(typeConverter, benefit),
        supportsAcquireFence(supportsAcquireFence) {}

  const bool supportsAcquireFence;

  LogicalResult lowerFlagsGridBarrier(
      tle::DistributedBarrierOp op, ConversionPatternRewriter &rewriter,
      Value groupBase, Value localRank, Value logicalTid) const {
    auto layout = tle::getFlagsGridLayout(op);
    if (failed(layout))
      return failure();
    auto loc = op.getLoc();
    auto *ctx = rewriter.getContext();
    TritonLLVMOpBuilder b(loc, rewriter);
    auto i32Ty = IntegerType::get(ctx, 32);
    auto i8Ty = IntegerType::get(ctx, 8);
    auto ptrTy = cast<LLVM::LLVMPointerType>(groupBase.getType());
    auto assembly = [&](StringRef code, StringRef constraints,
                        ValueRange inputs) -> Value {
      return LLVM::InlineAsmOp::create(rewriter, loc, i32Ty, inputs, code,
          constraints, true, false, LLVM::TailCallKind::None,
          LLVM::AsmDialectAttr::get(ctx, LLVM::AsmDialect::AD_ATT), ArrayAttr()).getResult(0);
    };
    Block *initial = rewriter.getInsertionBlock();
    Block *end = initial->splitBlock(rewriter.getInsertionPoint());
    Block *work = rewriter.createBlock(end);
    Block *round = rewriter.createBlock(end);
    round->addArgument(i32Ty, loc);
    Block *poll = rewriter.createBlock(end);
    Block *next = rewriter.createBlock(end);
    Block *acquired = rewriter.createBlock(end);

    rewriter.setInsertionPointToEnd(initial);
    rewriter.create<mlir::gpu::BarrierOp>(loc);
    Value scanner = b.icmp_ult(logicalTid, b.i32_val(32));
    rewriter.create<LLVM::CondBrOp>(loc, scanner, work, ValueRange{}, end, ValueRange{});

    rewriter.setInsertionPointToEnd(work);
    Value ownOffset = b.mul(localRank, b.i32_val(4));
    Value own = b.gep(ptrTy, i8Ty, groupBase, ownOffset);
    // A single owner reads its prior epoch. Broadcast precedes its only store,
    // so no peer lane derives a target from the newly published value.
    Value target = assembly(
        "{ .reg .pred leader; .reg .b32 previous, next;\n"
        "mov.b32 previous, 0; setp.eq.u32 leader, $2, 0;\n"
        "@leader ld.relaxed.gpu.u32 previous, [$1];\n"
        "add.u32 next, previous, 1;\n"
        "shfl.sync.idx.b32 $0, next, 0, 31, 0xffffffff;\n"
        "@leader st.release.gpu.u32 [$1], $0;\n}",
        "=&r,l,r,~{memory}", ValueRange{own, logicalTid});
    rewriter.create<LLVM::BrOp>(loc, ValueRange{b.i32_val(0)}, round);

    rewriter.setInsertionPointToEnd(round);
    Value roundIndex = round->getArgument(0);
    Value peer = b.add(b.mul(roundIndex, b.i32_val(32)), logicalTid);
    Value peerOffset = b.mul(peer, b.i32_val(4));
    Value address = b.gep(ptrTy, i8Ty, groupBase, peerOffset);
    rewriter.create<LLVM::BrOp>(loc, ValueRange{}, poll);

    rewriter.setInsertionPointToEnd(poll);
    // Fixed peer-to-lane mapping preserves the preceding successful strong
    // read's coherence lower bound on subsequent epochs. Together with the
    // one-generation publication lead bound, modular subtraction handles
    // wrap/fast overwrites without accepting ancient coherent values.
    Value observed = assembly(
        "{ .reg .pred valid; mov.b32 $0, $4;\n"
        "setp.lt.u32 valid, $2, $3;\n"
        "@valid ld.relaxed.gpu.u32 $0, [$1];\n}",
        "=&r,l,r,r,r,~{memory}",
        ValueRange{address, peer, b.i32_val(layout->participants), target});
    Value reached = b.icmp_ult(b.sub(observed, target), b.i32_val(0x80000000u));
    Value all = assembly(
        "{ .reg .pred reached, complete; setp.ne.u32 reached, $1, 0;\n"
        "vote.sync.all.pred complete, reached, 0xffffffff;\n"
        "selp.u32 $0, 1, 0, complete;\n}",
        "=r,r", ValueRange{b.zext(i32Ty, reached)});
    rewriter.create<LLVM::CondBrOp>(loc, b.icmp_ne(all, b.i32_val(0)),
                                   next, ValueRange{}, poll, ValueRange{});

    rewriter.setInsertionPointToEnd(next);
    Value nextRound = b.add(roundIndex, b.i32_val(1));
    Value more = b.icmp_ult(nextRound, b.i32_val((layout->participants + 31) / 32));
    rewriter.create<LLVM::CondBrOp>(loc, more, round, ValueRange{nextRound},
                                   acquired, ValueRange{});

    rewriter.setInsertionPointToEnd(acquired);
    // Every acquiring lane executes this once, after every successful peer
    // read. The original role tail barrier relays all lanes' acquired writes.
    PTXBuilder ptx;
    auto &fence = *ptx.create<>("fence.acquire.gpu");
    fence();
    ptx.launch(rewriter, loc, LLVM::LLVMVoidType::get(ctx));
    rewriter.create<LLVM::BrOp>(loc, ValueRange{}, end);
    rewriter.setInsertionPointToStart(end);
    rewriter.create<mlir::gpu::BarrierOp>(loc);
    rewriter.eraseOp(op);
    return success();
  }

  LogicalResult lowerHierarchicalGridBarrier(
      tle::DistributedBarrierOp op, ConversionPatternRewriter &rewriter,
      Value groupBase, Value localRank, Value expectedCTAs,
      Value workerPred) const {
    auto loc = op.getLoc();
    auto *ctx = rewriter.getContext();
    TritonLLVMOpBuilder b(loc, rewriter);
    auto i32Ty = IntegerType::get(ctx, 32);
    auto i8Ty = IntegerType::get(ctx, 8);
    auto ptrTy = cast<LLVM::LLVMPointerType>(groupBase.getType());
    auto emitLoad = [&](Value ptr) -> Value {
      PTXBuilder ptx;
      auto &load = *ptx.create<>("ld.relaxed.gpu.u32");
      auto *result = ptx.newOperand("=r", true);
      auto *address = ptx.newAddrOperand(ptr, "l");
      load(result, address);
      return ptx.launch(rewriter, loc, i32Ty);
    };
    auto emitArrival = [&](Value ptr, Value add) -> Value {
      PTXBuilder ptx;
      auto &atom = *ptx.create<>("atom.add.release.gpu.u32");
      auto *result = ptx.newOperand("=r", true);
      auto *address = ptx.newAddrOperand(ptr, "l");
      auto *increment = ptx.newOperand(add, "r");
      atom(result, address, increment);
      return ptx.launch(rewriter, loc, i32Ty);
    };
    auto emitAcquire = [&]() {
      PTXBuilder ptx;
      auto &fence = *ptx.create<>("fence.acquire.gpu");
      fence();
      ptx.launch(rewriter, loc, LLVM::LLVMVoidType::get(ctx));
    };
    auto phaseDiffers = [&](Value before, Value now) {
      return b.icmp_ne(b.and_(b.xor_(before, now), b.i32_val(0x80000000u)),
                       b.i32_val(0));
    };

    Block *initial = rewriter.getInsertionBlock();
    Block *end = initial->splitBlock(rewriter.getInsertionPoint());
    Block *work = rewriter.createBlock(end);
    Block *leafWait = rewriter.createBlock(end);
    Block *rootArrival = rewriter.createBlock(end);
    Block *rootWait = rewriter.createBlock(end);
    Block *acquired = rewriter.createBlock(end);

    rewriter.setInsertionPointToEnd(initial);
    rewriter.create<mlir::gpu::BarrierOp>(loc);
    rewriter.create<LLVM::CondBrOp>(loc, workerPred, work, ValueRange{},
                                    end, ValueRange{});

    rewriter.setInsertionPointToEnd(work);
    Value buckets = b.select(b.icmp_ult(expectedCTAs,
                                      b.i32_val(tle::kHierarchicalGridBuckets)),
                             expectedCTAs, b.i32_val(tle::kHierarchicalGridBuckets));
    Value bucket = b.urem(localRank, buckets);
    Value leader = b.icmp_ult(localRank, buckets);
    Value leafSize = b.add(b.i32_val(1), b.udiv(
        b.sub(b.sub(expectedCTAs, b.i32_val(1)), bucket), buckets));
    Value leafOffset =
        b.mul(bucket, b.i32_val(tle::kHierarchicalGridSlotBytes));
    Value rootOffset = b.i32_val(tle::kHierarchicalGridRootOffset);
    Value leaf = b.gep(ptrTy, i8Ty, groupBase, leafOffset);
    Value root = b.gep(ptrTy, i8Ty, groupBase, rootOffset);
    // This read must precede this CTA's leaf arrival. A post-arrival snapshot
    // could observe this epoch's completed root sense and wait for the next.
    Value rootPhase = emitLoad(root);
    Value leafAdd = b.select(leader,
        b.sub(b.i32_val(0x80000000u), b.sub(leafSize, b.i32_val(1))),
        b.i32_val(1));
    Value oldLeaf = emitArrival(leaf, leafAdd);
    rewriter.create<LLVM::CondBrOp>(loc, leader, leafWait, ValueRange{},
                                    rootWait, ValueRange{});

    rewriter.setInsertionPointToEnd(leafWait);
    Value leafComplete = phaseDiffers(oldLeaf, emitLoad(leaf));
    rewriter.create<LLVM::CondBrOp>(loc, leafComplete, rootArrival, ValueRange{},
                                    leafWait, ValueRange{});

    rewriter.setInsertionPointToEnd(rootArrival);
    // Import every leaf release before publishing this leaf at the root.
    emitAcquire();
    Value rootAdd = b.select(b.icmp_eq(bucket, b.i32_val(0)),
        b.sub(b.i32_val(0x80000000u), b.sub(buckets, b.i32_val(1))),
        b.i32_val(1));
    emitArrival(root, rootAdd);
    rewriter.create<LLVM::BrOp>(loc, ValueRange{}, rootWait);

    rewriter.setInsertionPointToEnd(rootWait);
    Value rootComplete = phaseDiffers(rootPhase, emitLoad(root));
    rewriter.create<LLVM::CondBrOp>(loc, rootComplete, acquired, ValueRange{},
                                    rootWait, ValueRange{});

    rewriter.setInsertionPointToEnd(acquired);
    emitAcquire();
    rewriter.create<LLVM::BrOp>(loc, ValueRange{}, end);

    rewriter.setInsertionPointToStart(end);
    rewriter.create<mlir::gpu::BarrierOp>(loc);
    rewriter.eraseOp(op);
    return success();
  }

  LogicalResult lowerClusterBarrier(tle::DistributedBarrierOp op,
                                    ConversionPatternRewriter &rewriter) const {
    auto *ctx = rewriter.getContext();
    auto unit = UnitAttr::get(ctx);
    // Cluster arrive/wait does not provide CTA-wide synchronization semantics
    // for local shared-memory hazards. Add CTA barriers around it so
    // distributed_barrier behaves as a full barrier for each participating CTA.
    rewriter.create<mlir::gpu::BarrierOp>(op.getLoc());
    rewriter.create<NVVM::ClusterArriveOp>(op.getLoc(), unit);
    rewriter.create<NVVM::ClusterWaitOp>(op.getLoc(), unit);
    rewriter.create<mlir::gpu::BarrierOp>(op.getLoc());
    rewriter.eraseOp(op);
    return success();
  }

  LogicalResult lowerGridBarrier(tle::DistributedBarrierOp op,
                                 ConversionPatternRewriter &rewriter,
                                 Value censusOccurrence = {}) const {
    auto loc = op.getLoc();
    auto *ctx = rewriter.getContext();
    TritonLLVMOpBuilder b(loc, rewriter);
    auto i8Ty = IntegerType::get(ctx, 8);
    auto i32Ty = IntegerType::get(ctx, 32);

    if (tle::isHierarchicalGridBarrier(op) && !supportsAcquireFence)
      return op.emitOpError("hierarchical8 requires NVIDIA SM90+ and PTX ISA8.6+");
    if (tle::isFlagsGridBarrier(op) && !supportsAcquireFence)
      return op.emitOpError("flags32 requires NVIDIA SM90+ and PTX ISA8.6+");

    auto mod = op->getParentOfType<ModuleOp>();
    if (!mod)
      return op.emitOpError("cannot find parent module for grid lowering");

    SmallVector<int32_t> groupShape;
    SmallVector<int32_t> groupAxes;
    SmallVector<int32_t> domainShape;
    if (auto shapeAttr = op->getAttrOfType<DenseI32ArrayAttr>(kGroupShapeAttr))
      groupShape.assign(shapeAttr.asArrayRef().begin(),
                        shapeAttr.asArrayRef().end());
    if (auto axesAttr = op->getAttrOfType<DenseI32ArrayAttr>(kGroupAxesAttr))
      groupAxes.assign(axesAttr.asArrayRef().begin(),
                       axesAttr.asArrayRef().end());
    if (auto domainAttr =
            op->getAttrOfType<DenseI32ArrayAttr>(kGroupDomainShapeAttr))
      domainShape.assign(domainAttr.asArrayRef().begin(),
                         domainAttr.asArrayRef().end());
    const bool isAxisGroup = !domainShape.empty();
    int32_t participantCount = 1;
    SmallVector<int32_t> extentByAxis(domainShape.size(), 1);
    if (isAxisGroup) {
      if (groupShape.size() != groupAxes.size())
        return op.emitOpError("grid axis group shape/axes rank mismatch");
      for (auto [axis, extent] : llvm::zip(groupAxes, groupShape)) {
        if (axis < 0 || axis >= static_cast<int32_t>(domainShape.size()) ||
            extent <= 0 || domainShape[axis] % extent != 0)
          return op.emitOpError("invalid grid axis group descriptor");
        extentByAxis[axis] = extent;
        participantCount *= extent;
      }
    }

    auto scratchOffsetAttr =
        op->getAttrOfType<IntegerAttr>("ttg.global_scratch_memory_offset");
    if (!scratchOffsetAttr)
      return op.emitOpError("grid barrier requires global scratch allocation "
                            "before LLVM lowering");
    int64_t scratchOffsetValue = scratchOffsetAttr.getInt();
    if (scratchOffsetValue < 0 ||
        scratchOffsetValue > std::numeric_limits<int32_t>::max())
      return op.emitOpError("grid scratch offset is out of i32 range");
    int32_t scratchOffset = static_cast<int32_t>(scratchOffsetValue);

    auto func = op->getParentOfType<LLVM::LLVMFuncOp>();
    if (!func) {
      return op.emitOpError("grid lowering requires LLVM function context");
    }
    int32_t argIdx = static_cast<int32_t>(func.getNumArguments()) +
                     kGlobalScratchBufferOffset;
    if (argIdx < 0 || argIdx >= static_cast<int32_t>(func.getNumArguments())) {
      return op.emitOpError(
          "cannot locate global scratch argument for grid barrier lowering");
    }
    Value globalScratchBase = func.getArgument(static_cast<unsigned>(argIdx));
    auto globalPtrTy =
        dyn_cast<LLVM::LLVMPointerType>(globalScratchBase.getType());
    if (!globalPtrTy) {
      return op.emitOpError("global scratch argument must be an LLVM pointer");
    }

    auto globalI32PtrTy =
        LLVM::LLVMPointerType::get(ctx, globalPtrTy.getAddressSpace());
    Value linearBlockId = rewriter.create<NVVM::BlockIdZOp>(loc, i32Ty);
    Value gridDimX = rewriter.create<NVVM::GridDimXOp>(loc, i32Ty);
    Value gridDimY = rewriter.create<NVVM::GridDimYOp>(loc, i32Ty);
    Value blockIdX = rewriter.create<NVVM::BlockIdXOp>(loc, i32Ty);
    Value blockIdY = rewriter.create<NVVM::BlockIdYOp>(loc, i32Ty);
    linearBlockId = b.add(b.mul(linearBlockId, gridDimY), blockIdY);
    linearBlockId = b.add(b.mul(linearBlockId, gridDimX), blockIdX);

    Value groupIndex = b.i32_val(0);
    Value localRank = b.i32_val(0);
    if (isAxisGroup) {
      int32_t stride = 1;
      SmallVector<int32_t> strides(domainShape.size(), 1);
      for (int32_t axis = static_cast<int32_t>(domainShape.size()) - 1;
           axis >= 0; --axis) {
        strides[axis] = stride;
        stride *= domainShape[axis];
      }
      for (int32_t axis = 0; axis < static_cast<int32_t>(domainShape.size());
           ++axis) {
        Value coord = linearBlockId;
        if (strides[axis] != 1)
          coord = b.udiv(coord, b.i32_val(strides[axis]));
        if (domainShape[axis] != 1)
          coord = b.urem(coord, b.i32_val(domainShape[axis]));
        else
          coord = b.i32_val(0);
        int32_t extent = extentByAxis[axis];
        int32_t groupsOnAxis = domainShape[axis] / extent;
        Value groupCoord = coord;
        if (extent != 1)
          groupCoord = b.udiv(coord, b.i32_val(extent));
        groupIndex = b.add(b.mul(groupIndex, b.i32_val(groupsOnAxis)),
                           groupCoord);
        if (extent != 1) {
          Value localCoord = b.urem(coord, b.i32_val(extent));
          localRank = b.add(b.mul(localRank, b.i32_val(extent)),
                            localCoord);
        }
      }
    }
    Value scratchByteOffset = b.i32_val(scratchOffset +
                                        kGridArrivedOffsetBytes);
    if (isAxisGroup)
      scratchByteOffset =
          b.add(scratchByteOffset,
                b.mul(groupIndex, b.i32_val(tle::getGridBarrierGroupBytes(op))));
    Value arrivedBytePtr = b.gep(globalPtrTy, i8Ty, globalScratchBase,
                                 scratchByteOffset);
    Value arrivedPtr = b.bitcast(arrivedBytePtr, globalI32PtrTy);

    Value threadId = getThreadId(rewriter, loc);
    Value isThread0 = b.icmp_eq(threadId, b.i32_val(0));
    Value blockIdZ = rewriter.create<NVVM::BlockIdZOp>(loc, i32Ty);
    Value isBlock0;
    if (isAxisGroup) {
      isBlock0 = b.icmp_eq(localRank, b.i32_val(0));
    } else {
      isBlock0 = b.and_(b.and_(b.icmp_eq(blockIdX, b.i32_val(0)),
                               b.icmp_eq(blockIdY, b.i32_val(0))),
                        b.icmp_eq(blockIdZ, b.i32_val(0)));
    }
    Value workerPred = isThread0;
    auto emitCensus = [&](int32_t point) {
      if (!censusOccurrence)
        return;
      tle::TraceEventOp::create(rewriter, loc, censusOccurrence, threadId,
          b.i32_val(0), b.i32_val(0), rewriter.getI32IntegerAttr(280),
          rewriter.getI32IntegerAttr(point));
    };

    Value gridDimZ = rewriter.create<NVVM::GridDimZOp>(loc, i32Ty);
    Value totalCTAs = b.mul(gridDimX, gridDimY);
    totalCTAs = b.mul(totalCTAs, gridDimZ);
    Value expectedCTAs =
        isAxisGroup ? b.i32_val(participantCount) : totalCTAs;

    if (tle::isHierarchicalGridBarrier(op))
      return lowerHierarchicalGridBarrier(op, rewriter, arrivedPtr,
          isAxisGroup ? localRank : linearBlockId, expectedCTAs, workerPred);
    if (tle::isFlagsGridBarrier(op))
      return lowerFlagsGridBarrier(op, rewriter, arrivedPtr,
          isAxisGroup ? localRank : linearBlockId, threadId);

    Block *curBlock = rewriter.getInsertionBlock();
    Block *endBlock = curBlock->splitBlock(rewriter.getInsertionPoint());
    Block *workBlock = rewriter.createBlock(endBlock);
    Block *waitBlock = rewriter.createBlock(endBlock);
    waitBlock->addArgument(i32Ty, loc); // old_arrive
    Block *doneBlock = rewriter.createBlock(endBlock);
    Block *workerDoneBlock = rewriter.createBlock(endBlock);

    rewriter.setInsertionPointToEnd(curBlock);
    rewriter.create<mlir::gpu::BarrierOp>(loc);
    rewriter.create<LLVM::CondBrOp>(loc, workerPred, workBlock, ValueRange{},
                                    doneBlock, ValueRange{});

    rewriter.setInsertionPointToEnd(workBlock);
    Value expectedMinusOne = b.sub(expectedCTAs, b.i32_val(1));
    Value gpuMasterAdd = b.sub(b.i32_val(0x80000000u), expectedMinusOne);
    Value nb = b.select(isBlock0, gpuMasterAdd, b.i32_val(1));

    auto emitAtomAddReleaseGpu = [&](Value ptr, Value addVal) -> Value {
      ::mlir::triton::PTXBuilder ptxBuilder;
      auto &atom = *ptxBuilder.create<>("atom.add.release.gpu.u32");
      auto *dstOpr = ptxBuilder.newOperand("=r", /*init=*/true);
      auto *ptrOpr = ptxBuilder.newAddrOperand(ptr, "l");
      auto *addOpr = ptxBuilder.newOperand(addVal, "r");
      atom(dstOpr, ptrOpr, addOpr);
      return ptxBuilder.launch(rewriter, loc, i32Ty);
    };
    auto emitLoadPollGpu = [&](Value ptr) -> Value {
      ::mlir::triton::PTXBuilder ptxBuilder;
      auto &ld = *ptxBuilder.create<>(supportsAcquireFence
                                          ? "ld.relaxed.gpu.u32"
                                          : "ld.acquire.gpu.u32");
      auto *dstOpr = ptxBuilder.newOperand("=r", /*init=*/true);
      auto *ptrOpr = ptxBuilder.newAddrOperand(ptr, "l");
      ld(dstOpr, ptrOpr);
      return ptxBuilder.launch(rewriter, loc, i32Ty);
    };

    emitCensus(0);
    Value oldArrive = emitAtomAddReleaseGpu(arrivedPtr, nb);
    rewriter.create<LLVM::BrOp>(loc, ValueRange{oldArrive}, waitBlock);

    rewriter.setInsertionPointToEnd(waitBlock);
    Value oldArriveArg = waitBlock->getArgument(0);
    Value currentArrive = emitLoadPollGpu(arrivedPtr);
    Value xorVal = b.xor_(oldArriveArg, currentArrive);
    Value flippedBit = b.and_(xorVal, b.i32_val(0x80000000u));
    Value hasFlipped = b.icmp_ne(flippedBit, b.i32_val(0));
    rewriter.create<LLVM::CondBrOp>(loc, hasFlipped, workerDoneBlock,
                                    ValueRange{}, waitBlock,
                                    ValueRange{oldArriveArg});

    rewriter.setInsertionPointToEnd(workerDoneBlock);
    if (supportsAcquireFence) {
      // The successful strong read followed by this fence forms the acquire
      // pattern. Failed polls need no acquire; the trailing role barrier still
      // publishes the leader's acquired writes to every participating thread.
      ::mlir::triton::PTXBuilder ptxBuilder;
      auto &fence = *ptxBuilder.create<>("fence.acquire.gpu");
      fence();
      ptxBuilder.launch(rewriter, loc, LLVM::LLVMVoidType::get(ctx));
    }
    emitCensus(1);
    rewriter.create<LLVM::BrOp>(loc, ValueRange{}, endBlock);

    rewriter.setInsertionPointToEnd(doneBlock);
    rewriter.create<LLVM::BrOp>(loc, ValueRange{}, endBlock);

    rewriter.setInsertionPointToStart(endBlock);
    rewriter.create<mlir::gpu::BarrierOp>(loc);
    emitCensus(2);
    rewriter.eraseOp(op);
    return success();
  }

  LogicalResult lowerSubmeshBarrier(tle::DistributedBarrierOp op,
                                    ConversionPatternRewriter &rewriter) const {
    auto maskAttr = op->getAttrOfType<DenseI32ArrayAttr>(kGroupMaskAttr);
    if (!maskAttr) {
      return op.emitOpError("submesh lowering requires static group_mask attr");
    }

    SmallVector<int32_t> subgroupMask(maskAttr.asArrayRef().begin(),
                                      maskAttr.asArrayRef().end());
    if (subgroupMask.empty()) {
      return op.emitOpError("submesh lowering requires non-empty group_mask");
    }
    if (llvm::any_of(subgroupMask, [](int32_t v) { return v < 0; })) {
      return op.emitOpError(
          "submesh lowering requires non-negative group_mask entries");
    }

    if (auto shapeAttr =
            op->getAttrOfType<DenseI32ArrayAttr>(kGroupShapeAttr)) {
      int64_t subgroupFromShape = 1;
      for (int32_t dim : shapeAttr.asArrayRef())
        subgroupFromShape *= dim;
      if (subgroupFromShape != static_cast<int64_t>(subgroupMask.size())) {
        return op.emitOpError() << "group_shape product (" << subgroupFromShape
                                << ") must match group_mask size ("
                                << subgroupMask.size() << ")";
      }
    }

    auto loc = op.getLoc();
    auto *ctx = rewriter.getContext();
    TritonLLVMOpBuilder b(loc, rewriter);
    auto i8Ty = IntegerType::get(ctx, 8);
    auto i32Ty = IntegerType::get(ctx, 32);

    auto mod = op->getParentOfType<ModuleOp>();
    if (!mod)
      return op.emitOpError("cannot find parent module for submesh lowering");

    auto scratchOffsetOr = getOrCreateSubmeshScratchOffset(mod);
    if (failed(scratchOffsetOr)) {
      return op.emitOpError(
          "failed to reserve shared memory scratch for submesh barrier");
    }
    int32_t scratchOffset = *scratchOffsetOr;

    auto globalSmem = mod.lookupSymbol<LLVM::GlobalOp>("global_smem");
    if (!globalSmem) {
      return op.emitOpError("global_smem symbol is missing; submesh barrier "
                            "lowering requires shared memory base");
    }

    auto sharedPtrTy = LLVM::LLVMPointerType::get(
        ctx, static_cast<unsigned>(NVVM::NVVMMemorySpace::Shared));
    auto clusterPtrTy = LLVM::LLVMPointerType::get(
        ctx, static_cast<unsigned>(NVVM::NVVMMemorySpace::SharedCluster));

    Value sharedBase = rewriter.create<LLVM::AddressOfOp>(loc, globalSmem);
    sharedBase = b.bitcast(sharedBase, sharedPtrTy);

    Value counterLocalPtr =
        b.gep(sharedPtrTy, i8Ty, sharedBase,
              b.i32_val(scratchOffset + kSubmeshCounterOffsetBytes));
    Value phaseLocalPtr =
        b.gep(sharedPtrTy, i8Ty, sharedBase,
              b.i32_val(scratchOffset + kSubmeshPhaseOffsetBytes));

    int32_t leaderCTAId = subgroupMask.front();
    Value leaderCTA = b.i32_val(leaderCTAId);
    Value counterPtr = rewriter.create<NVVM::MapaOp>(
        loc, clusterPtrTy, counterLocalPtr, leaderCTA);
    Value phasePtr = rewriter.create<NVVM::MapaOp>(loc, clusterPtrTy,
                                                   phaseLocalPtr, leaderCTA);

    Value clusterCTAId = rewriter.create<triton::nvgpu::ClusterCTAIdOp>(loc);
    Value isParticipant = b.false_val();
    for (int32_t member : subgroupMask) {
      Value isMember = b.icmp_eq(clusterCTAId, b.i32_val(member));
      isParticipant = b.or_(isParticipant, isMember);
    }

    Value threadId = getThreadId(rewriter, loc);
    Value isThread0 = b.icmp_eq(threadId, b.i32_val(0));
    Value workerPred = b.and_(isParticipant, isThread0);
    Value isLeaderCTA = b.icmp_eq(clusterCTAId, leaderCTA);
    Value doInit = b.and_(isLeaderCTA, isThread0);

    auto unit = UnitAttr::get(ctx);
    Block *entryBlock = rewriter.getInsertionBlock();
    Block *postInitBlock = entryBlock->splitBlock(rewriter.getInsertionPoint());
    Block *initBlock = rewriter.createBlock(postInitBlock);

    rewriter.setInsertionPointToEnd(entryBlock);
    // Conservative initialization fence: ensure all CTAs observe counter reset
    // before any subgroup arrivals in this barrier instance.
    rewriter.create<NVVM::ClusterArriveOp>(loc, unit);
    rewriter.create<NVVM::ClusterWaitOp>(loc, unit);
    rewriter.create<LLVM::CondBrOp>(loc, doInit, initBlock, ValueRange{},
                                    postInitBlock, ValueRange{});

    rewriter.setInsertionPointToEnd(initBlock);
    rewriter.create<LLVM::StoreOp>(loc, b.i32_val(0), counterPtr);
    rewriter.create<LLVM::BrOp>(loc, ValueRange{}, postInitBlock);

    rewriter.setInsertionPointToStart(postInitBlock);
    rewriter.create<NVVM::ClusterArriveOp>(loc, unit);
    rewriter.create<NVVM::ClusterWaitOp>(loc, unit);

    Block *curBlock = rewriter.getInsertionBlock();
    Block *endBlock = curBlock->splitBlock(rewriter.getInsertionPoint());
    Block *workBlock = rewriter.createBlock(endBlock);
    Block *waitBlock = rewriter.createBlock(endBlock);
    waitBlock->addArgument(i32Ty, loc);
    Block *lastBlock = rewriter.createBlock(endBlock);
    Block *doneBlock = rewriter.createBlock(endBlock);

    rewriter.setInsertionPointToEnd(curBlock);
    rewriter.create<mlir::gpu::BarrierOp>(loc);
    rewriter.create<LLVM::CondBrOp>(loc, workerPred, workBlock, ValueRange{},
                                    doneBlock, ValueRange{});

    rewriter.setInsertionPointToEnd(workBlock);
    Value oldPhase =
        rewriter
            .create<LLVM::AtomicRMWOp>(
                loc, LLVM::AtomicBinOp::add, phasePtr, b.i32_val(0),
                LLVM::AtomicOrdering::acquire, StringRef("device"))
            .getResult();
    Value prevCount =
        rewriter
            .create<LLVM::AtomicRMWOp>(
                loc, LLVM::AtomicBinOp::add, counterPtr, b.i32_val(1),
                LLVM::AtomicOrdering::acq_rel, StringRef("device"))
            .getResult();
    Value arrived = b.add(prevCount, b.i32_val(1));
    Value isLast = b.icmp_eq(arrived, b.i32_val(subgroupMask.size()));
    rewriter.create<LLVM::CondBrOp>(loc, isLast, lastBlock, ValueRange{},
                                    waitBlock, ValueRange{oldPhase});

    rewriter.setInsertionPointToEnd(waitBlock);
    Value expectedPhase = waitBlock->getArgument(0);
    Value currentPhase =
        rewriter
            .create<LLVM::AtomicRMWOp>(
                loc, LLVM::AtomicBinOp::add, phasePtr, b.i32_val(0),
                LLVM::AtomicOrdering::acquire, StringRef("device"))
            .getResult();
    Value keepWaiting = b.icmp_eq(currentPhase, expectedPhase);
    rewriter.create<LLVM::CondBrOp>(loc, keepWaiting, waitBlock,
                                    ValueRange{expectedPhase}, doneBlock,
                                    ValueRange{});

    rewriter.setInsertionPointToEnd(lastBlock);
    rewriter.create<LLVM::AtomicRMWOp>(
        loc, LLVM::AtomicBinOp::add, counterPtr,
        b.i32_val(-static_cast<int64_t>(subgroupMask.size())),
        LLVM::AtomicOrdering::release, StringRef("device"));
    rewriter.create<LLVM::AtomicRMWOp>(
        loc, LLVM::AtomicBinOp::add, phasePtr, b.i32_val(1),
        LLVM::AtomicOrdering::release, StringRef("device"));
    rewriter.create<LLVM::BrOp>(loc, ValueRange{}, doneBlock);

    rewriter.setInsertionPointToEnd(doneBlock);
    rewriter.create<LLVM::BrOp>(loc, ValueRange{}, endBlock);

    rewriter.setInsertionPointToStart(endBlock);
    rewriter.create<mlir::gpu::BarrierOp>(loc);
    rewriter.eraseOp(op);
    return success();
  }

  LogicalResult
  lowerFlagCxSpaceBarrier(tle::DistributedBarrierOp op, OpAdaptor adaptor,
                          ConversionPatternRewriter &rewriter) const {
    auto spaceAttr = op->getAttrOfType<StringAttr>(kSpaceAttr);
    auto kindAttr = op->getAttrOfType<StringAttr>(kGroupKindAttr);
    auto orderAttr = op.getOrderAttr();
    auto indexAttr = op->getAttrOfType<IntegerAttr>(kIndexAttr);
    auto contextIdAttr = op->getAttrOfType<IntegerAttr>(kContextIdAttr);
    auto memoryScopeAttr = op.getMemoryScopeAttr();
    auto loc = op.getLoc();
    SmallVector<Value> srcElems;
    auto getTeamKind =
        [](StringRef space) -> std::optional<tle::FlagCXTeamKind> {
      return llvm::StringSwitch<std::optional<tle::FlagCXTeamKind>>(space)
          .Case("device", tle::FlagCXTeamKind::INTRA)
          .Case("inter", tle::FlagCXTeamKind::INTER)
          .Case("world", tle::FlagCXTeamKind::WORLD)
          .Default(std::nullopt);
    };
    auto getCoopKind =
        [](StringRef kind) -> std::optional<tle::FlagCXCoopKind> {
      return llvm::StringSwitch<std::optional<tle::FlagCXCoopKind>>(kind)
          .Case("thread", tle::FlagCXCoopKind::THREAD)
          .Case("warp", tle::FlagCXCoopKind::WARP)
          .Case("block", tle::FlagCXCoopKind::BLOCK)
          .Default(std::nullopt);
    };

    auto teamKind = getTeamKind(spaceAttr.getValue());
    auto coopKind = getCoopKind(kindAttr.getValue());
    if (!teamKind)
      return rewriter.notifyMatchFailure(op, "invalid FlagCX team space");
    if (!coopKind)
      return rewriter.notifyMatchFailure(op, "invalid coop_kind");

    if (auto src = adaptor.getSrc())
      srcElems = unpackLLElements(loc, src, rewriter);

    auto comm = getDistDevicePtr(op, srcElems);
    auto teamKindAttr =
        tle::FlagCXTeamKindAttr::get(rewriter.getContext(), *teamKind);
    auto coopKindAttr =
        tle::FlagCXCoopKindAttr::get(rewriter.getContext(), *coopKind);
    auto barrierTypeAttr = op.getBarrierTypeAttr();
#ifdef FLAGCX_ENABLED
    rewriter.replaceOpWithNewOp<tle::FlagCxBarrierOp>(
        op, comm, barrierTypeAttr, teamKindAttr, coopKindAttr, indexAttr,
        contextIdAttr, orderAttr, memoryScopeAttr);
    return success();
#else
    return rewriter.notifyMatchFailure(
        op, "FlagCX support is required for communicator barriers");
#endif
  }
  LogicalResult
  matchAndRewrite(tle::DistributedBarrierOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    if (op->getAttrOfType<StringAttr>(kSpaceAttr))
      return lowerFlagCxSpaceBarrier(op, adaptor, rewriter);

    if (auto kindAttr = op->getAttrOfType<StringAttr>(kGroupKindAttr)) {
      if (kindAttr.getValue() == "grid" ||
          kindAttr.getValue() == "grid_axis")
        return lowerGridBarrier(op, rewriter,
                                op->hasAttr("tle.census") ? adaptor.getSrc() : Value());
      if (kindAttr.getValue() == "submesh")
        return lowerSubmeshBarrier(op, rewriter);
      return lowerClusterBarrier(op, rewriter);
    }
    return lowerClusterBarrier(op, rewriter);
  }
};

struct CensusMarkOpConversion : public ConvertOpToLLVMPattern<tle::CensusMarkOp> {
  using ConvertOpToLLVMPattern::ConvertOpToLLVMPattern;
  LogicalResult matchAndRewrite(tle::CensusMarkOp op, OpAdaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    TritonLLVMOpBuilder b(op.getLoc(), rewriter);
    Value logicalTid = getThreadId(rewriter, op.getLoc());
    rewriter.replaceOpWithNewOp<tle::TraceEventOp>(op, adaptor.getOccurrence(), logicalTid,
        b.i32_val(0), b.i32_val(0), rewriter.getI32IntegerAttr(op.getMapping() ? 281 : 280),
        op.getPointAttr());
    return success();
  }
};

} // namespace

void tle::populateDistributedBarrierOpToLLVMPatterns(
    LLVMTypeConverter &typeConverter, RewritePatternSet &patterns,
    PatternBenefit benefit, bool supportsAcquireFence) {
  patterns.add<DistributedBarrierOpConversion>(typeConverter, benefit,
                                             supportsAcquireFence);
  patterns.add<CensusMarkOpConversion>(typeConverter, benefit);
}
