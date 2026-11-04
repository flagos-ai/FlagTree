#include "triton/Analysis/Membar.h"
#ifdef __ILUVATAR_TLE__
#include "triton/Dialect/Triton/IR/Dialect.h"
#endif
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/IR/LinearLayoutConversions.h"
#include "triton/Dialect/TritonNvidiaGPU/IR/Dialect.h"

#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Interfaces/ControlFlowInterfaces.h"
#include <deque>

#ifdef __ILUVATAR__
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "triton/Conversion/TritonGPUToLLVM/Utility.h"
#include <cstdlib>
#endif

namespace mlir {

void MembarOrFenceAnalysis::run(FuncBlockInfoMapT &funcBlockInfoMap) {
  FunctionOpInterface funcOp =
      dyn_cast<FunctionOpInterface>(allocation->getOperation());
  OpBuilder builder(funcOp.getContext());
  resolve(funcOp, &funcBlockInfoMap, &builder);
}

void MembarOrFenceAnalysis::resolve(FunctionOpInterface funcOp,
                                    FuncBlockInfoMapT *funcBlockInfoMap,
                                    OpBuilder *builder) {
  // Initialize the blockList. Operations are organized into "virtual blocks",
  // which represent segments of straight-line code analyzed by each iteration
  // of the dataflow analysis. Virtual blocks abstract over both control flow
  // represented by basic blocks and block successors (i.e. `BranchOpInterface`)
  // and control flow represented by regions (i.e. `RegionBranchOpInterface`).
  //
  // A virtual block consists of a parent block and a starting iterator, where
  // the virtual block starts on the operation *after* the starting iterator. A
  // null iterator is used to represent the beginning of the block. The virtual
  // block ends at any region branch operation or the basic block terminator.
  // Thus, basic blocks are broken up into multiple virtual blocks at each
  // region operation.
  //
  // Entry virtual blocks are represented by a null iterator. Populate the
  // blockList with the entry virtual blocks in the function. Then, each
  // iteration scans until a terminator or region branch operation is found.
  DenseMap<VirtualBlock, BlockInfo> inputBlockInfoMap;
  DenseMap<VirtualBlock, BlockInfo> outputBlockInfoMap;
  std::deque<VirtualBlock> blockList;
  funcOp.walk<WalkOrder::PreOrder>([&](Block *block) {
    // Start the analysis from the entry blocks of any nested isolated from
    // above regions.
    if (block->isEntryBlock() &&
        !isa<RegionBranchOpInterface>(block->getParentOp()))
      blockList.emplace_back(block, Block::iterator());
  });

  // A fixed point algorithm
  while (!blockList.empty()) {
    VirtualBlock block = blockList.front();
    blockList.pop_front();
    // Make a copy of the inputblockInfo but not update
    auto inputBlockInfo = inputBlockInfoMap[block];
    SmallVector<VirtualBlock> successors;
    Block::iterator startIt =
        block.second.isValid() ? std::next(block.second) : block.first->begin();
    for (Operation &op : llvm::make_range(startIt, block.first->end())) {
      if (op.hasTrait<OpTrait::IsTerminator>() ||
          isa<RegionBranchOpInterface>(op)) {
        visitTerminator(&op, successors);
        break;
      }
      update(&op, &inputBlockInfo, funcBlockInfoMap, builder);
    }
    // Get the reference because we want to update if it changed
    if (outputBlockInfoMap.count(block) &&
        inputBlockInfo == outputBlockInfoMap[block]) {
      // If we have seen the block before and the inputBlockInfo is the same as
      // the outputBlockInfo, we skip the successors
      continue;
    }
    // Update the current block. The block transfer function is not monotonic,
    // so overwrite the output state entirely.
    outputBlockInfoMap[block] = inputBlockInfo;
    // Update the successors
    for (VirtualBlock successor : successors) {
      inputBlockInfoMap[successor].join(outputBlockInfoMap[block]);
      blockList.emplace_back(successor);
    }
  }

  // Update the final dangling buffers that haven't been synced
  BlockInfo &funcBlockInfo = (*funcBlockInfoMap)[funcOp];
  funcOp.walk<WalkOrder::PreOrder>([&](triton::ReturnOp returnOp) {
    // A basic block can be broken into several virtual blocks. Find all virtual
    // blocks that belong to the basic block containing the return.
    SmallVector<std::pair<VirtualBlock, BlockInfo>> virtualBlocks;
    for (auto &[block, blockInfo] : outputBlockInfoMap) {
      if (block.first == returnOp->getBlock())
        virtualBlocks.emplace_back(block, blockInfo);
    }
    // The return is a terminator, so the virtual block that contains this
    // return starts after all other ones. Find it by comparing the start
    // iterators of the virtual blocks.
    auto maxIt = llvm::max_element(virtualBlocks, [&](auto &lhs, auto &rhs) {
      assert(lhs.first.first == rhs.first.first);
      Block::iterator lhsIt = lhs.first.second, rhsIt = rhs.first.second;
      return !lhsIt.isValid() ||
             (rhsIt.isValid() && lhsIt->isBeforeInBlock(&*rhsIt));
    });

    funcBlockInfo.join(maxIt->second);
  });
}

void MembarOrFenceAnalysis::visitTerminator(
    Operation *op, SmallVector<VirtualBlock> &successors) {
  if (isa<BranchOpInterface>(op)) {
    // Collect the block successors of the branch.
    for (Block *successor : op->getSuccessors())
      successors.emplace_back(successor, Block::iterator());
    return;
  }

  if (auto br = dyn_cast<RegionBranchOpInterface>(op)) {
    // The successors of an operation with regions can be queried via an
    // interface. The operation branches to the entry blocks of its region
    // successors. It can also branch to after itself.
    SmallVector<RegionSuccessor> regions;
    br.getSuccessorRegions(RegionBranchPoint::parent(), regions);
    for (RegionSuccessor &region : regions) {
      if (region.isParent()) {
        successors.emplace_back(br->getBlock(), br->getIterator());
      } else {
        Block &block = region.getSuccessor()->front();
        successors.emplace_back(&block, Block::iterator());
      }
    }
    return;
  }

  // FIXME: `ReturnLike` adds `RegionBranchTerminatorOpInterface` for some
  // reason. Check that the parent is actually a `RegionBranchOpInterface`.
  auto br = dyn_cast<RegionBranchTerminatorOpInterface>(op);
  if (br && isa<RegionBranchOpInterface>(br->getParentOp())) {
    // Check the successors of a region branch terminator. It can branch to
    // another region of its parent operation or to after the parent op.
    SmallVector<Attribute> operands(br->getNumOperands());
    SmallVector<RegionSuccessor> regions;
    br.getSuccessorRegions(operands, regions);
    for (RegionSuccessor &region : regions) {
      if (region.isParent()) {
        Operation *parent = br->getParentOp();
        successors.emplace_back(parent->getBlock(), parent->getIterator());
      } else {
        Block &block = region.getSuccessor()->front();
        successors.emplace_back(&block, Block::iterator());
      }
    }
    return;
  }

  // Otherwise, it could be a return op
  if (op->hasTrait<OpTrait::ReturnLike>())
    return;
  llvm_unreachable("Unknown terminator encountered in membar analysis");
}

void MembarAnalysis::insertBarrier(Operation *op, OpBuilder *builder) {
  OpBuilder::InsertionGuard g(*builder);
  auto barrierOp = triton::gpu::LocalBarrierOp::create(*builder, op->getLoc());
}

#ifdef __ILUVATAR__
namespace {
// Iluvatar SME barrier lightweighting.
//
// In an SME software-pipelined loop (num_stages > 1) the `ttg.async_wait`
// already lowers to `llvm.bi.sl.waitcnt` which drains the G2S queue (the SME
// global->shared engine). The CTA barrier membar inserts right after it then
// only needs to provide the cross-warp thread rendezvous, not the heavy
// `sl_barrier` (which additionally forces a full memory-system fence). So we
// replace that heavy barrier with a light `barrier.alu` (thread/warp sync
// only); the G2S drain is already covered by the async_wait's waitcnt.
// For reference the waitcnt queue-mask bits are bit2 = LM (regular shared
// stores) and bit3 = G2S (SME).
constexpr llvm::StringLiteral kIluvatarAluBarrierIntrin =
    "llvm.bi.sl.barrier.alu";

// Light CTA thread/warp sync only (no memory-queue drain).
void emitIluvatarAluBarrier(OpBuilder *builder, Location loc) {
  LLVM::createLLVMIntrinsicCallOp(*builder, loc, kIluvatarAluBarrierIntrin, {},
                                  {});
}

void emitIluvatarWaitAndAluBarrier(OpBuilder *builder, Location loc,
                                   int64_t waitCntValue) {
  auto i64Ty = builder->getI64Type();
  Value waitCnt = LLVM::ConstantOp::create(
      *builder, loc, i64Ty, builder->getIntegerAttr(i64Ty, waitCntValue));
  LLVM::createLLVMIntrinsicCallOp(*builder, loc, "llvm.bi.sl.waitcnt", {},
                                  {waitCnt});
  emitIluvatarAluBarrier(builder, loc);
}

bool isIluvatarSmeLocalAlloc(Operation *op) {
  auto alloc = dyn_cast<triton::gpu::LocalAllocOp>(op);
  if (!alloc || !alloc.getSrc())
    return false;
  auto srcTy = dyn_cast<RankedTensorType>(alloc.getSrc().getType());
  if (!srcTy)
    return false;
  auto blocked =
      dyn_cast<triton::gpu::BlockedEncodingAttr>(srcTy.getEncoding());
  return blocked && blocked.getIsSme();
}

// Recognize the `barrier.alu` we inject so the fixed-point membar traversal
// treats it as a real sync point. Otherwise re-visits of a loop body would not
// see it as a barrier, would not clear the pending intervals, and would insert
// a duplicate `barrier.alu` before the same op.
bool isIluvatarAluBarrier(Operation *op) {
  if (!op)
    return false;
  auto call = dyn_cast<LLVM::CallIntrinsicOp>(op);
  return call && call.getIntrin() == kIluvatarAluBarrierIntrin;
}

// An async copy produced by the Iluvatar SME pipeline has an SME blocked
// encoding and carries an explicit `inputStride`. The light
// `barrier.alu` after an `async_wait` (in place of the heavy CTA barrier) is
// only valid for the SME pipeline: there the SME G2S engine plus the
// `waitcnt` drain the data, and only a thread/warp sync is still needed. A
// non-SME pipelined loop (e.g. blocksparse) stages data through regular shared
// stores whose cross-warp visibility still requires the heavy barrier; using
// `barrier.alu` there under-synchronizes and corrupts results.
bool blockHasOnlySmeAsyncCopies(Operation *op) {
  Block *blk = op->getBlock();
  if (!blk)
    return false;
  bool foundAsyncCopy = false;
  for (Operation &o : *blk) {
    auto cp = dyn_cast<triton::gpu::AsyncCopyGlobalToLocalOp>(o);
    if (!cp)
      continue;
    foundAsyncCopy = true;
    if (!cp.isIluvatarSmeAsyncCopy())
      return false;
  }
  return foundAsyncCopy;
}

} // namespace
#endif

void MembarAnalysis::update(Operation *op, BlockInfo *blockInfo,
                            FuncBlockInfoMapT *funcBlockInfoMap,
                            OpBuilder *builder) {
  if (isa<gpu::BarrierOp, triton::gpu::LocalBarrierOp>(op)
#ifdef __ILUVATAR__
      || isIluvatarAluBarrier(op)
#endif
  ) {
    // If the current op is a barrier, we sync previous reads and writes
    blockInfo->sync();
    return;
  }

  if (isa<triton::gpu::AsyncWaitOp, triton::nvidia_gpu::TMAStoreWaitOp>(op) &&
      !isa<gpu::BarrierOp, triton::gpu::LocalBarrierOp>(op->getNextNode())
#ifdef __ILUVATAR__
      && !isIluvatarAluBarrier(op->getNextNode())
#endif
  ) {
    // If the current op is an async wait and the next op is not a barrier we
    // insert a barrier op and sync
    builder->setInsertionPointAfter(op);
#ifdef __ILUVATAR__
    // For an SME pipeline the async_wait already lowered to a `waitcnt` that
    // drains the G2S queue, so only a light thread/warp sync is needed here,
    // not a heavy CTA barrier (`sl_barrier`). This keeps the pipelined
    // (num_stages>1) SME loop light. Non-SME pipelined loops (regular cp.async)
    // still need the heavy barrier for cross-warp shared-memory visibility.
    if (blockHasOnlySmeAsyncCopies(op))
      emitIluvatarAluBarrier(builder, op->getLoc());
    else
      insertBarrier(op, builder);
#else
    insertBarrier(op, builder);
#endif
    blockInfo->sync();
    return;
  }

  BlockInfo curBlockInfo;
  auto scratchBufferId = Allocation::InvalidBufferId;
  if (isa<triton::CallOp>(op)) {
    // Inter-function dependencies
    auto callOpInterface = dyn_cast<CallOpInterface>(op);
    if (auto callee =
            dyn_cast<FunctionOpInterface>(callOpInterface.resolveCallable()))
      curBlockInfo = funcBlockInfoMap->lookup(callee);
  } else {
    // Intra-function dependencies
    if (auto memoryEffectOpInterface = dyn_cast<MemoryEffectOpInterface>(op)) {
      // Explicit buffer
      SmallVector<SideEffects::EffectInstance<MemoryEffects::Effect>>
          effectInstances;
      memoryEffectOpInterface.getEffects(effectInstances);
      for (auto effectInstance : effectInstances) {
        if (auto value = effectInstance.getValue()) {
          for (auto bufferId : allocation->getBufferIds(value)) {
            if (bufferId != Allocation::InvalidBufferId) {
              if (isa<MemoryEffects::Write>(effectInstance.getEffect()))
                curBlockInfo
                    .syncWriteIntervals[allocation->getAllocatedInterval(
                        bufferId)]
                    .insert(op);
              else if (isa<MemoryEffects::Read>(effectInstance.getEffect()))
                curBlockInfo
                    .syncReadIntervals[allocation->getAllocatedInterval(
                        bufferId)]
                    .insert(op);
            }
          }
        }
      }
    }
    // If this op is may be signalling other threads asynchronously, make sure
    // all shared memory transactions are complete beforehand.
    if (isa<triton::nvidia_gpu::ArriveBarrierOp>(op)) {
      Interval<size_t> allIntervals(0, std::numeric_limits<size_t>::max());
      curBlockInfo.syncWriteIntervals[allIntervals].insert(op);
      curBlockInfo.syncReadIntervals[allIntervals].insert(op);
    }
    scratchBufferId = allocation->getBufferId(op);
  }

#ifdef __ILUVATAR_TLE__
  // Preserve the 3.5 behavior for atomic chains in TLE mode: consecutive
  // atomics on overlapping shared intervals do not require an extra CTA
  // barrier here.
  MembarFilterFn effectiveFilter = [&](Operation *lhs, Operation *rhs) -> bool {
    if (isa<triton::AtomicRMWOp, triton::AtomicCASOp>(lhs) &&
        isa<triton::AtomicRMWOp, triton::AtomicCASOp>(rhs))
      return true;
    return filter ? filter(lhs, rhs) : false;
  };
#else
  MembarFilterFn effectiveFilter = filter;
#endif

  // Scratch buffer operations consist of a series of shared memory operations
  // starting from a shared memory write, followed by a series of shared memory
  // read/write operations, and ending with a shared memory read, i.e., shared
  // memory write -> ... -> shared memory read.
  bool handledIluvatarSmeReuse = false;
#ifdef __ILUVATAR__
  if (isIluvatarSmeLocalAlloc(op) &&
      blockInfo->isIntersected(curBlockInfo, filter)) {
    // The shared-memory allocator may reuse an interval after its SSA
    // lifetime ends, but an earlier SME transaction can still own that
    // address. Start a new SME epoch before writing the reused interval:
    // bit2 drains regular shared-memory stores and bit3 drains G2S, while the
    // ALU barrier provides the cross-warp rendezvous.
    builder->setInsertionPoint(op);
    emitIluvatarWaitAndAluBarrier(builder, op->getLoc(), /*LM|G2S=*/12);
    blockInfo->sync();
    handledIluvatarSmeReuse = true;
  }
#endif
  if (!handledIluvatarSmeReuse &&
      scratchBufferId != Allocation::InvalidBufferId) {
    // Detect warp-synchronous convert-layout operations. These emit a
    // warp-level barrier (warp.sync) rather than a CTA-wide barrier between
    // the internal shared-memory write and read phases. For these ops, we must
    // not globally clear pending dependencies.
    bool isWarpSync = false;
    if (auto cvt = dyn_cast<triton::gpu::ConvertLayoutOp>(op)) {
      auto srcTy = cast<RankedTensorType>(cvt.getSrc().getType());
      auto dstTy = cast<RankedTensorType>(cvt.getType());
      auto srcLayout = triton::gpu::toLinearLayout(srcTy);
      auto dstLayout = triton::gpu::toLinearLayout(dstTy);
      isWarpSync = mlir::isCvtWarpSync(srcLayout, dstLayout);
    }

#ifdef __ILUVATAR_TLE__
    // Some scratch-buffer ops can also carry explicit shared-memory effects.
    // Keep conservative dependency tracking instead of hard-failing here.
#else
    if (!curBlockInfo.syncReadIntervals.empty() ||
        !curBlockInfo.syncWriteIntervals.empty()) {
      llvm::report_fatal_error(
          "scratch buffer operations should not have any shared memory "
          "dependencies");
    }
#endif
    auto interval = allocation->getAllocatedInterval(scratchBufferId);
    curBlockInfo.syncWriteIntervals[interval].insert(op);
    auto insertCTABarrier =
        blockInfo->isIntersected(curBlockInfo, effectiveFilter);
    if (insertCTABarrier) {
      builder->setInsertionPoint(op);
      insertBarrier(op, builder);
    }
    // Ops with a scratch buffer that don't use warp.sync internally sync
    // read/write on shared memory
    if (insertCTABarrier || !isWarpSync)
      blockInfo->sync();
    curBlockInfo.syncReadIntervals[interval].insert(op);
  } else if (!handledIluvatarSmeReuse &&
             blockInfo->isIntersected(curBlockInfo, effectiveFilter)) {
    builder->setInsertionPoint(op);
    insertBarrier(op, builder);
    blockInfo->sync();
  }
  // Update the region info, even if barrier is inserted, we have to maintain
  // the current op's read/write buffers.
  blockInfo->join(curBlockInfo);
}
} // namespace mlir
