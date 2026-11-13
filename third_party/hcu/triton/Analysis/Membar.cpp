#include "triton/Analysis/Membar.h"
#include "triton/Analysis/PaddedSlotGeometry.h"
#ifdef __TLE__
#include "triton/Dialect/Triton/IR/Dialect.h"
#endif
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/IR/LinearLayoutConversions.h"
#include "triton/Dialect/TritonNvidiaGPU/IR/Dialect.h"

#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Interfaces/ControlFlowInterfaces.h"
#include "mlir/IR/Matchers.h"
#include <deque>

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


namespace {
// Use the forwarded-successor interface, as in TensorMemoryAllocation. Only
// identical SSA values on every edge are accepted; unknown and cyclic paths
// keep the original whole-allocation dependency.
static Value resolveIdenticalIncoming(Value value) {
  SmallVector<Value, 8> seen;
  for (unsigned depth = 0; depth < 8; ++depth) {
    auto arg = dyn_cast<BlockArgument>(value);
    if (!arg)
      return value;
    if (llvm::is_contained(seen, value))
      return {};
    seen.push_back(value);
    Block *block = arg.getOwner();
    Value incoming;
    for (Block *pred : block->getPredecessors()) {
      auto branch = dyn_cast<BranchOpInterface>(pred->getTerminator());
      if (!branch)
        return {};
      for (unsigned i = 0; i < branch->getNumSuccessors(); ++i) {
        if (branch->getSuccessor(i) != block)
          continue;
        auto operands = branch.getSuccessorOperands(i);
        unsigned produced = operands.getProducedOperandCount();
        unsigned number = arg.getArgNumber();
        if (number < produced ||
            number - produced >= operands.getForwardedOperands().size())
          return {};
        Value next = operands.getForwardedOperands()[number - produced];
        if (incoming && incoming != next)
          return {};
        incoming = next;
      }
    }
    if (!incoming)
      return {};
    value = incoming;
  }
  return {};
}

// Narrow counterpart of common Membar's static-access intervals. The existing
// MemDescIndex lowering advances by getAllocationShapePerCTA(result) elements.
// A full swizzled plane stays inside this interval; arbitrary subviews or padded
// layouts must not be linearized using their logical rectangular bounds.
static Interval<size_t> getFullSlotInterval(const Allocation *allocation,
                                           Value value,
                                           Allocation::BufferId bufferId,
                                           Operation *access) {
  auto whole = allocation->getAllocatedInterval(bufferId);
  // Independent default-off stage1. The option requests extra precision, not
  // permission to skip any geometry/alias/CFG check. Legacy path stays intact.
  auto mod = allocation->getOperation()->getParentOfType<ModuleOp>();
  auto enabled = mod ? mod->getAttrOfType<BoolAttr>(
                           "ttg.hcu.padded_slot_intervals") : BoolAttr{};
  auto target = mod ? mod->getAttrOfType<StringAttr>("ttg.target") : StringAttr{};
  auto ctas = mod ? mod->getAttrOfType<IntegerAttr>("ttg.num-ctas") : IntegerAttr{};
  auto localLoad = dyn_cast<triton::gpu::LocalLoadOp>(access);
  auto localStore = dyn_cast<triton::gpu::LocalStoreOp>(access);
  bool ordinaryAccess =
      (localLoad && !localLoad.getToken() && localLoad.getSrc() == value) ||
      (localStore && localStore.getDst() == value);
  if (ordinaryAccess && enabled && enabled.getValue() && target &&
      target.getValue() == "hip:gfx936" && ctas && ctas.getInt() == 1) {
    Value plane = resolveIdenticalIncoming(value);
    // Only bijective full-plane transposes may precede the index. Arbitrary
    // slices, reinterpretations and different root lifetimes stay whole.
    for (unsigned depth = 0; plane && depth < 8; ++depth) {
      auto trans = plane.getDefiningOp<triton::gpu::MemDescTransOp>();
      if (!trans)
        break;
      if (!triton::gpu::isFullPaddedSlotTranspose(trans)) {
        plane = {};
        break;
      }
      plane = resolveIdenticalIncoming(trans.getSrc());
    }
    auto index = plane ? plane.getDefiningOp<triton::gpu::MemDescIndexOp>()
                       : triton::gpu::MemDescIndexOp{};
    Value root = index ? resolveIdenticalIncoming(index.getSrc()) : Value{};
    auto alloc = root ? root.getDefiningOp<triton::gpu::LocalAllocOp>()
                      : triton::gpu::LocalAllocOp{};
    auto geometry = alloc && alloc.isSharedMemoryAlloc() &&
                            allocation->getBufferId(root) == bufferId
                        ? triton::gpu::getPaddedSlotGeometry(alloc.getType(),
                                                            index.getType())
                        : std::nullopt;
    llvm::APInt slot;
    if (geometry && matchPattern(index.getIndex(), m_ConstantInt(&slot)) &&
        slot.getBitWidth() <= 32 && !slot.isNegative() &&
        slot.getZExtValue() < geometry->slots &&
        geometry->rootBytes <= allocation->getAllocatedSize(bufferId)) {
      uint64_t relative, end, start;
      if (triton::gpu::paddedSlotMul(slot.getZExtValue(), geometry->strideBytes,
                                     relative) &&
          triton::gpu::paddedSlotAdd(relative, geometry->footprintBytes, end) &&
          end <= geometry->rootBytes &&
          triton::gpu::paddedSlotAdd(whole.start(), relative, start) &&
          start <= std::numeric_limits<size_t>::max() &&
          triton::gpu::paddedSlotAdd(whole.start(), end, end) &&
          end <= std::numeric_limits<size_t>::max())
        return Interval<size_t>(static_cast<size_t>(start),
                                static_cast<size_t>(end));
    }
  }
  value = resolveIdenticalIncoming(value);
  if (!value)
    return whole;
  auto index = value.getDefiningOp<triton::gpu::MemDescIndexOp>();
  if (!index)
    return whole;
  Value root = resolveIdenticalIncoming(index.getSrc());
  if (!root)
    return whole;
  auto alloc = root.getDefiningOp<triton::gpu::LocalAllocOp>();
  if (!alloc || !alloc.isSharedMemoryAlloc() ||
      allocation->getBufferId(root) != bufferId)
    return whole;
  auto srcTy = alloc.getType();
  auto dstTy = index.getType();
  if (srcTy.getRank() != 3 || dstTy.getRank() != 2 ||
      !isa<triton::gpu::SwizzledSharedEncodingAttr>(srcTy.getEncoding()) ||
      srcTy.getEncoding() != dstTy.getEncoding() ||
      srcTy.getShape() != srcTy.getAllocShape() ||
      dstTy.getShape() != dstTy.getAllocShape() ||
      srcTy.getShape().drop_front() != dstTy.getShape() ||
      ArrayRef<int64_t>(triton::gpu::getAllocationShapePerCTA(srcTy)) !=
          srcTy.getShape() ||
      ArrayRef<int64_t>(triton::gpu::getAllocationShapePerCTA(dstTy)) !=
          dstTy.getShape())
    return whole;
  Type elemTy = srcTy.getElementType();
  if (!elemTy.isIntOrFloat())
    return whole;
  unsigned bits = elemTy.getIntOrFloatBitWidth();
  llvm::APInt slot;
  if (!bits || bits % 8 ||
      !matchPattern(index.getIndex(), m_ConstantInt(&slot)) ||
      slot.getBitWidth() > 64 || slot.isNegative() ||
      slot.getZExtValue() >= static_cast<uint64_t>(srcTy.getShape()[0]))
    return whole;
  size_t total = allocation->getAllocatedSize(bufferId);
  size_t stride = bits / 8;
  for (int64_t dim : dstTy.getShape()) {
    if (dim <= 0 || stride > total / static_cast<size_t>(dim))
      return whole;
    stride *= dim;
  }
  int64_t slots = srcTy.getShape()[0];
  if (slots <= 0 || stride > total / static_cast<size_t>(slots) ||
      stride * slots != total)
    return whole;
  size_t start = whole.start() + slot.getZExtValue() * stride;
  return Interval<size_t>(start, start + stride);
}
} // namespace

void MembarAnalysis::update(Operation *op, BlockInfo *blockInfo,
                            FuncBlockInfoMapT *funcBlockInfoMap,
                            OpBuilder *builder) {
  if (isa<gpu::BarrierOp, triton::gpu::LocalBarrierOp>(op)) {
    // If the current op is a barrier, we sync previous reads and writes
    blockInfo->sync();
    return;
  }

  if (isa<triton::gpu::AsyncWaitOp, triton::nvidia_gpu::TMAStoreWaitOp>(op) &&
      !isa<gpu::BarrierOp, triton::gpu::LocalBarrierOp>(op->getNextNode())) {
    // If the current op is an async wait and the next op is not a barrier we
    // insert a barrier op and sync
    builder->setInsertionPointAfter(op);
    insertBarrier(op, builder);
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
              auto interval = getFullSlotInterval(allocation, value, bufferId, op);
              if (isa<MemoryEffects::Write>(effectInstance.getEffect()))
                curBlockInfo
                    .syncWriteIntervals[interval]
                    .insert(op);
              else if (isa<MemoryEffects::Read>(effectInstance.getEffect()))
                curBlockInfo
                    .syncReadIntervals[interval]
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

#ifdef __TLE__
  // Preserve the 3.5 behavior for atomic chains in TLE mode: consecutive
  // atomics on overlapping shared intervals do not require an extra CTA
  // barrier here.
  auto effectiveFilter = [&](Operation *lhs, Operation *rhs) -> bool {
    if (isa<triton::AtomicRMWOp, triton::AtomicCASOp>(lhs) &&
        isa<triton::AtomicRMWOp, triton::AtomicCASOp>(rhs))
      return true;
    return filter ? filter(lhs, rhs) : false;
  };
#endif

  // Scratch buffer operations consist of a series of shared memory operations
  // starting from a shared memory write, followed by a series of shared memory
  // read/write operations, and ending with a shared memory read, i.e., shared
  // memory write -> ... -> shared memory read.
  if (scratchBufferId != Allocation::InvalidBufferId) {
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

#ifdef __TLE__
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
#ifdef __TLE__
    auto insertCTABarrier =
        blockInfo->isIntersected(curBlockInfo, effectiveFilter);
#else
    auto insertCTABarrier = blockInfo->isIntersected(curBlockInfo, filter);
#endif
    if (insertCTABarrier) {
      builder->setInsertionPoint(op);
      insertBarrier(op, builder);
    }
    // Ops with a scratch buffer that don't use warp.sync internally sync
    // read/write on shared memory
    if (insertCTABarrier || !isWarpSync)
      blockInfo->sync();
    curBlockInfo.syncReadIntervals[interval].insert(op);
#ifdef __TLE__
  } else if (blockInfo->isIntersected(curBlockInfo, effectiveFilter)) {
#else
  } else if (blockInfo->isIntersected(curBlockInfo, filter)) {
#endif
    builder->setInsertionPoint(op);
    insertBarrier(op, builder);
    blockInfo->sync();
  }
  // Update the region info, even if barrier is inserted, we have to maintain
  // the current op's read/write buffers.
  blockInfo->join(curBlockInfo);
}
} // namespace mlir
