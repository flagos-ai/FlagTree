#include "triton/Analysis/Membar.h"
#ifdef __TLE__
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "tle/dialect/include/IR/Dialect.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#endif
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/IR/LinearLayoutConversions.h"
#include "triton/Dialect/TritonNvidiaGPU/IR/Dialect.h"

#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Interfaces/ControlFlowInterfaces.h"
#include <deque>

namespace mlir {

#ifdef __TLE__
namespace {

namespace tt = mlir::triton;
namespace ttg = mlir::triton::gpu;

struct StaticAccessView {
  Value root;
  SmallVector<int64_t> offsets;
  SmallVector<int64_t> sizes;
  int64_t rank = 0;
};

struct StaticIndexCoverage {
  int64_t offset = 0;
  int64_t size = 1;
};

static Value stripConvertLayouts(Value value) {
  Value current = value;
  while (auto cvt = current.getDefiningOp<ttg::ConvertLayoutOp>())
    current = cvt.getSrc();
  return current;
}

static Value stripIndexValueWrappers(Value value) {
  Value current = value;
  while (true) {
    if (auto cvt = current.getDefiningOp<ttg::ConvertLayoutOp>()) {
      current = cvt.getSrc();
      continue;
    }
    if (auto ext = current.getDefiningOp<arith::ExtSIOp>()) {
      current = ext.getIn();
      continue;
    }
    if (auto ext = current.getDefiningOp<arith::ExtUIOp>()) {
      current = ext.getIn();
      continue;
    }
    if (auto trunc = current.getDefiningOp<arith::TruncIOp>()) {
      current = trunc.getIn();
      continue;
    }
    if (auto cast = current.getDefiningOp<arith::IndexCastOp>()) {
      current = cast.getIn();
      continue;
    }
    break;
  }
  return current;
}

static std::optional<int64_t> getConstantIntLike(Value value) {
  value = stripIndexValueWrappers(value);
  if (auto splat = value.getDefiningOp<tt::SplatOp>())
    return getConstantIntLike(splat.getSrc());
  if (auto cst = value.getDefiningOp<arith::ConstantOp>()) {
    if (auto dense = dyn_cast<DenseIntElementsAttr>(cst.getValue())) {
      if (dense.isSplat())
        return dense.getSplatValue<APInt>().getSExtValue();
    }
  }
  if (auto cst = value.getDefiningOp<arith::ConstantIntOp>())
    return cst.value();
  if (auto cst = value.getDefiningOp<arith::ConstantIndexOp>())
    return cst.value();
  return std::nullopt;
}

static std::optional<StaticIndexCoverage>
matchRangeWithStaticOffset(Value value) {
  Value current = stripIndexValueWrappers(value);
  if (auto range = current.getDefiningOp<tt::MakeRangeOp>())
    return StaticIndexCoverage{range.getStart(),
                               range.getEnd() - range.getStart()};

  auto add = current.getDefiningOp<arith::AddIOp>();
  if (!add)
    return std::nullopt;

  auto tryMatch = [&](Value lhs,
                      Value rhs) -> std::optional<StaticIndexCoverage> {
    Value lhsStripped = stripIndexValueWrappers(lhs);
    auto range = lhsStripped.getDefiningOp<tt::MakeRangeOp>();
    if (!range)
      return std::nullopt;
    std::optional<int64_t> cst = getConstantIntLike(rhs);
    if (!cst)
      return std::nullopt;
    return StaticIndexCoverage{range.getStart() + *cst,
                               range.getEnd() - range.getStart()};
  };

  if (auto coverage = tryMatch(add.getLhs(), add.getRhs()))
    return coverage;
  return tryMatch(add.getRhs(), add.getLhs());
}

static std::optional<StaticIndexCoverage>
matchStaticIndexCoverage(Value index) {
  if (auto cst = getConstantIntLike(index))
    return StaticIndexCoverage{*cst, 1};

  Value current = stripIndexValueWrappers(index);
  if (auto coverage = matchRangeWithStaticOffset(current))
    return coverage;

  if (auto bcast = current.getDefiningOp<tt::BroadcastOp>())
    current = stripIndexValueWrappers(bcast.getSrc());

  while (auto expand = current.getDefiningOp<tt::ExpandDimsOp>())
    current = stripIndexValueWrappers(expand.getSrc());

  return matchRangeWithStaticOffset(current);
}

static std::optional<StaticAccessView> getStaticMemDescView(Value value) {
  auto memDescTy = dyn_cast<ttg::MemDescType>(value.getType());
  if (!memDescTy)
    return std::nullopt;

  if (auto index = value.getDefiningOp<ttg::MemDescIndexOp>()) {
    auto srcView = getStaticMemDescView(index.getSrc());
    if (!srcView)
      return std::nullopt;
    auto cstIndex = getConstantIntLike(index.getIndex());
    if (!cstIndex)
      return std::nullopt;

    auto srcTy = index.getSrc().getType();
    int64_t rootRank = srcView->offsets.size();
    int64_t srcRank = srcTy.getRank();
    int64_t axis = rootRank - srcRank;
    if (axis < 0 || axis >= rootRank)
      return std::nullopt;

    srcView->offsets[axis] += *cstIndex;
    srcView->sizes[axis] = 1;
    srcView->rank = memDescTy.getRank();
    auto dstShape = memDescTy.getShape();
    for (auto [i, dimSize] : llvm::enumerate(dstShape))
      srcView->sizes[axis + 1 + static_cast<int64_t>(i)] = dimSize;
    return srcView;
  }

  if (auto subslice = value.getDefiningOp<ttg::MemDescSubsliceOp>()) {
    auto srcView = getStaticMemDescView(subslice.getSrc());
    if (!srcView)
      return std::nullopt;

    auto srcTy = subslice.getSrc().getType();
    int64_t rootRank = srcView->offsets.size();
    int64_t srcRank = srcTy.getRank();
    int64_t firstAxis = rootRank - srcRank;
    if (firstAxis < 0)
      return std::nullopt;

    for (auto [i, offset] : llvm::enumerate(subslice.getOffsets()))
      srcView->offsets[firstAxis + static_cast<int64_t>(i)] += offset;
    auto dstShape = memDescTy.getShape();
    for (auto [i, dimSize] : llvm::enumerate(dstShape))
      srcView->sizes[firstAxis + static_cast<int64_t>(i)] = dimSize;
    srcView->rank = memDescTy.getRank();
    return srcView;
  }

  SmallVector<int64_t> shape(memDescTy.getShape().begin(),
                             memDescTy.getShape().end());
  return StaticAccessView{value, SmallVector<int64_t>(shape.size(), 0),
                          std::move(shape), memDescTy.getRank()};
}

static std::optional<StaticAccessView> getStaticLocalPointerView(Value value) {
  Value ptr = stripConvertLayouts(value);
  auto localPointers = ptr.getDefiningOp<tt::tle::LocalPointersOp>();
  if (!localPointers)
    return std::nullopt;

  auto srcView = getStaticMemDescView(localPointers.getSrc());
  if (!srcView)
    return std::nullopt;
  auto srcTy = localPointers.getSrc().getType();
  if (localPointers.getIndices().empty())
    return srcView;
  if (localPointers.getIndices().size() != static_cast<size_t>(srcTy.getRank()))
    return std::nullopt;

  int64_t rootRank = srcView->offsets.size();
  int64_t firstAxis = rootRank - srcTy.getRank();
  if (firstAxis < 0)
    return std::nullopt;

  for (auto [axis, index] : llvm::enumerate(localPointers.getIndices())) {
    auto coverage = matchStaticIndexCoverage(index);
    if (!coverage)
      return std::nullopt;
    int64_t rootAxis = firstAxis + static_cast<int64_t>(axis);
    srcView->offsets[rootAxis] += coverage->offset;
    srcView->sizes[rootAxis] = coverage->size;
  }
  srcView->rank = srcTy.getRank();
  return srcView;
}

static std::optional<StaticAccessView> getStaticAccessView(Value value) {
  if (isa<ttg::MemDescType>(value.getType()))
    return getStaticMemDescView(value);
  return getStaticLocalPointerView(value);
}

static std::optional<unsigned> getElementByteWidth(Type type) {
  if (auto intTy = dyn_cast<IntegerType>(type))
    return llvm::divideCeil(intTy.getWidth(), 8);
  if (auto floatTy = dyn_cast<FloatType>(type))
    return llvm::divideCeil(floatTy.getWidth(), 8);
  if (isa<tt::PointerType>(type))
    return 8;
  return std::nullopt;
}

static size_t linearizeStatic(ArrayRef<int64_t> coord, ArrayRef<unsigned> shape,
                              ArrayRef<unsigned> order) {
  size_t linear = 0;
  for (unsigned dim : llvm::reverse(order))
    linear = linear * shape[dim] + coord[dim];
  return linear;
}

static bool containsBufferId(const Allocation::BufferIdSetT &bufferIds,
                             Allocation::BufferId bufferId) {
  return llvm::any_of(bufferIds,
                      [&](Allocation::BufferId id) { return id == bufferId; });
}

static std::optional<SmallVector<Interval<size_t>>>
getStaticAccessIntervals(const Allocation *allocation, Value value,
                         Allocation::BufferId bufferId) {
  auto view = getStaticAccessView(value);
  if (!view)
    return std::nullopt;
  if (!containsBufferId(allocation->getBufferIds(view->root), bufferId))
    return std::nullopt;

  auto rootTy = dyn_cast<ttg::MemDescType>(view->root.getType());
  if (!rootTy)
    return std::nullopt;
  auto elemBytes = getElementByteWidth(rootTy.getElementType());
  if (!elemBytes)
    return std::nullopt;

  SmallVector<unsigned> shape;
  ArrayRef<int64_t> typeShape = rootTy.getAllocShape();
  if (typeShape.empty())
    typeShape = rootTy.getShape();
  shape.reserve(typeShape.size());
  for (int64_t dim : typeShape) {
    if (dim <= 0)
      return std::nullopt;
    shape.push_back(static_cast<unsigned>(dim));
  }
  if (shape.size() != view->offsets.size() ||
      shape.size() != view->sizes.size())
    return std::nullopt;

  for (auto [offset, size, dim] :
       llvm::zip_equal(view->offsets, view->sizes, shape)) {
    if (offset < 0 || size <= 0 || offset + size > static_cast<int64_t>(dim))
      return std::nullopt;
  }

  auto order = ttg::getOrder(rootTy);
  if (order.size() != shape.size())
    return std::nullopt;

  size_t outerCount = 1;
  unsigned fastestDim = order.front();
  for (unsigned dim = 0, rank = shape.size(); dim < rank; ++dim) {
    if (dim == fastestDim)
      continue;
    outerCount *= static_cast<size_t>(view->sizes[dim]);
    if (outerCount > 4096)
      return std::nullopt;
  }

  SmallVector<unsigned> outerDims;
  for (unsigned dim = 0, rank = shape.size(); dim < rank; ++dim)
    if (dim != fastestDim)
      outerDims.push_back(dim);

  SmallVector<Interval<size_t>> intervals;
  intervals.reserve(std::max<size_t>(outerCount, 1));
  size_t base = allocation->getAllocatedInterval(bufferId).start();
  for (size_t linearOuter = 0; linearOuter < std::max<size_t>(outerCount, 1);
       ++linearOuter) {
    size_t residue = linearOuter;
    SmallVector<int64_t> start(view->offsets.begin(), view->offsets.end());
    SmallVector<int64_t> end = start;
    for (unsigned dim : llvm::reverse(outerDims)) {
      int64_t size = view->sizes[dim];
      int64_t idx = residue % size;
      residue /= size;
      start[dim] += idx;
      end[dim] += idx;
    }
    end[fastestDim] = start[fastestDim] + view->sizes[fastestDim] - 1;
    size_t startElem = linearizeStatic(start, shape, order);
    size_t endElem = linearizeStatic(end, shape, order) + 1;
    if (startElem > endElem)
      return std::nullopt;
    intervals.emplace_back(base + startElem * *elemBytes,
                           base + endElem * *elemBytes);
  }
  return intervals;
}

static void addEffectIntervals(const Allocation *allocation, Value value,
                               bool isWrite, Operation *op,
                               BlockInfo &curBlockInfo) {
  for (auto bufferId : allocation->getBufferIds(value)) {
    if (bufferId == Allocation::InvalidBufferId)
      continue;
    auto intervals = getStaticAccessIntervals(allocation, value, bufferId);
    if (!intervals)
      intervals = SmallVector<Interval<size_t>, 1>{
          allocation->getAllocatedInterval(bufferId)};
    for (auto interval : *intervals) {
      if (isWrite)
        curBlockInfo.syncWriteIntervals[interval].insert(op);
      else
        curBlockInfo.syncReadIntervals[interval].insert(op);
    }
  }
}

} // namespace
#endif

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
#ifdef __TLE__
          // TLE local_ptr and memdesc views can denote disjoint static subviews
          // of the same shared allocation. Track those subviews as byte
          // intervals so membar only inserts CTA barriers for real overlaps.
          if (isa<MemoryEffects::Write>(effectInstance.getEffect()))
            addEffectIntervals(allocation, value, /*isWrite=*/true, op,
                               curBlockInfo);
          else if (isa<MemoryEffects::Read>(effectInstance.getEffect()))
            addEffectIntervals(allocation, value, /*isWrite=*/false, op,
                               curBlockInfo);
#else
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
#endif
        }
      }
    }
    // If this op is may be signalling other threads asynchronously, make sure
    // all shared memory transactions are complete beforehand.
#ifdef __TLE__
    if (auto arrive = dyn_cast<triton::nvidia_gpu::ArriveBarrierOp>(op)) {
      // `participant_arrive` is only emitted after the pipe lowering has proven
      // the prefix set of writer lanes. Those lanes each execute their own
      // release fence and one-unit mbarrier arrive, so adding a CTA rendezvous
      // here would duplicate the publication edge that the op already models.
      // Non-participant/elected arrives still need this conservative
      // all-shared-memory dependency because one elected lane cannot publish
      // stores performed by other lanes without a preceding barrier.
      if (!arrive.getParticipantArrive()) {
        Interval<size_t> allIntervals(0, std::numeric_limits<size_t>::max());
        curBlockInfo.syncWriteIntervals[allIntervals].insert(op);
        curBlockInfo.syncReadIntervals[allIntervals].insert(op);
      }
    }
#else
    if (isa<triton::nvidia_gpu::ArriveBarrierOp>(op)) {
      Interval<size_t> allIntervals(0, std::numeric_limits<size_t>::max());
      curBlockInfo.syncWriteIntervals[allIntervals].insert(op);
      curBlockInfo.syncReadIntervals[allIntervals].insert(op);
    }
#endif
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
