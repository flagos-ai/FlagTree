#include "Analysis/Membar.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/Interfaces/CallInterfaces.h"
#include "mlir/Interfaces/ControlFlowInterfaces.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "tsingmicro-tx81/Dialect/IR/Tx81Ops.h"

#include <cstdint>
#include <deque>
#include <limits>

namespace mlir::triton::membar {
using namespace mlir;

static bool isIntersectedMap(const BlockInfo::IntervalMapT &lhsIntervalSet,
                             const BlockInfo::IntervalMapT &rhsIntervalSet,
                             MembarFilterFn filter, MembarHazardKind kind) {
  for (auto &lhs : lhsIntervalSet)
    for (auto &rhs : rhsIntervalSet)
      if (lhs.first.intersects(rhs.first))
        for (auto lhsOp : lhs.second)
          for (auto rhsOp : rhs.second)
            if (!filter || !filter(lhsOp, rhsOp, kind))
              return true;
  return false;
}

bool BlockInfo::isIntersected(const BlockInfo &other,
                              MembarFilterFn filter) const {
  return isIntersectedMap(syncWriteIntervals, other.syncReadIntervals, filter,
                          MembarHazardKind::WriteRead) ||
         isIntersectedMap(syncReadIntervals, other.syncWriteIntervals, filter,
                          MembarHazardKind::ReadWrite) ||
         isIntersectedMap(syncWriteIntervals, other.syncWriteIntervals, filter,
                          MembarHazardKind::WriteWrite);
}

// -----------------------------------------------------------------------------
// Helpers: resolve i64/index address computations back to a memref Value
// (avoid touching Alias.cpp; keep local).
// -----------------------------------------------------------------------------

static Value getBaseBuffer(Value v) {
  while (auto *defOp = v.getDefiningOp()) {
    if (auto op = dyn_cast<memref::SubViewOp>(defOp))
      v = op.getSource();
    else if (auto op = dyn_cast<memref::CastOp>(defOp))
      v = op.getSource();
    else if (auto op = dyn_cast<memref::ReinterpretCastOp>(defOp))
      v = op.getSource();
    else if (auto op = dyn_cast<memref::ViewOp>(defOp))
      v = op.getSource();
    else if (auto op = dyn_cast<memref::ExtractStridedMetadataOp>(defOp)) {
      if (v == op->getResult(0))
        v = op.getSource();
      else
        break;
    } else
      break;
  }
  return v;
}

static Value traceToOriginMemRef(Value v, unsigned maxDepth = 16) {
  if (maxDepth == 0)
    return {};
  if (isa<MemRefType>(v.getType()))
    return getBaseBuffer(v);
  Operation *defOp = v.getDefiningOp();
  if (!defOp)
    return {};

  if (auto op = dyn_cast<arith::IndexCastOp>(defOp))
    return traceToOriginMemRef(op.getIn(), maxDepth - 1);
  if (auto op = dyn_cast<arith::TruncIOp>(defOp))
    return traceToOriginMemRef(op.getIn(), maxDepth - 1);
  if (auto op = dyn_cast<arith::ExtSIOp>(defOp))
    return traceToOriginMemRef(op.getIn(), maxDepth - 1);
  if (auto op = dyn_cast<arith::ExtUIOp>(defOp))
    return traceToOriginMemRef(op.getIn(), maxDepth - 1);

  if (auto op = dyn_cast<memref::ExtractAlignedPointerAsIndexOp>(defOp))
    return getBaseBuffer(op.getSource());

  if (isa<arith::AddIOp, arith::SubIOp, arith::MulIOp, arith::OrIOp>(defOp)) {
    if (Value r = traceToOriginMemRef(defOp->getOperand(0), maxDepth - 1))
      return r;
    return traceToOriginMemRef(defOp->getOperand(1), maxDepth - 1);
  }
  return {};
}

static Value resolveForBufferLookup(Value v) {
  if (!v)
    return {};
  if (isa<MemRefType>(v.getType()))
    return getBaseBuffer(v);
  if (Value origin = traceToOriginMemRef(v))
    return origin;
  return {};
}

static bool getAllocationOffsetInterval(Value v,
                                        BlockInfo::IntervalT &interval) {
  Value base = resolveForBufferLookup(v);
  if (!base)
    return false;

  auto alloc = base.getDefiningOp<memref::AllocOp>();
  if (!alloc)
    return false;

  auto offsetAttr = alloc->getAttrOfType<IntegerAttr>("allocation.offset");
  if (!offsetAttr)
    return false;

  int64_t signedOffset = offsetAttr.getInt();
  if (signedOffset < 0)
    return false;

  MemRefType allocType = alloc.getType();
  if (!allocType.hasStaticShape())
    return false;

  int64_t numElements = allocType.getNumElements();
  unsigned bitWidth = allocType.getElementTypeBitWidth();
  uint64_t elemBytes = (bitWidth + 7) / 8;
  if (numElements < 0 || elemBytes == 0)
    return false;
  if (static_cast<uint64_t>(numElements) >
      std::numeric_limits<uint64_t>::max() / elemBytes)
    return false;

  uint64_t bytes = static_cast<uint64_t>(numElements) * elemBytes;
  uint64_t offset = static_cast<uint64_t>(signedOffset);
  uint64_t maxSize = std::numeric_limits<size_t>::max();
  if (offset > maxSize || bytes > maxSize - offset)
    return false;

  interval = BlockInfo::IntervalT(static_cast<size_t>(offset),
                                  static_cast<size_t>(offset + bytes));
  return true;
}

static bool isTxDialect(Operation *op) {
  auto *d = op->getDialect();
  return d && d->getNamespace() == "tx";
}

bool isPureAddressOp(Operation *op) {
  if (!op || isTxDialect(op))
    return false;
  // Real memref data movement / visibility — must participate in hazards.
  if (isa<memref::LoadOp, memref::StoreOp, memref::CopyOp>(op))
    return false;

  auto *dialect = op->getDialect();
  if (!dialect)
    return false;
  StringRef ns = dialect->getNamespace();
  // Addressing, layout, and control flow only (no SPM/DDR data dependence).
  if (ns == "arith" || ns == "memref" || ns == "scf" || ns == "cf" ||
      ns == "builtin" || ns == "affine" || ns == "index")
    return true;
  return false;
}

static void collectTx81Accesses(Operation *op, SmallVector<Value> &reads,
                                SmallVector<Value> &writes) {
  auto hasAccess = [&](Value v) {
    for (Value read : reads)
      if (read == v)
        return true;
    for (Value write : writes)
      if (write == v)
        return true;
    return false;
  };

  // Prefer MemoryEffectOpInterface if present.
  if (auto iface = dyn_cast<MemoryEffectOpInterface>(op)) {
    SmallVector<SideEffects::EffectInstance<MemoryEffects::Effect>> effects;
    iface.getEffects(effects);
    for (auto &e : effects) {
      Value v = e.getValue();
      if (!v)
        continue;
      if (isa<MemoryEffects::Read>(e.getEffect()))
        reads.push_back(v);
      else if (isa<MemoryEffects::Write>(e.getEffect()))
        writes.push_back(v);
    }
    // Many Tx81 ops annotate destination operands with MemWrite but leave
    // source address operands unannotated. Treat remaining address-like
    // operands as reads.
    for (Value v : op->getOperands())
      if (!hasAccess(v) && resolveForBufferLookup(v))
        reads.push_back(v);
    return;
  }

  for (Value v : op->getOperands())
    reads.push_back(v);
}

static void collectCpuAccesses(Operation *op, SmallVector<Value> &reads,
                               SmallVector<Value> &writes) {
  if (auto load = dyn_cast<memref::LoadOp>(op)) {
    reads.push_back(load.getMemRef());
    return;
  }
  if (auto store = dyn_cast<memref::StoreOp>(op)) {
    writes.push_back(store.getMemRef());
    return;
  }
  if (auto copy = dyn_cast<memref::CopyOp>(op)) {
    reads.push_back(copy.getSource());
    writes.push_back(copy.getTarget());
    return;
  }

  // CPU / unknown: operands may be real reads of shared buffers.
  for (Value v : op->getOperands())
    reads.push_back(v);
}

void MembarAnalysis::run(FuncBlockInfoMapT &funcBlockInfoMap) {
  FunctionOpInterface funcOp =
      dyn_cast<FunctionOpInterface>(allocation->getOperation());
  OpBuilder builder(funcOp.getContext());
  resolve(funcOp, &funcBlockInfoMap, &builder);
}

void MembarAnalysis::resolve(FunctionOpInterface funcOp,
                             FuncBlockInfoMapT *funcBlockInfoMap,
                             OpBuilder *builder) {
  DenseMap<VirtualBlock, BlockInfo> inputBlockInfoMap;
  DenseMap<VirtualBlock, BlockInfo> outputBlockInfoMap;
  std::deque<VirtualBlock> blockList;

  funcOp.walk<WalkOrder::PreOrder>([&](Block *block) {
    if (block->isEntryBlock() &&
        !isa<RegionBranchOpInterface>(block->getParentOp()))
      blockList.emplace_back(block, Block::iterator());
  });

  while (!blockList.empty()) {
    VirtualBlock block = blockList.front();
    blockList.pop_front();
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

    if (outputBlockInfoMap.count(block) &&
        inputBlockInfo == outputBlockInfoMap[block])
      continue;

    outputBlockInfoMap[block] = inputBlockInfo;
    for (VirtualBlock successor : successors) {
      inputBlockInfoMap[successor].join(outputBlockInfoMap[block]);
      blockList.emplace_back(successor);
    }
  }

  // Join dangling buffers at return sites.
  BlockInfo &funcBlockInfo = (*funcBlockInfoMap)[funcOp];
  funcOp.walk<WalkOrder::PreOrder>([&](Operation *retLike) {
    if (!retLike->hasTrait<OpTrait::ReturnLike>())
      return;
    SmallVector<std::pair<VirtualBlock, BlockInfo>> virtualBlocks;
    for (auto &[vb, blockInfo] : outputBlockInfoMap)
      if (vb.first == retLike->getBlock())
        virtualBlocks.emplace_back(vb, blockInfo);
    if (virtualBlocks.empty())
      return;
    auto maxIt = llvm::max_element(virtualBlocks, [&](auto &lhs, auto &rhs) {
      Block::iterator lhsIt = lhs.first.second, rhsIt = rhs.first.second;
      return !lhsIt.isValid() ||
             (rhsIt.isValid() && lhsIt->isBeforeInBlock(&*rhsIt));
    });
    funcBlockInfo.join(maxIt->second);
  });
}

void MembarAnalysis::visitTerminator(Operation *op,
                                     SmallVector<VirtualBlock> &successors) {
  if (isa<BranchOpInterface>(op)) {
    for (Block *successor : op->getSuccessors())
      successors.emplace_back(successor, Block::iterator());
    return;
  }

  if (auto br = dyn_cast<RegionBranchOpInterface>(op)) {
    SmallVector<RegionSuccessor> regions;
    br.getSuccessorRegions(RegionBranchPoint::parent(), regions);
    for (RegionSuccessor &region : regions) {
      if (region.isParent())
        successors.emplace_back(br->getBlock(), br->getIterator());
      else
        successors.emplace_back(&region.getSuccessor()->front(),
                                Block::iterator());
    }
    return;
  }

  auto br = dyn_cast<RegionBranchTerminatorOpInterface>(op);
  if (br && isa<RegionBranchOpInterface>(br->getParentOp())) {
    SmallVector<Attribute> operands(br->getNumOperands());
    SmallVector<RegionSuccessor> regions;
    br.getSuccessorRegions(operands, regions);
    for (RegionSuccessor &region : regions) {
      if (region.isParent()) {
        Operation *parent = br->getParentOp();
        successors.emplace_back(parent->getBlock(), parent->getIterator());
      } else {
        successors.emplace_back(&region.getSuccessor()->front(),
                                Block::iterator());
      }
    }
    return;
  }

  if (op->hasTrait<OpTrait::ReturnLike>())
    return;
  llvm_unreachable("Unknown terminator encountered in Tx81 membar analysis");
}

void MembarAnalysis::insertBarrier(Operation *op, OpBuilder *builder) {
  OpBuilder::InsertionGuard g(*builder);
  builder->create<tx::BarrierOp>(op->getLoc());
}

void MembarAnalysis::update(Operation *op, BlockInfo *blockInfo,
                            FuncBlockInfoMapT *funcBlockInfoMap,
                            OpBuilder *builder) {
  if (isa<tx::BarrierOp>(op)) {
    blockInfo->sync();
    return;
  }

  BlockInfo curBlockInfo;

  // Inter-procedural: treat calls as their callee's block info.
  if (isa<triton::CallOp>(op)) {
    auto callOpInterface = dyn_cast<CallOpInterface>(op);
    if (auto callee =
            dyn_cast<FunctionOpInterface>(callOpInterface.resolveCallable()))
      curBlockInfo = funcBlockInfoMap->lookup(callee);
  } else {
    SmallVector<Value> reads, writes;

    if (isTxDialect(op)) {
      collectTx81Accesses(op, reads, writes);
    } else if (isPureAddressOp(op)) {
      // memref/arith/scf scaffolding between tx ops — not a CPU data touch.
    } else {
      collectCpuAccesses(op, reads, writes);
    }

    auto addIntervals = [&](ArrayRef<Value> vals, bool isWrite) {
      for (Value v : vals) {
        Value lookup = resolveForBufferLookup(v);
        if (lookup) {
          for (auto bufferId : allocation->getBufferIds(lookup)) {
            if (bufferId == triton::alloc::Allocation::InvalidBufferId)
              continue;
            auto interval = allocation->getAllocatedInterval(bufferId);
            if (isWrite)
              curBlockInfo.syncWriteIntervals[interval].insert(op);
            else
              curBlockInfo.syncReadIntervals[interval].insert(op);
          }
        }

        BlockInfo::IntervalT physicalInterval;
        if (getAllocationOffsetInterval(v, physicalInterval)) {
          if (isWrite)
            curBlockInfo.syncWriteIntervals[physicalInterval].insert(op);
          else
            curBlockInfo.syncReadIntervals[physicalInterval].insert(op);
        }
      }
    };

    addIntervals(reads, /*isWrite=*/false);
    addIntervals(writes, /*isWrite=*/true);
  }

  if (blockInfo->isIntersected(curBlockInfo, filter)) {
    builder->setInsertionPoint(op);
    insertBarrier(op, builder);
    blockInfo->sync();
  }
  blockInfo->join(curBlockInfo);
}

} // namespace mlir::triton::membar
