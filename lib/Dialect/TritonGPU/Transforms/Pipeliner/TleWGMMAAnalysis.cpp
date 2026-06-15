#ifdef __TLE__
#include "TleWGMMAAnalysis.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "nvidia/include/Dialect/NVWS/IR/Dialect.h"
#include "tle/dialect/include/IR/Dialect.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/Triton/IR/Utility.h"
#include "triton/Dialect/TritonGPU/IR/Attributes.h"
#include "triton/Dialect/TritonGPU/Transforms/Utility.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/STLExtras.h"

using namespace mlir;
namespace tt = mlir::triton;
namespace ttg = mlir::triton::gpu;
namespace ttng = mlir::triton::nvidia_gpu;
namespace ttnvws = mlir::triton::nvws;
namespace tttle = mlir::triton::tle;

namespace mlir::triton::gpu::detail {

bool TlePipeResourceAnalysis::isLifetimeBoundary(Operation *op) const {
  if (isa<tttle::PipeReaderReleaseOp, ttnvws::ConsumerReleaseOp>(op))
    return true;

  // A plain arrive is a ready/full signal: it says that data is available to
  // readers, not that the storage may be reused by writers. Only arrives that
  // explicitly release fields end a buffer lifetime and can require a WGMMA
  // operand wait.
  auto arrive = dyn_cast<ttng::ArriveBarrierOp>(op);
  return arrive && !arrive.getReleasedFields().empty();
}

static unsigned ceilDiv(unsigned lhs, unsigned rhs) {
  return (lhs + rhs - 1) / rhs;
}

static bool hasSingleWgmmaAccumulatorTile(ttng::WarpGroupDotOp dotOp) {
  auto resultType = dyn_cast<RankedTensorType>(dotOp.getType());
  if (!resultType)
    return false;
  auto mmaEncoding =
      dyn_cast<ttg::NvidiaMmaEncodingAttr>(resultType.getEncoding());
  if (!mmaEncoding)
    return false;

  auto shapePerCTA = ttg::getShapePerCTA(resultType);
  auto instrShape = mmaEncoding.getInstrShape();
  auto warpsPerCTA = mmaEncoding.getWarpsPerCTA();
  unsigned tileM = instrShape[0] * warpsPerCTA[0];
  unsigned tileN = instrShape[1] * warpsPerCTA[1];
  return ceilDiv(shapePerCTA[0], tileM) == 1 &&
         ceilDiv(shapePerCTA[1], tileN) == 1;
}

static Value skipNoopDefs(Value value) {
  while (Operation *def = value.getDefiningOp()) {
    if (!isNoop(def))
      break;
    assert(def->getNumOperands() == 1 && def->getNumResults() == 1 &&
           "Expected no-op accumulator def to be single-input single-output");
    value = def->getOperand(0);
  }
  return value;
}

static ttng::WarpGroupDotOp
getAccumulatorChainSourceDot(ttng::WarpGroupDotOp dotOp) {
  return skipNoopDefs(dotOp.getC()).getDefiningOp<ttng::WarpGroupDotOp>();
}

static bool hasOnlyDirectAccumulatorChainUse(Value value,
                                             ttng::WarpGroupDotOp targetDot,
                                             scf::ForOp forOp) {
  SmallVector<Value> queue = {value};
  llvm::SmallDenseSet<Value, 8> visited;
  bool foundTargetUse = false;

  while (!queue.empty()) {
    Value current = queue.pop_back_val();
    if (!visited.insert(current).second)
      continue;

    for (OpOperand &use : current.getUses()) {
      Operation *user = use.getOwner();
      if (user->getParentOp() != forOp)
        return false;

      if (isNoop(user)) {
        if (user->getNumResults() != 1)
          return false;
        queue.push_back(user->getResult(0));
        continue;
      }

      auto userDot = dyn_cast<ttng::WarpGroupDotOp>(user);
      if (userDot.getOperation() != targetDot.getOperation() ||
          use.getOperandNumber() != 2)
        return false;
      foundTargetUse = true;
    }
  }

  return foundTargetUse;
}

static ttng::WarpGroupDotOp
getOnlyDirectAccumulatorChainTarget(Value value, scf::ForOp forOp) {
  SmallVector<Value> queue = {value};
  llvm::SmallDenseSet<Value, 8> visited;
  ttng::WarpGroupDotOp targetDot;

  while (!queue.empty()) {
    Value current = queue.pop_back_val();
    if (!visited.insert(current).second)
      continue;

    for (OpOperand &use : current.getUses()) {
      Operation *user = use.getOwner();
      if (user->getParentOp() != forOp)
        return {};

      if (isNoop(user)) {
        if (user->getNumResults() != 1)
          return {};
        queue.push_back(user->getResult(0));
        continue;
      }

      auto userDot = dyn_cast<ttng::WarpGroupDotOp>(user);
      if (!userDot || use.getOperandNumber() != 2)
        return {};
      if (targetDot && targetDot.getOperation() != userDot.getOperation())
        return {};
      targetDot = userDot;
    }
  }

  return targetDot;
}

static bool isWgmmaCommitGroupBoundary(Operation *op) {
  // Accumulator values may flow through multiple WGMMA commit groups, but a
  // single commit group must remain a contiguous WGMMA launch region. In
  // particular, a barrier wait may sit between two dependent WGMMA batches
  // (FlashMLA waits for K0R between QK-left and QK-right), but the first batch
  // must be committed before that wait.
  //
  // Keep both TLE-level and lowered NVIDIA-level synchronization ops here. This
  // analysis runs before all TLE ops are lowered, so missing a TLE fence can
  // leave a later fence.proxy inside an uncommitted WGMMA group. ptxas then has
  // to serialize the group with per-HGMMA depbar/gsb0 even though the IR-level
  // dependency only needs a commit boundary, not an accumulator wait.
  return isa<ttng::WaitBarrierOp, ttng::ArriveBarrierOp,
             ttng::FenceAsyncSharedOp, ttng::WarpGroupDotCommitOp,
             ttng::WarpGroupDotWaitOp, ttng::TMAStoreWaitOp,
             ttg::LocalBarrierOp, ttnvws::ProducerAcquireOp,
             ttnvws::ProducerCommitOp, ttnvws::ConsumerWaitOp,
             ttnvws::ConsumerReleaseOp, tttle::WGMMASharedOperandFenceOp,
             tttle::TMAStoreCommitGroupOp, tttle::DistributedBarrierOp>(op);
}

static bool isWgmmaAccumulatorChainBoundary(Operation *op) {
  // ptxas treats a WGMMA accumulator pipeline stage as open until a
  // wgmma.wait_group. Chaining the same accumulator through a later WGMMA is
  // valid only while the compiler can keep the region as a WGMMA-only launch
  // sequence plus local descriptor prep. Synchronization/fence ops split that
  // region: if a later WGMMA needs the previous accumulator after such a
  // boundary, first materialize it with warp_group_dot_wait so any register
  // copies or C-operand setup happen after the hardware wait. Otherwise ptxas
  // injects per-HGMMA depbar/gsb0 because it sees accumulator use across a
  // non-WGMMA boundary.
  return isa<ttng::WaitBarrierOp, ttng::ArriveBarrierOp,
             ttng::FenceAsyncSharedOp, ttng::WarpGroupDotWaitOp,
             ttng::TMAStoreWaitOp, ttg::LocalBarrierOp,
             ttnvws::ProducerAcquireOp, ttnvws::ProducerCommitOp,
             ttnvws::ConsumerWaitOp, ttnvws::ConsumerReleaseOp,
             tttle::WGMMASharedOperandFenceOp, tttle::TMAStoreCommitGroupOp,
             tttle::DistributedBarrierOp>(op);
}

static bool hasWgmmaCommitGroupBoundaryBetween(Operation *from, Operation *to) {
  assert(from && to && "expected valid commit-group endpoints");
  Operation *op = from->getNextNode();
  for (; op && op != to; op = op->getNextNode()) {
    if (isWgmmaCommitGroupBoundary(op))
      return true;
    if (op->hasTrait<OpTrait::IsTerminator>())
      return true;
  }
  return op != to;
}

static bool hasWgmmaAccumulatorChainBoundaryBetween(Operation *from,
                                                    Operation *to) {
  assert(from && to && "expected valid accumulator-chain endpoints");
  Operation *op = from->getNextNode();
  for (; op && op != to; op = op->getNextNode()) {
    if (isWgmmaAccumulatorChainBoundary(op))
      return true;
    if (op->hasTrait<OpTrait::IsTerminator>())
      return true;
  }
  return op != to;
}

bool TleWgmmaScheduleAnalysis::canDeferWaitToLaterDotC(
    ttng::WarpGroupDotOp dotOp) const {
  if (!hasSingleWgmmaAccumulatorTile(dotOp))
    return false;

  SmallVector<Value> queue = {dotOp.getResult()};
  llvm::SmallDenseSet<Value, 8> visited;
  bool hasLaterWarpGroupDotCUse = false;

  while (!queue.empty()) {
    Value value = queue.pop_back_val();
    if (!visited.insert(value).second)
      continue;

    for (OpOperand &use : value.getUses()) {
      Operation *user = use.getOwner();
      if (user->getParentOp() != forOp)
        return false;

      if (isNoop(user)) {
        if (user->getNumResults() != 1)
          return false;
        queue.push_back(user->getResult(0));
        continue;
      }

      auto userDot = dyn_cast<ttng::WarpGroupDotOp>(user);
      if (!userDot || use.getOperandNumber() != 2)
        return false;
      if (userDot->getBlock() != dotOp->getBlock() ||
          !dotOp->isBeforeInBlock(userDot))
        return false;
      if (!hasSingleWgmmaAccumulatorTile(userDot))
        return false;
      if (hasWgmmaAccumulatorChainBoundaryBetween(dotOp.getOperation(),
                                                  userDot.getOperation()))
        return false;
      SmallVector<MemDescResource, 2> reads =
          resources.getDotReadResources(dotOp);
      if (resources.hasAliasingLifetimeBoundaryBetween(
              dotOp.getOperation(), userDot.getOperation(), reads))
        return false;
      hasLaterWarpGroupDotCUse = true;
    }
  }

  return hasLaterWarpGroupDotCUse;
}

bool TleWgmmaScheduleAnalysis::canDeferCommitToLaterDotC(
    ttng::WarpGroupDotOp dotOp) const {
  ttng::WarpGroupDotOp targetDot =
      getOnlyDirectAccumulatorChainTarget(dotOp.getResult(), forOp);
  if (!targetDot)
    return false;
  if (!canReuseAccumulatorChainC(targetDot))
    return false;
  return !hasWgmmaCommitGroupBoundaryBetween(dotOp.getOperation(),
                                             targetDot.getOperation());
}

bool TleWgmmaScheduleAnalysis::canAppendToCurrentWgmmaCommitGroup(
    ttng::WarpGroupDotOp dotOp) const {
  if (!canReuseAccumulatorChainC(dotOp))
    return false;
  ttng::WarpGroupDotOp sourceDot = getAccumulatorChainSourceDot(dotOp);
  return sourceDot && !hasWgmmaCommitGroupBoundaryBetween(
                          sourceDot.getOperation(), dotOp.getOperation());
}

bool TlePipeResourceAnalysis::dotReadsSharedMemory(
    ttng::WarpGroupDotOp dotOp) const {
  return isa<ttg::MemDescType>(dotOp.getA().getType()) ||
         isa<ttg::MemDescType>(dotOp.getB().getType());
}

static Value getWarpSpecializeCapture(Value value) {
  auto blockArg = dyn_cast<BlockArgument>(value);
  if (!blockArg)
    return value;

  Block *block = blockArg.getOwner();
  auto partitions =
      dyn_cast_or_null<ttg::WarpSpecializePartitionsOp>(block->getParentOp());
  if (!partitions)
    return value;

  auto wsOp = dyn_cast<ttg::WarpSpecializeOp>(partitions->getParentOp());
  if (!wsOp)
    return value;

  unsigned argNo = blockArg.getArgNumber();
  OperandRange captures = wsOp.getExplicitCaptures();
  if (argNo >= captures.size())
    return value;
  return captures[argNo];
}

static bool isNonWarpSpecializeRegionMemDescArg(Value value) {
  auto blockArg = dyn_cast<BlockArgument>(value);
  if (!blockArg)
    return false;
  if (!isa<ttg::MemDescType>(blockArg.getType()))
    return false;

  Block *block = blockArg.getOwner();
  auto partitions =
      dyn_cast_or_null<ttg::WarpSpecializePartitionsOp>(block->getParentOp());
  if (partitions)
    return false;
  Operation *parent = block->getParentOp();
  return parent && !isa<tt::FuncOp>(parent);
}

static Value stripMemDescViewsForLocalAlloc(Value value) {
  while (true) {
    Value captured = getWarpSpecializeCapture(value);
    if (captured != value) {
      value = captured;
      continue;
    }
    if (auto index = value.getDefiningOp<ttg::MemDescIndexOp>()) {
      value = index.getSrc();
      continue;
    }
    if (auto subslice = value.getDefiningOp<ttg::MemDescSubsliceOp>()) {
      value = subslice.getSrc();
      continue;
    }
    if (auto trans = value.getDefiningOp<ttg::MemDescTransOp>()) {
      value = trans.getSrc();
      continue;
    }
    if (auto reshape = value.getDefiningOp<ttg::MemDescReshapeOp>()) {
      value = reshape.getSrc();
      continue;
    }
    if (auto reinterpret = value.getDefiningOp<ttg::MemDescReinterpretOp>()) {
      value = reinterpret.getSrc();
      continue;
    }
    if (auto view = value.getDefiningOp<tttle::MemDescWGMMAViewOp>()) {
      value = view.getSrc();
      continue;
    }
    return value;
  }
}

static bool isMemDescBackedByLocalAlloc(Value value,
                                        llvm::DenseSet<Value> &active);

static bool isForIterArgBackedByLocalAlloc(BlockArgument arg,
                                           llvm::DenseSet<Value> &active) {
  auto forOp = dyn_cast<scf::ForOp>(arg.getOwner()->getParentOp());
  if (!forOp)
    return false;

  unsigned argNo = arg.getArgNumber();
  if (argNo == 0)
    return false;
  unsigned iterIdx = argNo - 1;
  if (iterIdx >= forOp.getInitArgs().size())
    return false;

  if (!isMemDescBackedByLocalAlloc(forOp.getInitArgs()[iterIdx], active))
    return false;

  auto yield = dyn_cast<scf::YieldOp>(forOp.getBody()->getTerminator());
  if (!yield || iterIdx >= yield.getNumOperands())
    return false;

  return isMemDescBackedByLocalAlloc(yield.getOperand(iterIdx), active);
}

static bool isForResultBackedByLocalAlloc(scf::ForOp forOp, unsigned resultIdx,
                                          llvm::DenseSet<Value> &active) {
  if (resultIdx >= forOp.getInitArgs().size())
    return false;
  if (!isMemDescBackedByLocalAlloc(forOp.getInitArgs()[resultIdx], active))
    return false;

  auto yield = dyn_cast<scf::YieldOp>(forOp.getBody()->getTerminator());
  if (!yield || resultIdx >= yield.getNumOperands())
    return false;

  return isMemDescBackedByLocalAlloc(yield.getOperand(resultIdx), active);
}

static bool isIfResultBackedByLocalAlloc(scf::IfOp ifOp, unsigned resultIdx,
                                         llvm::DenseSet<Value> &active) {
  if (resultIdx >= ifOp.thenYield().getNumOperands() ||
      resultIdx >= ifOp.elseYield().getNumOperands())
    return false;

  return isMemDescBackedByLocalAlloc(ifOp.thenYield().getOperand(resultIdx),
                                     active) &&
         isMemDescBackedByLocalAlloc(ifOp.elseYield().getOperand(resultIdx),
                                     active);
}

static bool isMemDescBackedByLocalAlloc(Value value,
                                        llvm::DenseSet<Value> &active) {
  value = stripMemDescViewsForLocalAlloc(value);
  if (!active.insert(value).second)
    return true;

  if (value.getDefiningOp<ttg::LocalAllocOp>())
    return true;

  if (auto select = value.getDefiningOp<arith::SelectOp>())
    return isMemDescBackedByLocalAlloc(select.getTrueValue(), active) &&
           isMemDescBackedByLocalAlloc(select.getFalseValue(), active);

  if (auto arg = dyn_cast<BlockArgument>(value))
    return isForIterArgBackedByLocalAlloc(arg, active);

  auto result = dyn_cast<OpResult>(value);
  if (!result)
    return false;

  if (auto forOp = dyn_cast<scf::ForOp>(result.getOwner()))
    return isForResultBackedByLocalAlloc(forOp, result.getResultNumber(),
                                         active);
  if (auto ifOp = dyn_cast<scf::IfOp>(result.getOwner()))
    return isIfResultBackedByLocalAlloc(ifOp, result.getResultNumber(), active);

  return false;
}

static bool isMemDescBackedByLocalAlloc(Value value) {
  llvm::DenseSet<Value> active;
  return isMemDescBackedByLocalAlloc(value, active);
}

static Value getFoldableWgmmaStagingSource(Value value) {
  auto alloc = value.getDefiningOp<ttg::LocalAllocOp>();
  if (!alloc || !alloc.getSrc())
    return Value();

  auto localLoad = alloc.getSrc().getDefiningOp<ttg::LocalLoadOp>();
  if (!localLoad || localLoad.getToken())
    return Value();

  if (!llvm::all_of(alloc->getUsers(), [](Operation *user) {
        return isa<ttng::WarpGroupDotOp, ttng::WarpGroupDotWaitOp>(user);
      }))
    return Value();

  Value src = localLoad.getSrc();
  auto srcTy = dyn_cast<ttg::MemDescType>(src.getType());
  auto allocTy = dyn_cast<ttg::MemDescType>(alloc.getType());
  if (!srcTy || !allocTy)
    return Value();

  if (srcTy.getShape() != allocTy.getShape() ||
      srcTy.getElementType() != allocTy.getElementType() ||
      srcTy.getMemorySpace() != allocTy.getMemorySpace() ||
      srcTy.getEncoding() != allocTy.getEncoding())
    return Value();

  return src;
}

static Value getFoldableWgmmaALocalLoadSource(Value value) {
  auto localLoad = value.getDefiningOp<ttg::LocalLoadOp>();
  if (!localLoad || localLoad.getToken())
    return Value();

  Value src = localLoad.getSrc();
  auto srcTy = dyn_cast<ttg::MemDescType>(src.getType());
  auto localLoadTy = dyn_cast<RankedTensorType>(localLoad.getType());
  if (!srcTy || !localLoadTy)
    return Value();

  if (srcTy.getShape() != localLoadTy.getShape() ||
      srcTy.getElementType() != localLoadTy.getElementType())
    return Value();

  if (!isa<ttg::NVMMASharedEncodingAttr, ttg::SharedLinearEncodingAttr>(
          srcTy.getEncoding()))
    return Value();

  if (!isMemDescBackedByLocalAlloc(src))
    return Value();

  return src;
}

static MemDescResource getMemDescResource(Value value) {
  MemDescResource resource;
  SmallVector<Value, 2> reversedIndices;

  while (true) {
    Value captured = getWarpSpecializeCapture(value);
    if (captured != value) {
      value = captured;
      continue;
    }

    if (isNonWarpSpecializeRegionMemDescArg(value)) {
      resource.unknown = true;
      return resource;
    }

    if (Value stagingSource = getFoldableWgmmaStagingSource(value)) {
      value = stagingSource;
      continue;
    }
    if (auto index = value.getDefiningOp<ttg::MemDescIndexOp>()) {
      reversedIndices.push_back(index.getIndex());
      value = index.getSrc();
      continue;
    }
    if (auto subslice = value.getDefiningOp<ttg::MemDescSubsliceOp>()) {
      value = subslice.getSrc();
      continue;
    }
    if (auto trans = value.getDefiningOp<ttg::MemDescTransOp>()) {
      value = trans.getSrc();
      continue;
    }
    if (auto reshape = value.getDefiningOp<ttg::MemDescReshapeOp>()) {
      value = reshape.getSrc();
      continue;
    }
    if (auto reinterpret = value.getDefiningOp<ttg::MemDescReinterpretOp>()) {
      value = reinterpret.getSrc();
      continue;
    }
    if (auto view = value.getDefiningOp<tttle::MemDescWGMMAViewOp>()) {
      value = view.getSrc();
      continue;
    }

    resource.root = value;
    resource.indices.assign(reversedIndices.rbegin(), reversedIndices.rend());
    return resource;
  }
}

static std::optional<int64_t> getConstantIndex(Value value) {
  if (auto constant = value.getDefiningOp<arith::ConstantIntOp>())
    return constant.value();
  return std::nullopt;
}

static int64_t positiveMod(int64_t value, int64_t modulo) {
  int64_t result = value % modulo;
  return result < 0 ? result + modulo : result;
}

struct ModuloIndex {
  Value base;
  int64_t modulo;
  int64_t offset;
};

static Value peelConstantOffset(Value value, int64_t &offset) {
  while (Operation *def = value.getDefiningOp()) {
    if (auto add = dyn_cast<arith::AddIOp>(def)) {
      if (auto rhs = getConstantIndex(add.getRhs())) {
        offset += *rhs;
        value = add.getLhs();
        continue;
      }
      if (auto lhs = getConstantIndex(add.getLhs())) {
        offset += *lhs;
        value = add.getRhs();
        continue;
      }
    }

    if (auto sub = dyn_cast<arith::SubIOp>(def)) {
      if (auto rhs = getConstantIndex(sub.getRhs())) {
        offset -= *rhs;
        value = sub.getLhs();
        continue;
      }
    }

    break;
  }

  return value;
}

static std::optional<ModuloIndex> getModuloIndex(Value value) {
  auto rem = value.getDefiningOp<arith::RemSIOp>();
  if (!rem)
    return std::nullopt;

  auto modulo = getConstantIndex(rem.getRhs());
  if (!modulo || *modulo <= 0)
    return std::nullopt;

  int64_t offset = 0;
  Value base = peelConstantOffset(rem.getLhs(), offset);
  return ModuloIndex{base, *modulo, positiveMod(offset, *modulo)};
}

static bool indicesProvenDifferent(Value lhs, Value rhs) {
  std::optional<int64_t> lhsConst = getConstantIndex(lhs);
  std::optional<int64_t> rhsConst = getConstantIndex(rhs);
  if (lhsConst && rhsConst)
    return *lhsConst != *rhsConst;

  std::optional<ModuloIndex> lhsModulo = getModuloIndex(lhs);
  std::optional<ModuloIndex> rhsModulo = getModuloIndex(rhs);
  if (lhsModulo && rhsModulo && lhsModulo->base == rhsModulo->base &&
      lhsModulo->modulo == rhsModulo->modulo)
    return lhsModulo->offset != rhsModulo->offset;

  return false;
}

static bool resourcesMayAlias(const MemDescResource &lhs,
                              const MemDescResource &rhs) {
  if (lhs.unknown || rhs.unknown)
    return true;
  if (lhs.root != rhs.root)
    return false;

  unsigned commonDepth = std::min(lhs.indices.size(), rhs.indices.size());
  for (unsigned i = 0; i < commonDepth; ++i) {
    if (lhs.indices[i] == rhs.indices[i])
      continue;
    if (indicesProvenDifferent(lhs.indices[i], rhs.indices[i]))
      return false;
    return true;
  }

  return true;
}

SmallVector<MemDescResource, 2>
TlePipeResourceAnalysis::getDotReadResources(ttng::WarpGroupDotOp dotOp) const {
  SmallVector<MemDescResource, 2> reads;
  if (isa<ttg::MemDescType>(dotOp.getA().getType()))
    reads.push_back(getMemDescResource(dotOp.getA()));
  else if (Value src = getFoldableWgmmaALocalLoadSource(dotOp.getA()))
    reads.push_back(getMemDescResource(src));
  if (isa<ttg::MemDescType>(dotOp.getB().getType()))
    reads.push_back(getMemDescResource(dotOp.getB()));
  return reads;
}

static SmallVector<MemDescResource, 2>
getPipeReaderReleaseResources(tttle::PipeReaderReleaseOp release) {
  SmallVector<MemDescResource, 2> resources;
  for (Value field : release.getFields()) {
    MemDescResource resource = getMemDescResource(field);
    if (!resource.unknown)
      resource.indices.push_back(release.getStage());
    resources.push_back(std::move(resource));
  }
  if (resources.empty())
    resources.push_back(MemDescResource{/*root=*/Value(), /*indices=*/{},
                                        /*unknown=*/true});
  return resources;
}

static SmallVector<MemDescResource, 2>
getConsumerReleaseResources(ttnvws::ConsumerReleaseOp release) {
  SmallVector<MemDescResource, 2> resources;
  for (Value field : release.getReleasedFields()) {
    MemDescResource resource = getMemDescResource(field);
    if (!resource.unknown)
      resource.indices.push_back(release.getIdx());
    resources.push_back(std::move(resource));
  }
  if (resources.empty())
    resources.push_back(MemDescResource{/*root=*/Value(), /*indices=*/{},
                                        /*unknown=*/true});
  return resources;
}

static SmallVector<MemDescResource, 2>
getArriveBarrierReleasedResources(ttng::ArriveBarrierOp arrive) {
  SmallVector<MemDescResource, 2> resources;
  for (Value field : arrive.getReleasedFields()) {
    MemDescResource resource = getMemDescResource(field);
    if (!resource.unknown && arrive.getReleasedIdx())
      resource.indices.push_back(arrive.getReleasedIdx());
    resources.push_back(std::move(resource));
  }
  return resources;
}

SmallVector<MemDescResource, 2>
TlePipeResourceAnalysis::getBoundaryReleasedResources(Operation *op) const {
  if (auto release = dyn_cast<tttle::PipeReaderReleaseOp>(op))
    return getPipeReaderReleaseResources(release);
  if (auto release = dyn_cast<ttnvws::ConsumerReleaseOp>(op))
    return getConsumerReleaseResources(release);
  if (auto arrive = dyn_cast<ttng::ArriveBarrierOp>(op))
    return getArriveBarrierReleasedResources(arrive);
  return {};
}

bool TlePipeResourceAnalysis::releasedResourcesMayAliasReads(
    ArrayRef<MemDescResource> releasedResources,
    ArrayRef<MemDescResource> readResources) const {
  for (const MemDescResource &released : releasedResources) {
    for (const MemDescResource &read : readResources) {
      if (resourcesMayAlias(released, read))
        return true;
    }
  }
  return false;
}

bool TlePipeResourceAnalysis::boundaryMayAliasReads(
    Operation *op, ArrayRef<MemDescResource> readResources) const {
  return releasedResourcesMayAliasReads(getBoundaryReleasedResources(op),
                                        readResources);
}

bool TlePipeResourceAnalysis::hasAliasingLifetimeBoundaryBetween(
    Operation *from, Operation *to,
    ArrayRef<MemDescResource> readResources) const {
  assert(from && to && "expected valid boundary endpoints");
  Operation *op = from->getNextNode();
  for (; op && op != to; op = op->getNextNode()) {
    if (isLifetimeBoundary(op) && boundaryMayAliasReads(op, readResources))
      return true;
    if (op->hasTrait<OpTrait::IsTerminator>())
      return true;
  }
  return op != to;
}

bool TleWgmmaScheduleAnalysis::canReuseAccumulatorChainC(
    ttng::WarpGroupDotOp dotOp) const {
  ttng::WarpGroupDotOp sourceDot = getAccumulatorChainSourceDot(dotOp);
  if (!sourceDot || sourceDot->getBlock() != dotOp->getBlock())
    return false;
  if (!sourceDot->isBeforeInBlock(dotOp))
    return false;
  if (!hasSingleWgmmaAccumulatorTile(sourceDot) ||
      !hasSingleWgmmaAccumulatorTile(dotOp))
    return false;
  if (hasWgmmaAccumulatorChainBoundaryBetween(sourceDot.getOperation(),
                                              dotOp.getOperation()))
    return false;

  SmallVector<MemDescResource, 2> reads =
      resources.getDotReadResources(sourceDot);
  if (resources.hasAliasingLifetimeBoundaryBetween(sourceDot.getOperation(),
                                                   dotOp.getOperation(), reads))
    return false;
  if (!hasOnlyDirectAccumulatorChainUse(sourceDot.getResult(), dotOp, forOp))
    return false;
  return true;
}

} // namespace mlir::triton::gpu::detail

#endif // __TLE__
