//===----------------------------------------------------------------------===//
// TODO: Pass Description
//===----------------------------------------------------------------------===//

#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/TritonXPU/IR/Dialect.h"
#include "triton/Dialect/TritonXPU/Transforms/Passes.h"

#include "mlir/Analysis/TopologicalSortUtils.h"
#include "mlir/IR/IRMapping.h"

namespace mlir {
namespace triton {
namespace xpu {

#define GEN_PASS_DEF_TRITONXPUMASK
#include "triton/Dialect/TritonXPU/Transforms/Passes.h.inc"

/// Whether `bodyOp` is a reduce that yields values only, with no companion
/// index result.
///
/// Why: widening lets a fully-masked core contribute `other` instead of the
/// reduce identity, which is sound only because `other` is chosen neutral for
/// the combine (0 for a sum, -inf for a max). An index-carrying reduce --
/// argmax / argmin, two results -- runs a second lane through the same combine,
/// and an index has no neutral element: on a tie the index of the dead core
/// wins the comparison and is written out.
///
/// How: count results, not operands -- xpu::ReduceOp carries the tile loop
/// induction variable as a trailing operand, so even a single-value reduce
/// reports two operands.
static bool isValueOnlyReduce(Operation *bodyOp) {
  auto reduceOp = dyn_cast<triton::xpu::ReduceOp>(bodyOp);
  return reduceOp && reduceOp.getNumResults() == 1;
}

/// Whether `reduceOp` combines with addition.
///
/// Why: widening runs a fully-masked core through the reduce, where it adds the
/// `other` this pass selects in. That is sound only when `other` is the combine
/// identity -- true for add (0), false for max / min / prod (-inf / +inf / 1),
/// where the 0 a dead core adds would override an all-negative max.
static bool isAdditiveCombine(triton::xpu::ReduceOp reduceOp) {
  Region &combine = reduceOp.getCombineOp();
  if (combine.empty())
    return false;
  Operation *terminator = combine.front().getTerminator();
  if (terminator->getNumOperands() != 1)
    return false;
  Operation *def = terminator->getOperand(0).getDefiningOp();
  return isa_and_nonnull<arith::AddFOp, arith::AddIOp>(def);
}

/// Whether every value-only reduce in `bodyOps` combines additively.
///
/// Why: the widened guard runs the whole region, so one non-additive reduce is
/// enough to make a dead core contribution wrong.
static bool hasOnlyAdditiveValueReduces(ArrayRef<Operation *> bodyOps) {
  for (Operation *bodyOp : bodyOps) {
    if (!isValueOnlyReduce(bodyOp))
      continue;
    if (!isAdditiveCombine(cast<triton::xpu::ReduceOp>(bodyOp)))
      return false;
  }
  return true;
}

/// Whether the per-core tail guard around `bodyOps` may be widened to run on
/// all cores.
///
/// Why: a value reduce needs every core to execute, so the cores whose lanes
/// are all masked out still contribute `other` (the select this pass inserts)
/// instead of the reduce identity -- otherwise a masked `tl.load(..., other=C)`
/// feeding tl.sum silently drops the masked lanes
/// (test_masked_reduce_other_dropped). Widening is only safe when a
/// fully-masked core can run without touching memory, which the gm2lm length
/// clamp `readLen = smin(smax(len, 0), bufLen)` arranges -- but that clamp
/// reads one len per core along the last dim, so it only expresses a mask that
/// varies along the last dim. Rank 1 is the shape where the mask dim and the
/// len dim coincide; a rank-2 mask would let a trailing core read out of bounds
/// (sglang eagle_verify status 700).
///
/// How:
///   - decline unless the region holds a value-only reduce, so a plain masked
///     load/store keeps its per-core guard and an argmax/argmin keeps it too
///     (see isValueOnlyReduce);
///   - decline unless every such reduce combines additively, so the `other` a
///     dead core contributes is the combine identity (see isAdditiveCombine);
///   - decline unless the mask is a rank-1 tensor.
///
/// NOTE: this is layered on top of upstream's KTC-8071 tail-guard decision --
/// it only changes the guard condition for the value-only rank-1 reduce case;
/// every other case keeps upstream's extract(mask, 0) / bail-out behaviour.
static bool canWidenTailGuardToAllCores(ArrayRef<Operation *> bodyOps,
                                        Value mask) {
  if (!llvm::any_of(bodyOps, isValueOnlyReduce))
    return false;
  // An index-carrying reduce (argmax / argmin: two results) has no neutral
  // index. Widening runs a fully-masked core through the same combine, and on
  // a tie its index wins -- even when additive value reduces share the body.
  // Decline the mixed body so the index reduce keeps its per-core guard.
  if (llvm::any_of(bodyOps, [](Operation *op) {
        auto r = dyn_cast<triton::xpu::ReduceOp>(op);
        return r && r.getNumResults() != 1;
      }))
    return false;
  if (!hasOnlyAdditiveValueReduces(bodyOps))
    return false;
  auto maskTy = dyn_cast<RankedTensorType>(mask.getType());
  return maskTy && maskTy.getRank() == 1;
}

namespace {
// Rewriting `lhs < rhs` into the suffix length `rhs - lhs` is only sound when
// `lhs` advances by exactly one per lane (i.e. the predicate is "the first
// rhs-lhs lanes are on"). For anything else the length is unrelated to the
// predicate and the mask silently stops being applied.
//
// Only reject the cases we can positively prove non-affine (a lane-dependent
// rem/div/bit-op on the index), so shapes that work today keep their block
// DMA; everything unproven keeps the historical behaviour.
//
// Rank and layout changes are looked through. `expand_dims(remsi(r, N)) < lim`
// is the same predicate as `remsi(r, N) < lim`, only in 2D; not seeing through
// the rank change would leave the folder turning it into a length again and
// drop the mask exactly the way this pass exists to prevent.
//
// A modulo whose period is a compile-time multiple of the per-core tile is
// exempted by `coreAlignedPeriodKeepsFoldSound`: the wrap boundary cannot fall
// inside a core there, so the fold is still exact and the block DMA survives.
static bool isRankOrLayoutOnly(Operation *op) {
  return isa<triton::ExpandDimsOp, triton::SplatOp, triton::BroadcastOp,
             triton::xpu::BroadcastOp, triton::xpu::ConvertLayoutOp>(op);
}

// Memoised so a subexpression shared by several paths is walked once; without
// it the recursion is exponential in the depth of the common subexpression.
static bool isProvenNonAffineImpl(Value v, llvm::DenseMap<Value, bool> &memo) {
  auto cached = memo.find(v);
  if (cached != memo.end())
    return cached->second;

  bool result = false;
  if (Operation *defOp = v.getDefiningOp()) {
    if (isa<mlir::arith::RemSIOp, mlir::arith::RemUIOp, mlir::arith::DivSIOp,
            mlir::arith::DivUIOp, mlir::arith::AndIOp, mlir::arith::OrIOp,
            mlir::arith::XOrIOp, mlir::arith::ShLIOp, mlir::arith::ShRUIOp,
            mlir::arith::ShRSIOp>(defOp)) {
      result = true;
    } else if (isRankOrLayoutOnly(defOp)) {
      result = isProvenNonAffineImpl(defOp->getOperand(0), memo);
    } else if (isa<mlir::arith::AddIOp, mlir::arith::MulIOp,
                   mlir::arith::SubIOp, mlir::arith::ExtSIOp,
                   mlir::arith::ExtUIOp, mlir::arith::TruncIOp,
                   mlir::arith::IndexCastOp>(defOp)) {
      result = llvm::any_of(defOp->getOperands(), [&](Value operand) {
        return isProvenNonAffineImpl(operand, memo);
      });
    }
  }
  memo[v] = result;
  return result;
}

bool isProvenNonAffine(Value v) {
  llvm::DenseMap<Value, bool> memo;
  return isProvenNonAffineImpl(v, memo);
}

// The integer constant behind `v`, looking through the `tt.splat` a scalar
// constant (or a kernel argument folded to one) gets wrapped in.
static bool constantIntSplat(Value v, int64_t &out) {
  if (auto splat = v.getDefiningOp<triton::SplatOp>())
    v = splat.getSrc();
  auto constant = v.getDefiningOp<mlir::arith::ConstantOp>();
  if (!constant)
    return false;
  Attribute value = constant.getValue();
  if (auto intAttr = dyn_cast<IntegerAttr>(value)) {
    out = intAttr.getInt();
    return true;
  }
  if (auto splatAttr = dyn_cast<SplatElementsAttr>(value)) {
    if (auto intAttr =
            dyn_cast<IntegerAttr>(splatAttr.getSplatValue<Attribute>())) {
      out = intAttr.getInt();
      return true;
    }
  }
  return false;
}

// Whether `v` advances by exactly one per lane inside a core, i.e. whether
// `v[lane] == v[0] + lane` for the lanes a core owns. Only the forms Triton
// emits around `tl.arange` are seen through; anything else answers "cannot
// tell", which keeps the fold off just like the affine analysis does.
//
// A constant offset is only looked through when it is a multiple of the
// per-core tile. The wrap boundary of `(lane + C) % period` sits at
// `lane = -C (mod sizePerCore)`, so an unaligned `C` puts one wrap inside
// *every* core and the kept lanes stop being a prefix; `rhs - lhs` is then no
// longer their count. For the same reason `C - lane` is rejected outright --
// it counts down, not up.
bool isLaneStrideOne(Value v, int64_t laneAlignment) {
  Operation *defOp = v.getDefiningOp();
  if (!defOp)
    return false;
  if (auto range = dyn_cast<triton::MakeRangeOp>(defOp))
    return range.getStartAttr().getInt() % laneAlignment == 0;
  if (auto range = dyn_cast<triton::xpu::MakeRangeOp>(defOp))
    return range.getStartAttr().getInt() % laneAlignment == 0;
  if (isa<mlir::arith::ExtSIOp, mlir::arith::ExtUIOp, mlir::arith::TruncIOp,
          mlir::arith::IndexCastOp>(defOp))
    return isLaneStrideOne(defOp->getOperand(0), laneAlignment);
  if (auto subOp = dyn_cast<mlir::arith::SubIOp>(defOp)) {
    int64_t offset;
    if (constantIntSplat(subOp.getRhs(), offset))
      return offset % laneAlignment == 0 &&
             isLaneStrideOne(subOp.getLhs(), laneAlignment);
    return false;
  }
  if (auto addOp = dyn_cast<mlir::arith::AddIOp>(defOp)) {
    int64_t offset;
    if (constantIntSplat(addOp.getRhs(), offset))
      return offset % laneAlignment == 0 &&
             isLaneStrideOne(addOp.getLhs(), laneAlignment);
    if (constantIntSplat(addOp.getLhs(), offset))
      return offset % laneAlignment == 0 &&
             isLaneStrideOne(addOp.getRhs(), laneAlignment);
    return false;
  }
  return false;
}

// `offs % period` is flagged non-affine by `isProvenNonAffine`, but the fold is
// still sound for it whenever the modulo boundary never falls inside a core.
// The tile is strip-mined, so a core owns the lanes `base .. base +
// sizePerCore - 1` with `base = (core_id + loopIdx * core_num) * sizePerCore`;
// a `period` that is a multiple of `sizePerCore` therefore keeps every lane of
// a core on the same run, `lhs` really does advance by one within the run, and
// `rhs - lhs` is exactly the number of kept lanes -- which lets the block DMA
// survive instead of degrading to one DMA per lane.
//
// Only the shape `cmpi(remsi(idx, period), lim)` is recognised. A modulo hidden
// behind `expand_dims`/`broadcast` is still rejected, which costs the block DMA
// but keeps the mask; the rank-changed forms have no length operand that
// describes them per row anyway.
bool coreAlignedPeriodKeepsFoldSound(arith::CmpIOp cmpiOp) {
  auto tensorTy = dyn_cast<RankedTensorType>(cmpiOp.getLhs().getType());
  if (!tensorTy)
    return false;
  auto layout =
      dyn_cast_or_null<triton::xpu::ClusterLayoutAttr>(tensorTy.getEncoding());
  if (!layout)
    return false;
  ArrayRef<unsigned> sizePerCore = layout.getSizePerCore();
  if (sizePerCore.empty() || sizePerCore.back() == 0)
    return false;

  Value lhs = cmpiOp.getLhs();
  Operation *defOp = lhs.getDefiningOp();
  if (!isa_and_nonnull<mlir::arith::RemSIOp, mlir::arith::RemUIOp>(defOp))
    return false;
  int64_t period = 0;
  if (!constantIntSplat(defOp->getOperand(1), period) || period <= 0)
    return false;
  if (period % sizePerCore.back() != 0)
    return false;
  // `(2 * i) % period` wraps on the same boundaries but no longer advances by
  // one per lane, so the dividend has to be the lane index itself.
  return isLaneStrideOne(defOp->getOperand(0), sizePerCore.back());
}

// Whether `rhs - lhs` may be used as the DMA length at all.
bool lengthFoldIsSound(arith::CmpIOp cmpiOp) {
  // Every case the affine analysis does not flag keeps the historical fold.
  if (!isProvenNonAffine(cmpiOp.getLhs()))
    return true;
  return coreAlignedPeriodKeepsFoldSound(cmpiOp);
}

// The tailing `scf.if` skips the whole tile when lane 0 sits outside the mask.
// That is only *wrong* when the masked-off lanes still carry a value that has
// to reach the global memory: with an explicit `other` fill the load defines
// one, and a tailing store that is not predicated itself then has to write it
// (a fused constant_pad otherwise keeps its uninitialised LM). Without `other`
// the masked-off lanes hold nothing, and skipping the tile is the historical
// -- and load bearing -- behaviour: it is what keeps an out-of-range wave from
// running the rest of the body, e.g. `mask = pid < bs` on a per-core tile,
// where the tail is `tl.store(out + pid, v)` with no mask of its own. Dropping
// the guard there makes every wave past `bs` execute that store and write out
// of bounds.
bool tailIfIsSafe(Operation *cmpiOp, Operation *terminator) {
  for (Operation *cur = cmpiOp->getNextNode(); cur && cur != terminator;
       cur = cur->getNextNode()) {
    if (auto lm2gm = dyn_cast<triton::xpu::LM2GMOp>(cur)) {
      if (!lm2gm.getLen())
        return false;
    } else if (auto sm2gm = dyn_cast<triton::xpu::SM2GMOp>(cur)) {
      if (!sm2gm.getLen())
        return false;
    } else if (auto storeOp = dyn_cast<triton::xpu::StoreOp>(cur)) {
      if (!storeOp.getMask())
        return false;
    }
  }
  return true;
}

// Whether every core owns at least one lane of the tile the predicate masks.
// The lowering hands core `c` the contiguous run starting at
// `c * elemsPerThread` with `elemsPerThread = ceil(shape / cores)` (see
// MakeRangeOpToLLVM and ClusterLayoutAttr::getElemsPerThread), so the trailing
// cores fall off the end whenever the shape is not a multiple of the core
// count. Those waves hold no element at all, yet they are still addressed as
// `base + core_id`: running them would read a lane of the tile they do not take
// part in (and store it back), so the guard has to stay even when the tail
// writes per-lane addresses. Comparing the lane count with the core count is
// not enough -- it only holds when the run per core is a single element.
bool tileCoversEveryCore(arith::CmpIOp cmpiOp) {
  auto tensorTy = dyn_cast<RankedTensorType>(cmpiOp.getResult().getType());
  if (!tensorTy)
    return false;
  auto layout =
      dyn_cast_or_null<triton::xpu::ClusterLayoutAttr>(tensorTy.getEncoding());
  if (!layout)
    return false;
  ArrayRef<int64_t> shape = tensorTy.getShape();
  ArrayRef<unsigned> coresPerGroup = layout.getCoresPerGroup();
  ArrayRef<unsigned> groupsPerCluster = layout.getGroupsPerCluster();
  if (coresPerGroup.size() != shape.size() ||
      groupsPerCluster.size() != shape.size())
    return false;
  SmallVector<unsigned> elemsPerThread = layout.getElemsPerThread(shape);
  for (unsigned d = 0; d < shape.size(); ++d) {
    int64_t cores = (int64_t)coresPerGroup[d] * (int64_t)groupsPerCluster[d];
    if ((cores - 1) * (int64_t)elemsPerThread[d] >= shape[d])
      return false;
  }
  return true;
}

// Every unpredicated store that dropping the guard would expose must address
// global memory per lane (a tile of pointers, `tl.store(out + offs, v)`): then
// the extra waves write exactly the lanes they own, which is what carries the
// `other` fill of the masked-off lanes to global memory.
//
// A store behind a scalar pointer is different: `tl.store(out, reduced)` (the
// argmax/argmin pattern) addresses the same location from every core, so the
// guard is the only thing that keeps the waves holding no lane from clobbering
// it -- and the loads feeding those waves would be issued from lane-0 addresses
// of a tile they do not take part in, reading out of bounds. Keep the guard.
bool tailWritesTileLanes(Operation *cmpiOp, Operation *terminator) {
  for (Operation *cur = cmpiOp->getNextNode(); cur && cur != terminator;
       cur = cur->getNextNode()) {
    Value ptr;
    if (auto lm2gm = dyn_cast<triton::xpu::LM2GMOp>(cur)) {
      if (lm2gm.getLen())
        continue;
      ptr = lm2gm.getPtr();
    } else if (auto sm2gm = dyn_cast<triton::xpu::SM2GMOp>(cur)) {
      if (sm2gm.getLen())
        continue;
      ptr = sm2gm.getPtr();
    } else if (auto storeOp = dyn_cast<triton::xpu::StoreOp>(cur)) {
      if (storeOp.getMask())
        continue;
      ptr = storeOp.getPtr();
    } else {
      continue;
    }
    if (!isa<RankedTensorType>(ptr.getType()))
      return false;
  }
  return true;
}

// Whether the load(s) this predicate masks define a value for the masked-off
// lanes, i.e. were given an `other` operand.
bool maskUsersDefineOtherFill(const llvm::SetVector<Operation *> &users) {
  for (Operation *user : users) {
    if (auto gm2lm = dyn_cast<triton::xpu::GM2LMOp>(user)) {
      if (gm2lm.getOther())
        return true;
    } else if (auto gm2lm = dyn_cast<triton::xpu::GM2LMMaskOp>(user)) {
      if (gm2lm.getOther())
        return true;
    }
  }
  return false;
}
} // namespace

struct TritonXPUMaskPass : public impl::TritonXPUMaskBase<TritonXPUMaskPass> {

public:
  using impl::TritonXPUMaskBase<TritonXPUMaskPass>::TritonXPUMaskBase;

  void getOpChain(llvm::SetVector<Operation *> &opChain, Operation *op) {
    if (!op || opChain.contains(op))
      return;

    opChain.insert(op);

    if (op->use_empty())
      return;

    for (Operation *user : op->getUsers()) {
      getOpChain(opChain, user);
    }
  }

  bool isFindUserOpImpl(Operation *startingOp, Operation *targetOp,
                        llvm::SetVector<Operation *> &visitedOps) {
    if (startingOp == targetOp) {
      return true;
    }

    if (!startingOp || visitedOps.contains(startingOp)) {
      return false;
    }

    visitedOps.insert(startingOp);

    for (auto userOp : startingOp->getUsers()) {
      if (isFindUserOpImpl(userOp, targetOp, visitedOps)) {
        return true;
      }
    }

    return false;
  }

  bool isFindUserOp(Operation *startingOp, Operation *targetOp) {
    llvm::SetVector<Operation *> visitedOps;
    return isFindUserOpImpl(startingOp, targetOp, visitedOps);
  }

  // Block Access Optimization Degradation
  // If the loadOp/storeOp .len() is not computed by subIOp, the block
  // read/write optimizations cannot be applied.
  void blockAccessOptDeGrad(mlir::ModuleOp m) {
    m.walk([&](triton::xpu::GM2LMOp gm2lmOp) {
      auto len = gm2lmOp.getLen();
      if (len) {
        auto subIOp = len.getDefiningOp<mlir::arith::SubIOp>();
        if (!subIOp) {
          OpBuilder builder(gm2lmOp);
          auto loc = gm2lmOp->getLoc();
          gm2lmOp->setAttr("offsetState",
                           builder.getSI32IntegerAttr(
                               static_cast<int32_t>(OffsetState::Unknown)));
        }
      }
    });

    m.walk([&](triton::xpu::LM2GMOp lm2gmOp) {
      auto len = lm2gmOp.getLen();
      if (len) {
        auto subIOp = len.getDefiningOp<mlir::arith::SubIOp>();
        if (!subIOp) {
          OpBuilder builder(lm2gmOp);
          auto loc = lm2gmOp->getLoc();
          lm2gmOp->setAttr("offsetState",
                           builder.getSI32IntegerAttr(
                               static_cast<int32_t>(OffsetState::Unknown)));
        }
      }
    });

    m.walk([&](triton::xpu::SM2GMOp sm2gmOp) {
      auto len = sm2gmOp.getLen();
      if (len) {
        auto subIOp = len.getDefiningOp<mlir::arith::SubIOp>();
        if (!subIOp) {
          OpBuilder builder(sm2gmOp);
          auto loc = sm2gmOp->getLoc();
          sm2gmOp->setAttr("offsetState",
                           builder.getSI32IntegerAttr(
                               static_cast<int32_t>(OffsetState::Unknown)));
        }
      }
    });

    m.walk([&](triton::xpu::GM2LMMaskOp gm2lmOp) {
      auto len = gm2lmOp.getLen();
      if (len) {
        auto subIOp = len.getDefiningOp<mlir::arith::SubIOp>();
        if (!subIOp) {
          OpBuilder builder(gm2lmOp);
          auto loc = gm2lmOp->getLoc();
          gm2lmOp->setAttr("offsetState",
                           builder.getSI32IntegerAttr(
                               static_cast<int32_t>(OffsetState::Unknown)));
        }
      }
    });

    m.walk([&](triton::xpu::LM2GMMaskOp lm2gmOp) {
      auto len = lm2gmOp.getLen();
      if (len) {
        auto subIOp = len.getDefiningOp<mlir::arith::SubIOp>();
        if (!subIOp) {
          OpBuilder builder(lm2gmOp);
          auto loc = lm2gmOp->getLoc();
          lm2gmOp->setAttr("offsetState",
                           builder.getSI32IntegerAttr(
                               static_cast<int32_t>(OffsetState::Unknown)));
        }
      }
    });

    m.walk([&](triton::xpu::SM2GMMaskOp sm2gmOp) {
      auto len = sm2gmOp.getLen();
      if (len) {
        auto subIOp = len.getDefiningOp<mlir::arith::SubIOp>();
        if (!subIOp) {
          OpBuilder builder(sm2gmOp);
          auto loc = sm2gmOp->getLoc();
          sm2gmOp->setAttr("offsetState",
                           builder.getSI32IntegerAttr(
                               static_cast<int32_t>(OffsetState::Unknown)));
        }
      }
    });
  }

  void addThreadIdMask(mlir::scf::IfOp ifOp, triton::xpu::LoadOp loadOp) {

    auto resTensorType = mlir::cast<RankedTensorType>(loadOp.getType());

    OpBuilder builder(ifOp);
    auto loc = ifOp->getLoc();

    // Step 1. get val
    auto rcvVal = builder.create<mlir::triton::xpu::ExtractOp>(
        loc, builder.getI64Type(), builder.getI32IntegerAttr(0), loadOp);
    auto rcvValI32 = builder.create<mlir::arith::TruncIOp>(
        loc, builder.getI32Type(), rcvVal);

    // Step 2. get ThreadNum
    auto clusterNum = builder.create<mlir::gpu::GridDimOp>(
        loc, builder.getIndexType(), mlir::gpu::Dimension::x);
    auto clusterNum_cast = builder.create<mlir::arith::IndexCastOp>(
        loc, builder.getI32Type(), clusterNum);
    auto coreNum = builder.create<mlir::gpu::BlockDimOp>(
        loc, builder.getIndexType(), mlir::gpu::Dimension::x);
    auto coreNum_cast = builder.create<mlir::arith::IndexCastOp>(
        loc, builder.getI32Type(), coreNum);
    auto threadNum = builder.create<mlir::arith::MulIOp>(
        loc, builder.getI32Type(), clusterNum_cast, coreNum_cast);

    // Step 3. val % ThreadNum
    auto remFOp =
        builder.create<mlir::arith::RemSIOp>(loc, rcvValI32, threadNum);

    // Step 4. get ThreadId
    auto threadIdOp = builder.create<mlir::triton::xpu::GetThreadIdOp>(
        loc, builder.getI32Type(), builder.getSI32IntegerAttr(1));

    // Step 5. threadId == val % ThreadNum
    auto threadCond = builder.create<mlir::arith::CmpIOp>(
        loc, builder.getI1Type(), mlir::arith::CmpIPredicate::eq, remFOp,
        threadIdOp);

    auto originCondOp = ifOp.getCondition();

    auto newCondOp = builder.create<mlir::arith::AndIOp>(
        loc, builder.getI1Type(), originCondOp, threadCond);

    ifOp->setOperand(0, newCondOp);
  }

  void atoNaiveMask(mlir::ModuleOp m) {
    m.walk([&](scf::IfOp ifOp) {
      OpBuilder builder(ifOp);
      auto loc = ifOp->getLoc();

      auto threadIdOp = builder.create<mlir::triton::xpu::GetCoreIdOp>(
          loc, builder.getI32Type());

      auto core0Op = builder.create<mlir::arith::ConstantIntOp>(loc, 0, 32);

      auto atomicCondOp = builder.create<mlir::arith::CmpIOp>(
          loc, builder.getI1Type(), mlir::arith::CmpIPredicate::eq, core0Op,
          threadIdOp);

      auto originCondOp = ifOp.getCondition();

      auto newCondOp = builder.create<mlir::arith::AndIOp>(
          loc, builder.getI1Type(), originCondOp, atomicCondOp);

      ifOp->setOperand(0, newCondOp);
    });
  }

  void atoOptimizationMask(mlir::ModuleOp m) {
    /*********************** Atomic Mask Opt **************************
    * %cst_0 = arith.constant dense<-1> : tensor<2048xi64>
    * %cst_1 = arith.constant dense<0.000000e+00> : tensor<2048xf32>
    *
    * Before Optimization:
    * %1 = tt.load: tensor<2048xi64>
    * %2 = tt.load: tensor<2048xf32>
    * %3 = arith.cmpi eq, %1, %cst_0 -> tensor<2048xi1>
    * %4 = arith.select %3, %cst_1, %2 -> tensor<2048xf32>,
    * %5 = tt.atomic_add %4 -> tensor<2048xf32>

    * After Optimization:
    * %1 = tt.load: tensor<2048xi64>
    * %3 = arith.cmpi eq, %1, %cst_0 -> tensor<2048xi1>
    * %4 = arith.neg %3        ------>     negativeCond
    * scf.if %4 {
    *   %2 = tt.load: tensor<2048xf32>
    *   %5 = tt.atomic_add %4 -> tensor<2048xf32>
    * }
    *
    *****************************************************************/
    arith::CmpIOp rcvCmpIOp;
    triton::xpu::LoadOp rcvLoadOp;
    arith::SelectOp rcvSelectOp;
    AtomicMaskCond atoMaskCond = AtomicMaskCond::NonActivate;
    bool atoMaskOpt = false;

    m.walk([&](arith::CmpIOp cmpIOp) {
      if (cmpIOp.getPredicate() == arith::CmpIPredicate::eq) {
        auto lhs = cmpIOp.getLhs();
        auto rhs = cmpIOp.getRhs();
        auto res = cmpIOp.getResult();

        // Assert Only Have One SelectOp User
        auto cmpiop_user_begin = cmpIOp->user_begin();
        auto cmpiop_user_end = cmpIOp->user_end();
        if (std::distance(cmpiop_user_begin, cmpiop_user_end) != 1)
          return;

        if (auto selectOp =
                dyn_cast<mlir::arith::SelectOp>(*cmpiop_user_begin)) {

          // Assert Only Have One AddFOp User
          auto selectop_user_begin = selectOp->user_begin();
          auto selectop_user_end = selectOp->user_end();
          if (std::distance(selectop_user_begin, selectop_user_end) != 1)
            return;

          if (auto addFOp = dyn_cast<arith::AddFOp>(*selectop_user_begin)) {
            auto trueVal = selectOp.getTrueValue();
            auto falseVal = selectOp.getFalseValue();

            auto trueLoadOp =
                dyn_cast<triton::xpu::LoadOp>(trueVal.getDefiningOp());
            auto falseConstOp =
                dyn_cast<arith::ConstantOp>(falseVal.getDefiningOp());

            auto trueConstOp =
                dyn_cast<arith::ConstantOp>(trueVal.getDefiningOp());
            auto falseLoadOp =
                dyn_cast<triton::xpu::LoadOp>(falseVal.getDefiningOp());

            if (trueLoadOp && falseConstOp) {
              if (auto denseAttr = mlir::dyn_cast<DenseElementsAttr>(
                      falseConstOp.getValue())) {
                float constValue = *denseAttr.getValues<float>().begin();
                if (constValue == 0.0f)
                  atoMaskCond = AtomicMaskCond::PostiveCond;
              }
            } else if (trueConstOp && falseLoadOp) {
              if (auto denseAttr = mlir::dyn_cast<DenseElementsAttr>(
                      trueConstOp.getValue())) {
                float constValue = *denseAttr.getValues<float>().begin();
                if (constValue == 0.0f)
                  atoMaskCond = AtomicMaskCond::NegativeCond;
              }
            }

            auto lhsLoadOp = dyn_cast<triton::xpu::LoadOp>(lhs.getDefiningOp());
            auto rhsConstOp = dyn_cast<arith::ConstantOp>(rhs.getDefiningOp());

            auto lhsConstOp = dyn_cast<arith::ConstantOp>(lhs.getDefiningOp());
            auto rhsLoadOp = dyn_cast<triton::xpu::LoadOp>(rhs.getDefiningOp());

            if ((lhsLoadOp && rhsConstOp) || (lhsConstOp && rhsLoadOp)) {
              rcvCmpIOp = cmpIOp;
              rcvLoadOp = lhsLoadOp ? lhsLoadOp : rhsLoadOp;
              rcvSelectOp = selectOp;
              atoMaskOpt = true;
            }
          }
        }
      }
    });

    // Step 1. Create If
    if (rcvCmpIOp && rcvLoadOp && atoMaskOpt &&
        atoMaskCond != AtomicMaskCond::NonActivate) {
      Operation *nextOp = rcvLoadOp->getNextNode();
      // Move the cmpi operation right after the load operation
      OpBuilder builder(rcvLoadOp);
      auto loc = rcvLoadOp->getLoc();
      builder.setInsertionPointAfter(rcvLoadOp);

      auto newCmpiOp = builder.create<mlir::arith::CmpIOp>(
          loc, rcvCmpIOp.getType(), mlir::arith::CmpIPredicate::eq,
          rcvCmpIOp.getLhs(), rcvCmpIOp.getRhs());
      rcvCmpIOp.replaceAllUsesWith(newCmpiOp.getResult());
      rcvCmpIOp.erase();

      auto trueValue = builder.create<mlir::arith::ConstantOp>(
          loc, builder.getI1Type(),
          builder.getIntegerAttr(builder.getI1Type(), 1));

      auto posCond = builder.create<mlir::triton::xpu::ExtractOp>(
          loc, builder.getI1Type(), builder.getI32IntegerAttr(0), newCmpiOp);

      auto negCond =
          builder.create<mlir::arith::XOrIOp>(loc, trueValue, posCond);

      // Create the SCF if operation
      scf::IfOp scfIfOp;
      if (atoMaskCond == AtomicMaskCond::PostiveCond) {
        scfIfOp = builder.create<mlir::scf::IfOp>(loc, posCond,
                                                  /*withElseRegion=*/false);
      } else if (atoMaskCond == AtomicMaskCond::NegativeCond) {
        scfIfOp = builder.create<mlir::scf::IfOp>(loc, negCond,
                                                  /*withElseRegion=*/false);
      }

      addThreadIdMask(scfIfOp, rcvLoadOp);

      // Move subsequent operations inside the ifOp's then block
      builder.setInsertionPointToStart(&scfIfOp.getThenRegion().front());
      Operation *yieldOp = scfIfOp.getThenRegion().front().getTerminator();
      while (nextOp) {
        Operation *currentOp = nextOp;
        nextOp = nextOp->getNextNode();
        if (!isa<scf::YieldOp>(currentOp)) {
          currentOp->moveBefore(yieldOp);
        }
      }
    }

    // Step 2. Eliminate Unvalid SelectOp
    if (rcvCmpIOp && rcvLoadOp && atoMaskOpt &&
        atoMaskCond != AtomicMaskCond::NonActivate) {
      auto true_value = rcvSelectOp.getTrueValue();
      auto false_value = rcvSelectOp.getFalseValue();

      if (atoMaskCond == AtomicMaskCond::PostiveCond) {
        rcvSelectOp.replaceAllUsesWith(true_value);
      } else if (atoMaskCond == AtomicMaskCond::NegativeCond) {
        rcvSelectOp.replaceAllUsesWith(false_value);
      }
      rcvSelectOp->erase();
    }
  }

  bool isOptMask(mlir::ModuleOp m) {
    bool isOpt = false;
    m.walk([&](triton::xpu::LM2GMOp lm2gmOp) {
      auto op = findDefOpBwd<mlir::arith::SelectOp>(lm2gmOp.getValue());
      if (op) {
        auto selectOp = cast<mlir::arith::SelectOp>(op);
        if (auto cmpIOp = dyn_cast<mlir::arith::CmpIOp>(
                selectOp.getCondition().getDefiningOp())) {
          if (cmpIOp.getPredicate() == arith::CmpIPredicate::eq) {
            isOpt = true;
          }
        }
      }
    });
    return isOpt;
  }

  // Add AtomicOp Simulation Condition
  // We must use the specifiy core to avoid access race
  void addAtomicSimulationCond(mlir::ModuleOp m) {
    bool atomicSim = false;
    m->walk([&](triton::xpu::GM2LMOp gm2lmOp) {
      auto tensorTy = gm2lmOp.getResult().getType();
      if (auto rankTensorTy = mlir::dyn_cast<RankedTensorType>(tensorTy)) {
        auto gEncoding = mlir::cast<triton::xpu::ClusterLayoutAttr>(
            rankTensorTy.getEncoding());
        auto coresPerGroup = gEncoding.getCoresPerGroup();
        auto groupsPerCluster = gEncoding.getGroupsPerCluster();

        atomicSim = (llvm::find_if(coresPerGroup,
                                   [](unsigned int num) { return num != 1; }) ==
                     coresPerGroup.end()) &&
                    (llvm::find_if(groupsPerCluster, [](unsigned int num) {
                       return num != 1;
                     }) == groupsPerCluster.end());

        if (findUserOp<triton::ReduceOp>(gm2lmOp) ||
            findUserOp<triton::xpu::ReduceOp>(gm2lmOp)) {
          atomicSim = false;
        }
      }
    });

    m->walk([&](triton::xpu::GM2LMMaskOp gm2lmOp) {
      auto tensorTy = gm2lmOp.getResult().getType();
      if (auto rankTensorTy = mlir::dyn_cast<RankedTensorType>(tensorTy)) {
        auto gEncoding = mlir::cast<triton::xpu::ClusterLayoutAttr>(
            rankTensorTy.getEncoding());
        auto coresPerGroup = gEncoding.getCoresPerGroup();
        auto groupsPerCluster = gEncoding.getGroupsPerCluster();

        atomicSim = (llvm::find_if(coresPerGroup,
                                   [](unsigned int num) { return num != 1; }) ==
                     coresPerGroup.end()) &&
                    (llvm::find_if(groupsPerCluster, [](unsigned int num) {
                       return num != 1;
                     }) == groupsPerCluster.end());

        if (findUserOp<triton::ReduceOp>(gm2lmOp) ||
            findUserOp<triton::xpu::ReduceOp>(gm2lmOp)) {
          atomicSim = false;
        }
      }
    });

    if (!atomicSim)
      return;

    AtomicMaskType atoMaskTy = AtomicMaskType::NaiveMask;
    if (maskValue != -1 && isOptMask(m)) {
      atoMaskTy = AtomicMaskType::OptimizationMask;
    }

    switch (atoMaskTy) {
    case AtomicMaskType::NaiveMask: {
      atoNaiveMask(m);
      break;
    }
    case AtomicMaskType::OptimizationMask: {
      atoOptimizationMask(m);
      break;
    }
    default:
      llvm_unreachable("Unknown Atomic Mask Type");
    }
  }

  void addOneCoreCalculationCond(mlir::ModuleOp m) {
    m.walk([&](triton::FuncOp func) {
      auto &initOp = *(func.getOps().begin());
      //   llvm::errs() << "initOp:";
      //   initOp.dump();
      //   llvm::errs() << "\n";

      OpBuilder builder(&initOp);
      auto loc = initOp.getLoc();
      auto threadIdOp = builder.create<mlir::triton::xpu::GetCoreIdOp>(
          loc, builder.getI32Type());
      auto core0Op = builder.create<mlir::arith::ConstantIntOp>(loc, 0, 32);

      auto cond = builder.create<mlir::arith::CmpIOp>(
          loc, builder.getI1Type(), mlir::arith::CmpIPredicate::eq, core0Op,
          threadIdOp);

      auto scfIfOp = builder.create<mlir::scf::IfOp>(loc, cond,
                                                     /*withElseRegion=*/false);

      builder.setInsertionPointToStart(&scfIfOp.getThenRegion().front());
      Operation *yieldOp = scfIfOp.getThenRegion().front().getTerminator();
      Operation *nextOp = initOp.getNextNode();
      initOp.moveBefore(yieldOp);
      while (nextOp) {
        Operation *currentOp = nextOp;
        nextOp = nextOp->getNextNode();
        if (!isa<scf::YieldOp>(currentOp) &&
            !isa<triton::ReturnOp>(currentOp)) {
          currentOp->moveBefore(yieldOp);
        }
      }
    });
  }

  void runOnOperation() override {
    mlir::ModuleOp m = getOperation();

    // Step 0. Convert CmpOp+SplatOp to SplatOp+CmpOp
    m.walk([&](triton::SplatOp splatOp) {
      if (auto splatDefOp = splatOp.getSrc().getDefiningOp()) {
        if (auto cmpiOp = dyn_cast<arith::CmpIOp>(splatDefOp)) {
          OpBuilder builder(splatOp);
          auto loc = cmpiOp->getLoc();
          auto lhs = cmpiOp.getLhs();
          auto rhs = cmpiOp.getRhs();
          auto lhsElemTy = lhs.getType();
          if (auto lhsTensorTy = dyn_cast<RankedTensorType>(lhs.getType())) {
            lhsElemTy = lhsTensorTy.getElementType();
          }
          auto resTy = cast<RankedTensorType>(splatOp.getType());
          auto newSplatTy = RankedTensorType::get(resTy.getShape(), lhsElemTy,
                                                  resTy.getEncoding());
          auto lhsSplatOp =
              builder.create<triton::SplatOp>(loc, newSplatTy, lhs);
          auto rhsSplatOp =
              builder.create<triton::SplatOp>(loc, newSplatTy, rhs);
          auto newCmpiOp = builder.create<arith::CmpIOp>(
              loc, resTy, cmpiOp.getPredicate(), lhsSplatOp, rhsSplatOp);
          splatOp.replaceAllUsesWith(newCmpiOp.getResult());
        }
      }
    });

    // Step 1. Replace CmiOp with SubOp(replace mask with len)
    llvm::DenseMap<Operation *, llvm::SetVector<Operation *>> maskUsersMap;
    llvm::SetVector<Operation *> cmpiOps;

    m.walk([&](mlir::arith::CmpIOp cmpiOp) {
      llvm::SetVector<Operation *> maskUsers;
      llvm::SetVector<Operation *> cmpiOpUsers;
      if (cmpiOp.getPredicate() == arith::CmpIPredicate::slt ||
          cmpiOp.getPredicate() == arith::CmpIPredicate::ult) {
        getOpChain(cmpiOpUsers, cmpiOp);
        for (auto user : cmpiOpUsers) {
          Value mask;
          if (this->isUseMaskZero) {
            if (auto loadOp = dyn_cast<triton::xpu::GM2LMMaskOp>(user)) {
              mask = loadOp.getMask();
            }
            if (auto storeOp = dyn_cast<triton::xpu::LM2GMMaskOp>(user)) {
              mask = storeOp.getMask();
            }
            if (auto storeOp = dyn_cast<triton::xpu::SM2GMMaskOp>(user)) {
              mask = storeOp.getMask();
            }
          } else {
            if (auto loadOp = dyn_cast<triton::xpu::GM2LMOp>(user)) {
              mask = loadOp.getLen();
            }
            if (auto storeOp = dyn_cast<triton::xpu::LM2GMOp>(user)) {
              mask = storeOp.getLen();
            }
            if (auto storeOp = dyn_cast<triton::xpu::SM2GMOp>(user)) {
              mask = storeOp.getLen();
            }
          }
          if (mask) {
            if (mask == cmpiOp.getResult()) {
              if (lengthFoldIsSound(cmpiOp)) {
                cmpiOps.insert(cmpiOp);
                maskUsers.insert(user);
              }
            }
            if (auto andIOp =
                    dyn_cast<mlir::arith::AndIOp>(mask.getDefiningOp())) {
              if (isFindUserOp(cmpiOp, andIOp.getLhs().getDefiningOp())) {
                if (auto cmpiOpDef =
                        findDefOpBwd<triton::ExpandDimsOp>(andIOp.getLhs())) {
                  if (lengthFoldIsSound(cmpiOp)) {
                    cmpiOps.insert(cmpiOp);
                    auto expandDimOp = cast<triton::ExpandDimsOp>(cmpiOpDef);
                    //  expand axis = 0 means inner dim
                    if (expandDimOp.getAxis() == 0) {
                      maskUsers.insert(user);
                    }
                  }
                }
              } else if (isFindUserOp(cmpiOp,
                                      andIOp.getRhs().getDefiningOp())) {
                if (auto cmpiOpDef =
                        findDefOpBwd<triton::ExpandDimsOp>(andIOp.getRhs())) {
                  if (lengthFoldIsSound(cmpiOp)) {
                    cmpiOps.insert(cmpiOp);
                    auto expandDimOp = cast<triton::ExpandDimsOp>(cmpiOpDef);
                    //  expand axis = 0 means inner dim
                    if (expandDimOp.getAxis() == 0) {
                      maskUsers.insert(user);
                    }
                  }
                }
              }
            }
          }
        }
        if (!maskUsers.empty()) {
          maskUsersMap[cmpiOp] = maskUsers;
        }
      }
    });

    // Step 2 Replace the mask of LoadOp/StoreOp
    for (const auto &pair : maskUsersMap) {
      Value rhs;
      Value lhs;
      auto op = pair.first;
      auto users = pair.second;

      if (auto cmpiOp = dyn_cast<mlir::arith::CmpIOp>(op)) {
        rhs = cmpiOp.getRhs();
        lhs = cmpiOp.getLhs();
      } else {
        llvm_unreachable(
            "cmpiOp only is mlir::arith::CmpIOp/triton::gpu::CmpIOp");
      }

      OpBuilder builder(op);
      auto loc = op->getLoc();

      auto elemLenOp =
          builder.create<mlir::arith::SubIOp>(loc, rhs.getType(), rhs, lhs);

      // get maskValue from subIOp.rhs()
      if (rhs.getDefiningOp()) {
        if (auto rMaskConstOp =
                dyn_cast<arith::ConstantOp>(rhs.getDefiningOp())) {
          auto resTy = rMaskConstOp.getResult().getType();
          if (auto tensorType = mlir::dyn_cast<RankedTensorType>(resTy)) {
            if (auto denseAttr = mlir::dyn_cast<DenseElementsAttr>(
                    rMaskConstOp.getValue())) {
              auto values = denseAttr.getValues<mlir::APInt>();
              if (!values.empty()) {
                maskValue = values[0].getZExtValue();
              }
            }
          }
        }
      }

      for (auto user : users) {
        if (this->isUseMaskZero) {
          if (auto gm2lmOp = dyn_cast<triton::xpu::GM2LMMaskOp>(user)) {
            auto operandSegmentSizesAttr =
                gm2lmOp->getAttrOfType<DenseI32ArrayAttr>(
                    "operandSegmentSizes");
            SmallVector<int32_t, 4> operandSegmentSizes(
                operandSegmentSizesAttr.asArrayRef());
            ++operandSegmentSizes[3]; // 0: ptr, 1: mask, 2: other, 3: len, 4:
                                      // bufPtr
            gm2lmOp->setAttr("operandSegmentSizes",
                             builder.getDenseI32ArrayAttr(operandSegmentSizes));
            int numOperand = gm2lmOp->getNumOperands();
            gm2lmOp->insertOperands(gm2lmOp->getNumOperands(), {elemLenOp});
          } else if (auto lm2gmOp = dyn_cast<triton::xpu::LM2GMMaskOp>(user)) {
            auto operandSegmentSizesAttr =
                lm2gmOp->getAttrOfType<DenseI32ArrayAttr>(
                    "operandSegmentSizes");
            SmallVector<int, 5> operandSegmentSizes(
                operandSegmentSizesAttr.asArrayRef());
            ++operandSegmentSizes[3]; // 0: ptr, 1: value, 2: mask, 3: len, 4:
                                      // bufPtr
            lm2gmOp->setAttr("operandSegmentSizes",
                             builder.getDenseI32ArrayAttr(operandSegmentSizes));
            lm2gmOp->insertOperands(lm2gmOp->getNumOperands(), {elemLenOp});
          } else if (auto sm2gmOp = dyn_cast<triton::xpu::SM2GMMaskOp>(user)) {
            auto operandSegmentSizesAttr =
                sm2gmOp->getAttrOfType<DenseI32ArrayAttr>(
                    "operandSegmentSizes");
            SmallVector<int, 5> operandSegmentSizes(
                operandSegmentSizesAttr.asArrayRef());
            ++operandSegmentSizes[3]; // 0: ptr, 1: value, 2: mask, 3: len, 4:
                                      // bufPtr
            sm2gmOp->setAttr("operandSegmentSizes",
                             builder.getDenseI32ArrayAttr(operandSegmentSizes));
            sm2gmOp->insertOperands(lm2gmOp->getNumOperands(), {elemLenOp});
          }
        } else {
          if (auto loadOp = dyn_cast<triton::xpu::GM2LMOp>(user)) {
            loadOp.setOperand(1, elemLenOp.getResult());
          } else if (auto storeOp = dyn_cast<triton::xpu::LM2GMOp>(user)) {
            storeOp.setOperand(2, elemLenOp.getResult());
          } else if (auto storeOp = dyn_cast<triton::xpu::SM2GMOp>(user)) {
            storeOp.setOperand(1, elemLenOp.getResult());
          }
        }
      }
    }

    if (!this->isUseMaskZero) {
      // Step 3. Create scf::IfOp to deal with tailing
      cmpiOps = mlir::topologicalSort(cmpiOps);
      llvm::SmallVector<Operation *> sortedCmpiOps;
      // Recorded before Step 2 rewrote the `len` operands, so that a cmpiOp
      // cloned into an earlier guard still knows whether its load had an
      // `other` fill.
      llvm::SmallVector<bool> sortedHasOtherFill;
      for (auto it = cmpiOps.begin(); it != cmpiOps.end(); ++it) {
        sortedCmpiOps.emplace_back(*it);
        sortedHasOtherFill.emplace_back(
            maskUsersDefineOtherFill(maskUsersMap.lookup(*it)));
      }
      llvm::SmallVector<llvm::SetVector<Operation *>> ifBlockTrees;
      for (int i = 0; i < sortedCmpiOps.size(); ++i) {
        Operation *op = sortedCmpiOps[i];
        OpBuilder builder(op);
        mlir::Block *block = op->getBlock();
        auto loc = builder.getUnknownLoc();
        // ops to be moved into ifblock and later earsed
        llvm::SetVector<Operation *> opsToMoveAndErase;
        // Get the ops that from current to the end of the block
        mlir::Operation *terminator = block->getTerminator();
        auto it = op->getIterator();
        ++it;
        for (; &*it != terminator; ++it) {
          opsToMoveAndErase.insert(&*it);
        }
        if (auto yieldOp = dyn_cast<scf::YieldOp>(terminator)) {
          if (yieldOp.getResults().size() > 0) {
            opsToMoveAndErase.insert(terminator);
          }
        }
        auto sortedOpsToMoveAndErase = sortOpTreeBwd(opsToMoveAndErase);
        // Bail out instead of wrapping the tail only when the masked-off lanes
        // carry an `other` fill that a tailing unpredicated store has to write
        // out, that store addresses global memory per lane, and the tile covers
        // every core; otherwise keep the historical tile-skipping guard.
        if (!tailIfIsSafe(op, terminator) && sortedHasOtherFill[i] &&
            tailWritesTileLanes(op, terminator) &&
            tileCoversEveryCore(cast<arith::CmpIOp>(op)))
          continue;
        ifBlockTrees.emplace_back(sortedOpsToMoveAndErase);
        // Create scf::IfOp
        builder.setInsertionPoint(terminator);
        mlir::scf::IfOp newIfOp;
        if (auto cmpiOp = dyn_cast<mlir::arith::CmpIOp>(op)) {
          // The per-core tail guard skips every core whose lane-0 mask bit is
          // off. A value-only reduce instead needs those cores to run so their
          // `other`-filled lanes still reach the accumulation; widen the guard
          // to a constant true for that case only. argmax/argmin (index reduce)
          // and rank-2 masks are declined. See canWidenTailGuardToAllCores.
          Value cond;
          if (canWidenTailGuardToAllCores(sortedOpsToMoveAndErase.getArrayRef(),
                                          cmpiOp.getResult()))
            cond = builder.create<mlir::arith::ConstantOp>(
                loc, builder.getI1Type(),
                builder.getIntegerAttr(builder.getI1Type(), 1));
          else
            cond = builder.create<triton::xpu::ExtractOp>(
                loc, builder.getI1Type(), builder.getI32IntegerAttr(0),
                cmpiOp.getResult());
          if (auto yieldOp = dyn_cast<scf::YieldOp>(terminator)) {
            if (yieldOp.getResults().size() > 0) {
              newIfOp = builder.create<mlir::scf::IfOp>(
                  loc, yieldOp.getResults().getType(), cond,
                  /*withElseRegion=*/true);
              builder.setInsertionPointToStart(newIfOp.thenBlock());
            } else {
              newIfOp =
                  builder.create<mlir::scf::IfOp>(loc, cond,
                                                  /*withElseRegion=*/false);
              builder.setInsertionPointToStart(newIfOp.thenBlock());
            }
          } else if (auto funcRetureOp =
                         dyn_cast<triton::ReturnOp>(terminator)) {
            newIfOp = builder.create<mlir::scf::IfOp>(loc, cond,
                                                      /*withElseRegion=*/false);
            builder.setInsertionPointToStart(newIfOp.thenBlock());
          }
        } else {
          llvm_unreachable("cmpiOp only is mlir::arith::CmpIOp");
        }

        mlir::IRMapping mapping;
        for (int j = sortedOpsToMoveAndErase.size() - 1; j >= 0; --j) {
          auto bodyOp = sortedOpsToMoveAndErase[j];
          auto newBodyOp = builder.clone(*bodyOp, mapping); // Clone bodyOps
          if (auto reduceOp = dyn_cast<triton::xpu::ReduceOp>(bodyOp)) {
            auto newReduceOp = cast<triton::xpu::ReduceOp>(newBodyOp);
            ReduceOpHelper helper(reduceOp);
            ReduceOpHelper newHelper(newReduceOp);
            newHelper.setReduceId(helper.getReduceId());
            newHelper.setReduceNum(helper.getReduceNum());
          } else if (auto storeOp = dyn_cast<triton::xpu::StoreSMOp>(bodyOp)) {
            SMHelper helper(storeOp);
            SMHelper newHelper(newBodyOp);
            newHelper.setOffset(helper.getOffset());
          } else if (auto sm2gmOp = dyn_cast<triton::xpu::SM2GMOp>(bodyOp)) {
            SMHelper helper(sm2gmOp);
            SMHelper newHelper(newBodyOp);
            newHelper.setOffset(helper.getOffset());
          }
        }

        if (auto yieldOp = dyn_cast<scf::YieldOp>(terminator)) {
          if (yieldOp.getResults().size() > 0) {
            builder.setInsertionPointToStart(newIfOp.elseBlock());
            auto block = yieldOp->getBlock();
            auto *region = block->getParent();
            auto *parentOp = region->getParentOp();
            if (auto parentForOp = dyn_cast<scf::ForOp>(parentOp)) {
              // Create YiledOp For newIfOp
              if (parentForOp.getRegionIterArgs().size() > 0) {
                builder.create<mlir::scf::YieldOp>(
                    loc, parentForOp.getRegionIterArgs());
              }
              // Create YiledOp For YieldOp's ParentOp
              builder.setInsertionPointToEnd(parentForOp.getBody());
              auto ifResults = newIfOp.getResults();
              builder.create<mlir::scf::YieldOp>(loc, ifResults);
            } else if (auto parentIfOp = dyn_cast<scf::IfOp>(parentOp)) {
              // Create YiledOp For newIfOp
              auto elseYieldOp = parentIfOp.elseYield();
              auto elseResults = elseYieldOp.getResults();
              builder.create<mlir::scf::YieldOp>(loc, elseResults);
              // Create YiledOp For YieldOp's ParentOp
              builder.setInsertionPointToEnd(parentIfOp.getBody());
              builder.create<mlir::scf::YieldOp>(loc, newIfOp.getResults());
            } else {
              llvm_unreachable("Unknown Mask YiledOp Pattern");
            }
          }
        }

        // update next iteration of sortedCmpiOps
        for (int k = i + 1; k < sortedCmpiOps.size(); ++k) {
          auto mappedMaskVal =
              mapping.lookupOrDefault(sortedCmpiOps[k]->getResult(0));
          sortedCmpiOps[k] = mappedMaskVal.getDefiningOp();
        }
        // Erase Old Ops
        for (auto op : sortedOpsToMoveAndErase) {
          if (op->getParentOp() != nullptr) {
            op->erase();
          }
        }
      }
    }

    // Step 3. Block Read/Write Optimizations Degradation
    blockAccessOptDeGrad(m);

    // Step 4. Add AtomicOp Simulation Conditon
    addAtomicSimulationCond(m);

    if (oneCoreActOnly)
      addOneCoreCalculationCond(m);
  }

private:
  int32_t maskValue = -1;
};

} // namespace xpu
} // namespace triton
} // namespace mlir
