//===----------------------------------------------------------------------===//
// TritonXPULoopInvariantStaging
//
// Stage a loop-invariant, read-only, small-packet `gm2lm` out of the grid
// `scf.for` into a cluster-shared SM buffer once in the preheader, then read
// from that staged buffer inside the loop with a single indexed scalar load
// (`triton_xpu.load_scalar_indexed`).
//
// Target pattern (test 88/89): inside the grid loop a `DiscreteSame` gm2lm
// reads one scalar `idx[x1]` from a kernel-arg array `%arg0`, where `x1` is a
// per-iteration index uniform across all lanes. Instead of doing a per-
// iteration GM->LM small packet DMA + mfence, we stage the whole `idx` array
// once outside the loop into a cluster-shared SM buffer: core 0 issues a single
// GM->SM DMA (bounded at runtime by `n_idx = ceilDiv(xnumel, stride)`) followed
// by a cluster barrier, and all 64 cores then share that one SM copy. The
// staging buffer is sized dynamically from `n_idx` and bounded only by the
// physical SM ceiling (no external budget).
//===----------------------------------------------------------------------===//

#include "triton/Analysis/NewAnalysis/Utility.h"
#include "triton/Dialect/TritonXPU/IR/Dialect.h"
#include "triton/Dialect/TritonXPU/Transforms/Passes.h"

#include "mlir/Dialect/SCF/IR/SCF.h"
#include "llvm/Support/Debug.h"

#define DEBUG_TYPE "tritonxpu-loop-invariant-staging"

namespace mlir {
namespace triton {
namespace xpu {

#define GEN_PASS_DEF_TRITONXPULOOPINVARIANTSTAGING
#include "triton/Dialect/TritonXPU/Transforms/Passes.h.inc"

namespace {

// Returns true if `v` is defined outside `loop` (i.e. loop-invariant w.r.t. the
// grid loop region).
static bool isDefinedOutside(Value v, scf::ForOp loop) {
  if (auto *def = v.getDefiningOp())
    return !loop->isAncestor(def);
  // Block argument: invariant iff its owning block is not inside the loop.
  return !loop->isAncestor(v.getParentBlock()->getParentOp());
}

// Walk back through tt.splat / tt.addptr style chains to find whether the base
// pointer of `gm2lmOp` originates from a kernel pointer argument (FuncOp block
// argument). Returns that base Value if found.
static Value findInvariantBasePtr(triton::xpu::GM2LMOp gm2lmOp,
                                   scf::ForOp loop) {
  Value ptr = gm2lmOp.getPtr();
  while (auto addptr = ptr.getDefiningOp<triton::AddPtrOp>())
    ptr = addptr.getPtr();
  if (auto splat = ptr.getDefiningOp<triton::SplatOp>())
    ptr = splat.getSrc();
  if (isDefinedOutside(ptr, loop) && isa<BlockArgument>(ptr))
    return ptr;
  return Value();
}

// Peel `gm2lmOp.getPtr()` to the index tensor feeding the addptr offset.
static Value getIndexTensor(triton::xpu::GM2LMOp gm2lmOp) {
  Value ptr = gm2lmOp.getPtr();
  if (auto addptr = ptr.getDefiningOp<triton::AddPtrOp>())
    return addptr.getOffset();
  return Value();
}

// From the index tensor `%idx = arith.divsi(%dividend, %constStride)`, return
// the constant stride. Returns 0 if the pattern does not match.
static int64_t getConstStride(Value indexTensor) {
  auto divsi = indexTensor.getDefiningOp<arith::DivSIOp>();
  if (!divsi)
    return 0;
  auto cstOp = divsi.getRhs().getDefiningOp<arith::ConstantOp>();
  if (!cstOp)
    return 0;
  auto denseAttr = dyn_cast<DenseElementsAttr>(cstOp.getValue());
  if (!denseAttr || !denseAttr.isSplat())
    return 0;
  return denseAttr.getSplatValue<APInt>().getSExtValue();
}

// Staging the first `n_idx = ceilDiv(xnumel, stride)` elements of the array is
// only correct if the per-iteration index `x1 = dividend / stride` densely
// covers `0 .. n_idx-1`, i.e. `dividend` is derived from the grid loop's
// induction variable and the per-block `make_range(0..XBLOCK)` (a contiguous
// `xindex`). If `dividend` originates from some other source (a gathered pid,
// an index loaded from memory, a value-dependent selection), the
// contiguous-prefix assumption breaks and staging would read wrong / unstaged
// entries.
//
// A contiguous `xindex` may be reshaped into multi-dimensional grid
// coordinates by constant integer div/rem (e.g. `(pid / a) % b`). Such
// decompositions keep dense coverage: each derived coordinate still ranges
// over a contiguous grid, so they are allowed to sit on the chain. Only
// *non-constant* div/rem (data-dependent reshaping) and genuinely
// coverage-breaking ops (gather load, select, ...) are rejected.
//
// Returns true only when at least one contiguous source (induction var or
// make_range) is found and no unexpected non-invariant root is reached.
static bool isContiguousIndex(Value dividend, scf::ForOp loop) {
  Value iv = loop.getInductionVar();
  SmallVector<Value, 16> work{dividend};
  SmallPtrSet<Value, 16> seen;
  bool sawContiguousSource = false;
  while (!work.empty()) {
    Value v = work.pop_back_val();
    if (!seen.insert(v).second)
      continue;

    if (v == iv) {
      sawContiguousSource = true;
      continue;
    }

    Operation *def = v.getDefiningOp();
    if (!def) {
      // Block argument other than the induction var: only safe if it is
      // loop-invariant (e.g. a kernel scalar arg used as an additive offset).
      if (isDefinedOutside(v, loop))
        continue;
      return false;
    }

    // A make_range is a contiguous 0..N source.
    if (isa<triton::MakeRangeOp, triton::xpu::MakeRangeOp>(def)) {
      sawContiguousSource = true;
      continue;
    }

    // Constants are loop-invariant scalars/splats and never break contiguity of
    // the dividend's variable part (they only scale/shift it). A constant whose
    // defining op happens to sit inside the loop body would not be reported as
    // "defined outside", so handle it explicitly here.
    if (isa<arith::ConstantOp>(def))
      continue;

    // Loop-invariant values (kernel args, anything defined before the loop)
    // cannot break contiguity of the dividend's variable part.
    if (isDefinedOutside(v, loop))
      continue;

    // Constant-divisor div/rem reshape a contiguous index into dense grid
    // coordinates; keep walking only the dividend operand. A non-constant
    // (loop-variant) divisor could reorder/sparsify -> bail out.
    if (auto divsi = dyn_cast<arith::DivSIOp>(def)) {
      if (!isDefinedOutside(divsi.getRhs(), loop))
        return false;
      work.push_back(divsi.getLhs());
      continue;
    }
    if (auto remsi = dyn_cast<arith::RemSIOp>(def)) {
      if (!isDefinedOutside(remsi.getRhs(), loop))
        return false;
      work.push_back(remsi.getLhs());
      continue;
    }

    // Linear / shape-only integer ops are allowed to sit between the
    // contiguous source and the dividend.
    if (isa<arith::AddIOp, arith::MulIOp, arith::SubIOp, arith::ExtSIOp,
            arith::ExtUIOp, arith::TruncIOp, arith::IndexCastOp,
            triton::SplatOp, triton::ExpandDimsOp, triton::BroadcastOp,
            triton::xpu::BroadcastOp, triton::xpu::ConvertLayoutOp>(def)) {
      for (Value operand : def->getOperands())
        work.push_back(operand);
      continue;
    }

    // Anything else (gather load, select, data-dependent index, ...) may
    // destroy dense coverage -> bail out conservatively.
    return false;
  }
  return sawContiguousSource;
}


// From the gm2lm `len` operand `%len = arith.subi(tt.splat(%xnumel), %idx)`,
// return the loop-invariant scalar `%xnumel`. The splat source may be wrapped
// in extension ops (e.g. `arith.extsi %arg3`); peel them to reach an invariant
// root scalar. Returns null if not matched.
static Value getXnumelFromLen(triton::xpu::GM2LMOp gm2lmOp, scf::ForOp loop) {
  Value len = gm2lmOp.getLen();
  if (!len)
    return Value();
  auto subi = len.getDefiningOp<arith::SubIOp>();
  if (!subi)
    return Value();
  auto splat = subi.getLhs().getDefiningOp<triton::SplatOp>();
  if (!splat)
    return Value();
  Value scalar = splat.getSrc();
  // Peel integer extension ops to reach the invariant root.
  while (true) {
    if (auto ext = scalar.getDefiningOp<arith::ExtSIOp>()) {
      scalar = ext.getIn();
      continue;
    }
    if (auto ext = scalar.getDefiningOp<arith::ExtUIOp>()) {
      scalar = ext.getIn();
      continue;
    }
    if (auto tr = scalar.getDefiningOp<arith::TruncIOp>()) {
      scalar = tr.getIn();
      continue;
    }
    break;
  }
  if (!isDefinedOutside(scalar, loop))
    return Value();
  return scalar;
}

} // namespace

struct TritonXPULoopInvariantStaging
    : public impl::TritonXPULoopInvariantStagingBase<
          TritonXPULoopInvariantStaging> {

  using impl::TritonXPULoopInvariantStagingBase<
      TritonXPULoopInvariantStaging>::TritonXPULoopInvariantStagingBase;

  void runOnOperation() override {
    ModuleOp m = getOperation();

    SmallVector<std::pair<scf::ForOp, triton::xpu::GM2LMOp>> candidates;
    m.walk([&](scf::ForOp loop) {
      // Only target the grid-dispatch loop produced by TritonXPULoopGrid:
      // it lives directly in the FuncOp body.
      if (!isa<triton::FuncOp>(loop->getParentOp()))
        return;

      loop.getBody()->walk([&](triton::xpu::GM2LMOp gm2lmOp) {
        auto offsetState = static_cast<OffsetState>(gm2lmOp.getOffsetState());
        bool isScalarAttr = false;
        if (auto a = gm2lmOp->getAttrOfType<BoolAttr>("isScalar"))
          isScalarAttr = a.getValue();
        bool isSmallPacket =
            isScalarAttr || offsetState == OffsetState::DiscreteSame;
        if (!isSmallPacket)
          return;
        if (!findInvariantBasePtr(gm2lmOp, loop))
          return;
        candidates.push_back({loop, gm2lmOp});
      });
    });

    for (auto &[loop, gm2lmOp] : candidates)
      tryStage(loop, gm2lmOp);
  }

  void tryStage(scf::ForOp loop, triton::xpu::GM2LMOp gm2lmOp) {
    Value base = findInvariantBasePtr(gm2lmOp, loop);
    Value indexTensor = getIndexTensor(gm2lmOp);
    if (!indexTensor)
      return;
    int64_t stride = getConstStride(indexTensor);
    if (stride <= 0)
      return;
    // The staged buffer holds the contiguous prefix base[0 .. n_idx). This is
    // only valid when the per-iteration index densely covers that prefix, i.e.
    // the dividend feeding `x1 = dividend / stride` is built from the grid
    // loop's induction variable / make_range. Reject non-contiguous pids.
    auto divsi = indexTensor.getDefiningOp<arith::DivSIOp>();
    if (!divsi || !isContiguousIndex(divsi.getLhs(), loop))
      return;
    Value xnumel = getXnumelFromLen(gm2lmOp, loop);
    if (!xnumel || !xnumel.getType().isIntOrIndex())
      return;

    // The gm2lm must feed a single load whose result we will replace.
    triton::xpu::LoadOp loadOp;
    for (auto *user : gm2lmOp.getResult().getUsers()) {
      if (auto l = dyn_cast<triton::xpu::LoadOp>(user)) {
        if (loadOp)
          return; // more than one load: bail out conservatively.
        loadOp = l;
      } else {
        return; // unexpected consumer.
      }
    }
    if (!loadOp)
      return;

    auto idxTensorTy = dyn_cast<RankedTensorType>(indexTensor.getType());
    auto loadResTy = dyn_cast<RankedTensorType>(loadOp.getResult().getType());
    if (!idxTensorTy || !loadResTy)
      return;

    Type elemTy = idxTensorTy.getElementType(); // staged scalar element type.

    // --- Build the preheader staging ops, right before the grid loop. ---
    OpBuilder builder(loop);
    Location loc = gm2lmOp.getLoc();

    auto i32Ty = builder.getI32Type();

    // Runtime length n_idx = ceilDiv(xnumel, stride) = (xnumel + stride-1)/stride.
    Type xnumelTy = xnumel.getType();
    auto sMinus1 = builder.create<arith::ConstantOp>(
        loc, xnumelTy, builder.getIntegerAttr(xnumelTy, stride - 1));
    auto strideC = builder.create<arith::ConstantOp>(
        loc, xnumelTy, builder.getIntegerAttr(xnumelTy, stride));
    Value sum = builder.create<arith::AddIOp>(loc, xnumel, sMinus1);
    Value nIdx = builder.create<arith::DivSIOp>(loc, sum, strideC);

    // n_idx as i32 (the DMA length / buffer-sizing arithmetic is byte-granular).
    Value nIdxI32 = nIdx;
    if (!xnumelTy.isInteger(32)) {
      nIdxI32 = builder.create<arith::TruncIOp>(loc, i32Ty, nIdx);
    }

    // Staging capacity is bounded purely by the physical SM ceiling, not by an
    // external budget. SM is 256KB shared cluster-wide; reduce/scan scratch
    // grows from offset 0 upward, so the staging buffer is placed at the top of
    // SM and may use at most `kStagingMaxBytes` (the rest is reserved for
    // scratch). The element capacity `availElems = kStagingMaxBytes/elemBytes`
    // is a compile-time constant derived from the staged element width.
    unsigned elemBytes = elemTy.getIntOrFloatBitWidth() / 8u;
    constexpr int32_t kSMTotalBytes = 256 * 1024;
    // Reserve the lower half of SM for reduce/scan scratch; stage into the top.
    constexpr int32_t kScratchReserveBytes = 128 * 1024;
    constexpr int32_t kStagingMaxBytes = kSMTotalBytes - kScratchReserveBytes;
    int32_t availElems = kStagingMaxBytes / static_cast<int32_t>(elemBytes);

    // bufElems = clamp(n_idx, 0, availElems) as i32 (runtime). The stage_sm DMA
    // in the preheader runs unconditionally, so the buffer size must stay within
    // the SM ceiling even when the runtime guard below selects the fallback.
    auto zeroI32 = builder.create<arith::ConstantOp>(
        loc, i32Ty, builder.getI32IntegerAttr(0));
    auto availI32 = builder.create<arith::ConstantOp>(
        loc, i32Ty, builder.getI32IntegerAttr(availElems));
    Value bufElemsV = builder.create<arith::MaxSIOp>(loc, nIdxI32, zeroI32);
    bufElemsV = builder.create<arith::MinSIOp>(loc, bufElemsV, availI32);

    // smOffset = 256KB - bufElems*elemBytes (runtime, byte granularity).
    auto elemBytesC = builder.create<arith::ConstantOp>(
        loc, i32Ty, builder.getI32IntegerAttr(static_cast<int32_t>(elemBytes)));
    Value bufBytesV = builder.create<arith::MulIOp>(loc, bufElemsV, elemBytesC);
    auto smTotalC = builder.create<arith::ConstantOp>(
        loc, i32Ty, builder.getI32IntegerAttr(kSMTotalBytes));
    Value smOffsetV = builder.create<arith::SubIOp>(loc, smTotalC, bufBytesV);

    // SM staging op: core0-only GM->SM DMA + cluster barrier. The result is a
    // scalar SM base pointer (addrspace 2) shared by all cores. `$ptr` is the
    // scalar GM base pointer and `$len` is the scalar n_idx; the DMA length is
    // clamped to bufElems by the StageSM lowering. When n_idx > availElems this
    // stages a truncated prefix; the runtime guard below makes sure the
    // truncated buffer is never read.
    auto smPtrTy = triton::PointerType::get(elemTy, /*addrSpace=*/2);
    auto stageSM = builder.create<triton::xpu::StageSMOp>(
        loc, smPtrTy, /*ptr=*/base, /*len=*/nIdxI32,
        /*smOffset=*/smOffsetV, /*bufElems=*/bufElemsV, gm2lmOp.getSyncMode());

    // Runtime guard: the staged buffer covers at most availElems elements (the
    // SM ceiling). When the live length n_idx exceeds it, staging is unsafe
    // (truncated reads -> out-of-bounds / wrong results); fall back to the
    // original per-iteration gm2lm+load. n_idx is uniform across cores, so this
    // branch is taken identically by all lanes.
    auto guardC = builder.create<arith::ConstantOp>(
        loc, xnumelTy,
        builder.getIntegerAttr(xnumelTy, static_cast<int64_t>(availElems)));
    Value useStaged = builder.create<arith::CmpIOp>(
        loc, arith::CmpIPredicate::sle, nIdx, guardC);

    // --- Inside the loop: replace the per-iteration gm2lm+load with a runtime
    // branch selecting the staged SM read or the original gm2lm fallback. ---
    OpBuilder inLoop(loadOp);
    Location lloc = loadOp.getLoc();
    auto ifOp = inLoop.create<scf::IfOp>(lloc, TypeRange{loadResTy}, useStaged,
                                         /*withElseRegion=*/true);

    // then: read the staged scalar from the cluster-shared SM buffer.
    {
      OpBuilder::InsertionGuard g(inLoop);
      inLoop.setInsertionPointToStart(ifOp.thenBlock());
      auto idxScalar = inLoop.create<triton::xpu::ExtractOp>(
          lloc, elemTy, inLoop.getI32IntegerAttr(0), indexTensor);
      auto scalarLoad = inLoop.create<triton::xpu::LoadScalarIndexedOp>(
          lloc, loadResTy, stageSM.getResult(), idxScalar,
          gm2lmOp.getSyncMode());
      inLoop.create<scf::YieldOp>(lloc, scalarLoad.getResult());
    }

    // else: original path. Move the per-iteration gm2lm + load into the else
    // region and yield the loaded value. Their operands are defined before the
    // grid loop body / before this point, so they still dominate the region.
    scf::YieldOp elseYield;
    {
      OpBuilder::InsertionGuard g(inLoop);
      gm2lmOp->moveBefore(ifOp.elseBlock(), ifOp.elseBlock()->end());
      loadOp->moveBefore(ifOp.elseBlock(), ifOp.elseBlock()->end());
      inLoop.setInsertionPointToEnd(ifOp.elseBlock());
      elseYield = inLoop.create<scf::YieldOp>(lloc, loadOp.getResult());
    }

    // Route every consumer outside the new region to the if-result; the only
    // remaining direct use of loadOp is the else-region yield.
    SmallPtrSet<Operation *, 1> except;
    except.insert(elseYield.getOperation());
    loadOp.getResult().replaceAllUsesExcept(ifOp.getResult(0), except);
  }
};

} // namespace xpu
} // namespace triton
} // namespace mlir
