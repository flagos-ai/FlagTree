//===----------------------------------------------------------------------===//
// TLE Core Tiling: distribute a 2D working-tile across the coreNum cores using
// a SINGLE unified parametric ClusterLayout, canonicalized to order == [0, 1].
//
// The XPU TLE compile path never runs the normal tritonxpu-core-tiling pass, so
// every tensor keeps the *default* ClusterLayout, which splits the LAST dim
// across all `core_num` cores (coresPerGroup = [1, core_num]). For a 2D
// row-reduce (axis == rank-1) that forces a cross-core reduction, only correct
// on the fixed TLE memory partition when the row block is 1 (XBLOCK == 1). Pure
// elementwise kernels stay stuck at XBLOCK == 1 as well. Either way this wastes
// (core_num-1)/core_num of the row parallelism.
//
// UNIFIED LAYOUT (one formula; RowTiled / LargeN are just the g==1 / g>1
// forms):
//   sizePerCore      = [ceil(M, ngroup), ceil(N, g)]
//   coresPerGroup    = [1, g]
//   groupsPerCluster = [ngroup, 1]
//   order            = [0, 1]
// The parameters (g, ngroup) are derived from the anchor M vs coreNum:
//   * M % coreNum == 0            -> g = 1,          ngroup = coreNum
//                                    (rows own whole cores; rpc = M/coreNum;
//                                     reduce is core-LOCAL, isCoreSynchronous)
//   * coreNum % M == 0 && M<coreNum-> g = coreNum/M, ngroup = M    (requires
//   N%g==0)
//                                    (one row split across g col-cores; rpc =
//                                    1;
//                                     reduce is CROSS-CORE, smem + barrier)
//   * otherwise                   -> no-op (safe fallback to today's path).
// Module attrs follow: threads-per-warp == product(cpg) == g,
// num-warps == product(gpc) == ngroup. Both g==1 and g>1 are the SAME layout
// the non-TLE CoreTiling pass emits (order == [0, 1]); the local-vs-cross-core
// reduce and block-vs-cyclic 1D output are derived from the layout in the
// lowering (isCoreSynchronous, cpg[0] > 1), NOT branched on here.
//
// ANCHOR (reduce-priority + fallback): if there is a 2D tt.reduce
// (axis==rank-1, ClusterLayout source) use its [M, N]; else fall back to any 2D
// ClusterLayout working-tile (shape[0] > 1) and take dim0 = M. No 2D
// working-tile (pure 1D, e.g. a data-dependent gather kernel) -> no-op.
//
// ORDER == [0, 1] canonicalization: for rpc == 1 (all real ams kernels) both
// orders emit the identical (0, n) slots, so this is transparent. For rpc > 1
// (synthetic tests only) the RowTiled result is stamped order [1, 0], so the
// reduce coreDealMultiRows gate (colMajorEmitted = order[0] == 0) is already
// false and the multi-row transpose never fires (a double-flip is avoided).
//
// MEMORY STAMP: because g==1 sets threads-per-warp == 1, the legacy flat
// row-major copy path (which cuts by threads-per-warp) would hand the whole
// tile to a single core. So EVERY copy_g2l / copy_l2g / local_alloc is stamped
// with its actual ClusterLayout (`xpu.tile_layout`) to force the layout-driven
// segment planner (LoadStoreOpToLLVM Branch C).
//
// Runs BEFORE tritonxpu-tle-legalize, so reduces are still triton::ReduceOp.
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypeInterfaces.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/Debug.h"

#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "triton/Dialect/TritonXPU/Transforms/Passes.h"

#include <algorithm>

#define DEBUG_TYPE "tritonxpu-tle-core-tiling"

namespace mlir {
namespace triton {
namespace xpu {

#define GEN_PASS_DEF_TRITONXPUTLECORETILING
#include "triton/Dialect/TritonXPU/Transforms/Passes.h.inc"

struct TritonXPUTLECoreTilingPass
    : public impl::TritonXPUTLECoreTilingBase<TritonXPUTLECoreTilingPass> {

  using impl::TritonXPUTLECoreTilingBase<
      TritonXPUTLECoreTilingPass>::TritonXPUTLECoreTilingBase;

  TritonXPUTLECoreTilingPass() = default;
  TritonXPUTLECoreTilingPass(bool dumpFlag, unsigned bufferSize,
                             unsigned coreNum) {
    this->dumpFlag = dumpFlag;
    this->bufferSize = bufferSize;
    this->coreNum = coreNum;
  }

  // Unified tiling parameters (set by decideTiling). groupSize == 0 means the
  // pass is disabled (no-op). Otherwise:
  //   groupSize      == product(coresPerGroup) == module threads-per-warp
  //                 (1 => RowTiled / local reduce; >1 => LargeN / cross-core).
  //   numGroups == product(groupsPerCluster) == module num-warps.
  // (groupSize, numGroups) come from the anchor M vs coreNum (see
  // decideTiling):
  //   M % coreNum == 0             -> (1, coreNum).
  //   coreNum % M == 0, M<coreNum  -> (coreNum/M, M).
  unsigned groupSize = 0;
  unsigned numGroups = 0;

  bool tilingEnabled() const { return this->groupSize != 0; }

  // RowTiled (groupSize == 1): each core owns whole rows; the last-axis reduce
  // is core-local (isCoreSynchronous). LargeN (groupSize > 1): each row is
  // split across `groupSize` cooperating column-cores; the reduce is
  // cross-core. Both are the SAME unified layout; these name the two derived
  // regimes so the callers below read as intent rather than a bare `groupSize
  // == 1` test.
  bool isRowTiled() const { return this->groupSize == 1; }
  bool isLargeN() const { return this->groupSize > 1; }

  // order is [1, 0] (row-major) only for RowTiled with more than one row per
  // core (rpc > 1): a core's LM then holds `rpc` WHOLE rows, so the col dim
  // must be the fastest for memory / elementwise / reduce to agree (see
  // getTiled2D). Every other case keeps the canonical [0, 1].
  std::vector<unsigned> chooseOrder(unsigned rowsPerCore) const {
    if (isRowTiled() && rowsPerCore > 1)
      return {1, 0};
    return {0, 1};
  }

  // Unified 2D ClusterLayout for a tensor of shape [a, b], canonical order
  // [0,1]:
  //   sizePerCore      = [ceil(a, ngroup), ceil(b, g)]
  //   coresPerGroup    = [1, g]
  //   groupsPerCluster = [ngroup, 1]
  //   order            = [0, 1]
  // g == 1 (RowTiled): [ceil(a,coreNum), b] / [1,1] / [coreNum,1] -- each of
  // the
  //   coreNum groups (warps) owns ceil(a,coreNum) WHOLE rows; the reduce (axis
  //   == last dim) is core-LOCAL (isCoreSynchronous, no smem/barrier).
  // g > 1  (LargeN):   [ceil(a,ngroup), ceil(b,g)] / [1,g] / [ngroup,1] -- each
  // of
  //   the ngroup row-groups owns rows cyclically and its g col-cores split the
  //   columns; the reduce spans the g col-cores -> CROSS-CORE (smem + barrier).
  // For rpc == sizePerCore[0] == 1 (all real ams kernels) and LargeN (g > 1)
  // order is irrelevant (single row per core, slots collapse to (0, n)): keep
  // the canonical [0, 1].
  //
  // For RowTiled (g == 1) with rpc > 1 declare order [1, 0] (方案③). A core
  // owns rpc WHOLE rows, so its LM must be ROW-MAJOR. order [1, 0] makes the
  // col dim the fastest, so emitOffsetForClusterLayout /
  // planTileSegmentsFromLayout / BroadcastOp all enumerate slots row-major and
  // mutually agree:
  //   - copy_g2l coalesces into rpc contiguous runs of N (2 DMAs), instead of
  //     the N*rpc length-1 per-element DMAs that col-major [0,1] would force
  //     (no innermost run coalesces under col-major) -> stack overflow / slow.
  //   - the flat load aval[i] = LM[i] reads row-major, matching the fill.
  //   - the broadcast result declared [1,0] emits row-major directly, so NO
  //     ConvertLayoutOp relabel is needed (addCvtForBCOp is dropped).
  //   - the reduce gate colMajorEmitted = (order[0] == 0) is false, so
  //     coreDealMultiRows stays off (no double-flip); the emitted offsets are
  //     already row-major and match the physical values.
  //   - the reduce-result slice expansion in emitOffsetForSliceLayoutXPU fires
  //     (reduce axis == last dim == order[0]), fixing rpc>1 result packing.
  Attribute getTiled2D(MLIRContext *ctx, ArrayRef<int64_t> shape) {
    std::vector<unsigned> sizePerCore = {
        ceil<unsigned>(static_cast<unsigned>(shape[0]), this->numGroups),
        ceil<unsigned>(static_cast<unsigned>(shape[1]), this->groupSize)};
    std::vector<unsigned> coresPerGroup = {1, this->groupSize};
    std::vector<unsigned> groupsPerCluster = {this->numGroups, 1};
    std::vector<unsigned> order = chooseOrder(/*rowsPerCore=*/sizePerCore[0]);
    return triton::xpu::ClusterLayoutAttr::get(ctx, sizePerCore, coresPerGroup,
                                               groupsPerCluster, order);
  }

  // Unified 1D ClusterLayout for a distributed output vector of length `len`
  // (reduce result / elementwise 1D output), canonical order [0]:
  //   sizePerCore      = [ceil(len, ngroup)]
  //   coresPerGroup    = [g]
  //   groupsPerCluster = [ngroup]
  // g == 1: [ceil(len,coreNum)] / [1] / [coreNum]  -- BLOCK writeback: core c
  // owns
  //   contiguous rows [c*rpc, c*rpc+rpc); the lowering skips the cyclic special
  //   case because cpg[0] == 1.
  // g > 1:  [ceil(len,ngroup)] / [g] / [ngroup]    -- CYCLIC writeback: the g
  //   col-cores of a group replicate the row sum; only col-core 0 writes it
  //   back at row groupId + k*ngroup (lowering keys on rank==1 && gpc[0]>1 &&
  //   cpg[0]>1).
  Attribute getOutput1D(MLIRContext *ctx, unsigned len) {
    std::vector<unsigned> sizePerCore = {ceil<unsigned>(len, this->numGroups)};
    std::vector<unsigned> coresPerGroup = {this->groupSize};
    std::vector<unsigned> groupsPerCluster = {this->numGroups};
    std::vector<unsigned> order = {0};
    return triton::xpu::ClusterLayoutAttr::get(ctx, sizePerCore, coresPerGroup,
                                               groupsPerCluster, order);
  }

  // 1D ClusterLayout equivalent to slicing dim `d` out of the 2D parent, used
  // for a make_range whose convert_layout consumer produces a
  // SliceEncodingAttr(dim = d). Keeping the make_range's own encoding equal to
  // the slice makes the convert an identity (avoids the naive cyclic-replicate
  // in ConvertLayoutOpToLLVM, which is wrong for a replicated column index).
  // Canonical order [0]; role derived from the slice dim of the 2D parent:
  //   d == 0 (column vector, per-column):
  //     g == 1  replicated:  sizePerCore=[len],            cpg=[1],  gpc=[1]
  //     g >  1  distributed over the g col-cores:
  //                           sizePerCore=[ceil(len,g)],   cpg=[g],  gpc=[1]
  //   d == 1 (row vector, distributed over the ngroup row-groups):
  //                           sizePerCore=[ceil(len,ngroup)], cpg=[1],
  //                           gpc=[ngroup]
  Attribute getCluster1DForSliceDim(MLIRContext *ctx, unsigned len,
                                    unsigned d) {
    std::vector<unsigned> sizePerCore;
    std::vector<unsigned> coresPerGroup;
    std::vector<unsigned> groupsPerCluster = {1};
    std::vector<unsigned> order = {0};
    if (d == 0) {
      if (isRowTiled()) {
        sizePerCore = {len};
        coresPerGroup = {1};
      } else {
        sizePerCore = {ceil<unsigned>(len, this->groupSize)};
        coresPerGroup = {this->groupSize};
      }
    } else {
      sizePerCore = {ceil<unsigned>(len, this->numGroups)};
      coresPerGroup = {1};
      groupsPerCluster = {this->numGroups};
    }
    return triton::xpu::ClusterLayoutAttr::get(ctx, sizePerCore, coresPerGroup,
                                               groupsPerCluster, order);
  }

  // Mechanical canonicalization for a 1D ClusterLayout WITHOUT a convert->slice
  // consumer (acc constants, 1D vectors with no slice role). Under the
  // canonical module attrs (tpw == g, num-warps == ngroup):
  //   * a distributed 1D (product(cpg) > 1, i.e. the default last-dim split
  //     cpg=[coreNum]) is flipped to cpg=[1] / gpc=[ngroup], moving the cores
  //     from coresPerGroup to groupsPerCluster (else product(cpg)=coreNum !=
  //     tpw would fail the verifier when g == 1 => tpw == 1);
  //   * a replicated 1D (product(cpg) == 1) is already canonical and kept
  //   as-is.
  Attribute flipDefault1D(MLIRContext *ctx, triton::xpu::ClusterLayoutAttr cur,
                          unsigned len) {
    unsigned coresPerGroupProduct = 1;
    for (unsigned c : cur.getCoresPerGroup())
      coresPerGroupProduct *= c;
    if (coresPerGroupProduct <= 1)
      return cur;                 // replicated: already canonical.
    return getOutput1D(ctx, len); // distributed: cpg=[g], gpc=[ngroup].
  }

  // Find the 1D role (slice dim) of a 1D vector by following its FORWARD data
  // flow to the first structural anchor:
  //   * a convert_layout producing a SliceEncoding  -> slice.getDim();
  //   * an expand_dims                              -> getAxis()
  //     (axis == the slice dim of the broadcast source: arange(N)[None,:] is
  //      axis 0 == slice dim 0 == replicated column; arange(M)[:,None] is
  //      axis 1 == slice dim 1 == distributed row).
  // The search is TRANSITIVE through 1D-preserving elementwise / cast ops
  // (extf, minsi, maxsi, addi, ...) and 1D cluster-producing converts, so that
  // every value along the chain make_range/load -> extf -> minsi -> convert ->
  // expand resolves to the SAME role. A non-transitive per-op-result lookup
  // gives mismatched operand/result encodings across an intervening cast and
  // fails the arith verifier (same-encoding constraint). Returns -1 if no
  // anchor is reachable (leave the encoding to flipDefault1D).
  int findSliceDimConsumer(Value v) {
    llvm::SmallVector<Value> worklist;
    llvm::SmallPtrSet<Operation *, 16> visited;
    worklist.push_back(v);
    while (!worklist.empty()) {
      Value cur = worklist.pop_back_val();
      for (auto *user : cur.getUsers()) {
        if (!visited.insert(user).second)
          continue;
        // Anchor 1: convert_layout -> SliceEncoding.
        if (auto cvt = dyn_cast<triton::xpu::ConvertLayoutOp>(user)) {
          auto cvtTy = dyn_cast<RankedTensorType>(cvt.getResult().getType());
          if (!cvtTy)
            continue;
          if (auto slice = dyn_cast_or_null<triton::gpu::SliceEncodingAttr>(
                  cvtTy.getEncoding()))
            return static_cast<int>(slice.getDim());
          // A 1D cluster-producing convert: keep following the chain.
          if (cvtTy.getShape().size() == 1)
            worklist.push_back(cvt.getResult());
          continue;
        }
        // Anchor 2: expand_dims (axis == the slice dim of the broadcast
        // source).
        if (auto expand = dyn_cast<triton::ExpandDimsOp>(user))
          return static_cast<int>(expand.getAxis());
        // Traverse forward through 1D-preserving ops (elementwise / casts).
        for (auto res : user->getResults()) {
          auto rt = dyn_cast<RankedTensorType>(res.getType());
          if (rt && rt.getShape().size() == 1)
            worklist.push_back(res);
        }
      }
    }
    return -1;
  }

  // Compute the unified tiling parameters (g, ngroup) from an anchor row count
  // `m` and static tile column width `n`. Returns false if `m` is not tileable
  // (or, for g > 1, if `n` does not divide across the g column-cores).
  //   * m % coreNum == 0             -> g = 1,          ngroup = coreNum
  //     (whole rows per core; rpc = m/coreNum; reduce is core-LOCAL). N is just
  //     the row length: the flat row-major memory partition and the order=[0,1]
  //     offset emission are whole-row-contiguous => ANY N is correct.
  //   * coreNum % m == 0 && m<coreNum-> g = coreNum/m,  ngroup = m (needs
  //   n%g==0)
  //     (m rows each split across g = coreNum/m cooperating column-cores; rpc =
  //     1; reduce is CROSS-CORE). Each col-core owns ceil(n, g) columns, so the
  //     static tile col width must divide evenly (g | n). The m independent
  //     rows map to groupsPerCluster and the g cooperating col-cores to
  //     coresPerGroup
  //     -- exactly the two-level layout the non-TLE CoreTiling pass emits.
  //   * otherwise                    -> not tileable (false).
  // N here is the STATIC tile width (YBLOCK), NOT the runtime matrix column
  // count -- an arbitrary matrix N is always handled by the out-of-bounds mask.
  bool computeParams(unsigned m, unsigned n, unsigned &groupSizeOut,
                     unsigned &numGroupsOut) {
    if (m % this->coreNum == 0) {
      groupSizeOut = 1;
      numGroupsOut = this->coreNum;
      return true;
    }
    if (m >= 2 && this->coreNum % m == 0) {
      unsigned candidateGroupSize = this->coreNum / m;
      if (n % candidateGroupSize != 0)
        return false;
      groupSizeOut = candidateGroupSize;
      numGroupsOut = m;
      return true;
    }
    return false;
  }

  // Outcome of the reduce-priority anchor scan (Anchor 1).
  enum class ReduceAnchor {
    Found,    // >= 1 supported reduce; (groupSize, numGroups) set.
    NoReduce, // module has no tt.reduce -> try the elementwise fallback.
    Reject    // a reduce is unsupported / reduces disagree -> no-op.
  };

  // Enable gate. Returns true only when the rewrite is provably correct against
  // the fixed TLE memory partition (see file header). Also computes the unified
  // (groupSize, numGroups) parameters. ANCHOR = reduce-priority + fallback:
  //   1. any 2D axis==last ClusterLayout reduce -> anchor on its [M, N];
  //   2. no reduce -> fall back to a 2D ClusterLayout working-tile (dim0 = M);
  //   3. neither (pure 1D, e.g. data-dependent gather) -> no-op.
  bool decideTiling(ModuleOp &mod) {
    // Reset param state (the pass instance may be reused across modules).
    this->groupSize = 0;
    this->numGroups = 0;

    switch (findReduceAnchor(mod)) {
    case ReduceAnchor::Reject:
      return false;
    case ReduceAnchor::NoReduce:
      if (!findElementwiseAnchor(mod))
        return false;
      break;
    case ReduceAnchor::Found:
      break;
    }

    if (!tilingEnabled())
      return false;

    return verifyAllTilesCompatible(mod);
  }

  // Anchor 1 (priority): every 2D reduce over the last axis with a
  // ClusterLayout source. All reduces must agree on (groupSize, numGroups).
  // Uses an interruptible walk so the scan stops at the first unsupported
  // reduce instead of pointlessly visiting the rest after deciding to bail.
  ReduceAnchor findReduceAnchor(ModuleOp &mod) {
    bool hasReduce = false;
    bool rejected = false;
    auto reject = [&](const char *reason) {
      LLVM_DEBUG(llvm::dbgs()
                 << "[TLECoreTiling] reject (reduce): " << reason << "\n");
      rejected = true;
      return WalkResult::interrupt();
    };
    mod.walk([&](triton::ReduceOp reduceOp) -> WalkResult {
      hasReduce = true;
      auto srcTy = dyn_cast<RankedTensorType>(reduceOp.getOperandTypes()[0]);
      if (!srcTy)
        return reject("reduce source is not a ranked tensor");
      auto shape = srcTy.getShape();
      if (shape.size() != 2)
        return reject("reduce source is not 2D");
      if (reduceOp.getAxis() != shape.size() - 1)
        return reject("reduce axis is not the last dim");
      auto srcEnc = srcTy.getEncoding();
      if (!srcEnc || !isa<triton::xpu::ClusterLayoutAttr>(srcEnc))
        return reject("reduce source is not a ClusterLayout");
      unsigned m = static_cast<unsigned>(shape[0]);
      unsigned n = static_cast<unsigned>(shape[1]);
      unsigned candGroupSize = 0, candNumGroups = 0;
      if (!computeParams(m, n, candGroupSize, candNumGroups))
        return reject("reduce [M, N] is not tileable");
      if (!tilingEnabled()) {
        this->groupSize = candGroupSize;
        this->numGroups = candNumGroups;
      } else if (this->groupSize != candGroupSize ||
                 this->numGroups != candNumGroups) {
        return reject("reduces disagree on (groupSize, numGroups)");
      }
      return WalkResult::advance();
    });
    if (rejected)
      return ReduceAnchor::Reject;
    return hasReduce ? ReduceAnchor::Found : ReduceAnchor::NoReduce;
  }

  // Anchor 2 (fallback): no reduce -> pick a 2D ClusterLayout working-tile
  // (dim0 = M). Take the largest shape[0] > 1 as the anchor (the working tile,
  // not a broadcast-source row vector of shape[0] == 1). No 2D tile -> no-op.
  bool findElementwiseAnchor(ModuleOp &mod) {
    unsigned anchorM = 0, anchorN = 0;
    auto scan = [&](Value v) {
      if (auto ty = dyn_cast<RankedTensorType>(v.getType())) {
        if (auto enc = dyn_cast_or_null<triton::xpu::ClusterLayoutAttr>(
                ty.getEncoding())) {
          auto shape = ty.getShape();
          if (shape.size() == 2 && static_cast<unsigned>(shape[0]) > anchorM) {
            anchorM = static_cast<unsigned>(shape[0]);
            anchorN = static_cast<unsigned>(shape[1]);
          }
        }
      }
    };
    forEachValue(mod, scan);
    if (anchorM <= 1) {
      LLVM_DEBUG(llvm::dbgs()
                 << "[TLECoreTiling] reject: no 2D working-tile (pure 1D)\n");
      return false; // no 2D working-tile (pure 1D) -> no-op.
    }
    unsigned candGroupSize = 0, candNumGroups = 0;
    if (!computeParams(anchorM, anchorN, candGroupSize, candNumGroups)) {
      LLVM_DEBUG(llvm::dbgs()
                 << "[TLECoreTiling] reject: anchor tile is not tileable\n");
      return false;
    }
    this->groupSize = candGroupSize;
    this->numGroups = candNumGroups;
    return true;
  }

  // Every 2D ClusterLayout tensor's shape[0] must be compatible with the chosen
  // params so each core owns whole rows (RowTiled) or a single row's column
  // slice (LargeN). shape[0] == 1 always covers the broadcast-source row
  // vectors produced by expand_dims(axis=0) (e.g. arange(N)[None, :]).
  //   RowTiled: shape[0] in {1} U {multiples of coreNum}.
  //   LargeN:   shape[0] == 1 or shape[0] % numGroups == 0 (rows divide evenly
  //             into the numGroups row-groups).
  bool verifyAllTilesCompatible(ModuleOp &mod) {
    bool ok = true;
    auto checkVal = [&](Value v) {
      auto ty = dyn_cast<RankedTensorType>(v.getType());
      if (!ty)
        return;
      auto enc =
          dyn_cast_or_null<triton::xpu::ClusterLayoutAttr>(ty.getEncoding());
      if (!enc)
        return;
      auto shape = ty.getShape();
      if (shape.size() != 2)
        return;
      unsigned m = static_cast<unsigned>(shape[0]);
      if (m == 1)
        return; // broadcast-source row vector: always compatible.
      unsigned divisor = isLargeN() ? this->numGroups : this->coreNum;
      if (m % divisor != 0)
        ok = false;
    };
    forEachValue(mod, checkVal);
    return ok;
  }

  // Visit every op-result and block-argument Value in the module. Shared by the
  // Anchor 2 scan and the compatibility check (both need the same traversal).
  template <typename FnT> void forEachValue(ModuleOp &mod, FnT &&fn) {
    mod.walk([&](Operation *op) {
      for (auto res : op->getResults())
        fn(res);
      for (auto &region : op->getRegions())
        for (auto &block : region)
          for (auto arg : block.getArguments())
            fn(arg);
    });
  }

  // Clone make_range so each use owns a private copy (mirrors CoreTiling's
  // recoverMakeRange). A single make_range feeding both a row and a column path
  // would otherwise need two different encodings.
  void recoverMakeRange(ModuleOp &mod) {
    // Collect first: creating ops inside the walk would let the walker visit
    // the freshly inserted make_ranges (and is generally unsafe).
    llvm::SmallVector<triton::MakeRangeOp> rangeOps;
    mod.walk([&](triton::MakeRangeOp rangeOp) { rangeOps.push_back(rangeOp); });
    for (auto rangeOp : rangeOps) {
      OpBuilder builder(rangeOp);
      auto loc = rangeOp.getLoc();
      Value rangeValue = rangeOp.getResult();
      llvm::SmallVector<mlir::OpOperand *> usesToChange;
      int i = 0;
      for (mlir::OpOperand &use : rangeValue.getUses()) {
        if (i++ > 0)
          usesToChange.push_back(&use);
      }
      for (mlir::OpOperand *operandToChange : usesToChange) {
        auto newRangeOp = builder.create<triton::MakeRangeOp>(
            loc, rangeOp.getType(), rangeOp.getStart(), rangeOp.getEnd());
        operandToChange->set(newRangeOp.getResult());
      }
    }
  }

  // Rewrite all encodings to the unified tiled layout. Orchestrates the ordered
  // steps below; each is a self-contained method so the sequence reads as a
  // pipeline rather than one long body.
  void rewriteEncodings(ModuleOp &mod, MLIRContext *context) {
    setModuleLayoutAttrs(mod, context);   // module threads-per-warp / num-warps
    rewriteResultEncodings(mod, context); // Step 1: every op-result encoding
    rebuildConstants(mod, context);       // Step 2.1: ConstantOp raw-buffer
    fixExpandDimsParents(mod, context);   // Step 2.2: expand_dims slice parent
    syncForOpTypes(mod);                  // Step 2.3: scf.for arg/result types
    fixReduceResultParents(mod,
                           context); // Step 2.4: reduce result slice parent
  }

  // Keep module-level attrs consistent with the ClusterLayout we install:
  // threads-per-warp == product(coresPerGroup) == groupSize, num-warps ==
  // product(groupsPerCluster) == numGroups. RowTiled gives (tpw==1,
  // num-warps==coreNum); LargeN gives (tpw==groupSize, num-warps==numGroups,
  // groupSize*numGroups == coreNum). Both match the non-TLE CoreTiling module
  // attrs. Set explicitly because the 3.6 TritonGPUVerifyTensorLayout interface
  // runs between passes and checks these against every encoding's cpg/gpc
  // products.
  void setModuleLayoutAttrs(ModuleOp &mod, MLIRContext *context) {
    Builder b(context);
    mod->setAttr(::mlir::triton::gpu::AttrNumThreadsPerWarp,
                 b.getI32IntegerAttr(static_cast<int32_t>(this->groupSize)));
    mod->setAttr(::mlir::triton::gpu::AttrNumWarpsName,
                 b.getI32IntegerAttr(static_cast<int32_t>(this->numGroups)));
  }

  // Step 1. Rewrite every op-result encoding.
  //   - 2D ClusterLayout        -> getTiled2D (unified formula).
  //   - 1D ClusterLayout        -> slice-dim-derived (make_range) when it has a
  //     convert->slice consumer, else mechanically canonicalized
  //     (flipDefault1D: distributed cpg=[coreNum] -> cpg=[1]/gpc=[numGroups];
  //     replicated cpg=[1] kept). The flip is REQUIRED for RowTiled (tpw == 1),
  //     where the default cpg=[coreNum] would fail the verifier.
  //   - SliceEncoding / unencoded -> skipped here (fixed in 2.2 / 2.4, or
  //     intentionally left unencoded for the TLE memory ops).
  void rewriteResultEncodings(ModuleOp &mod, MLIRContext *context) {
    mod.walk([&](Operation *op) {
      for (auto opResult : op->getResults()) {
        auto resTy = dyn_cast<RankedTensorType>(opResult.getType());
        if (!resTy)
          continue;
        auto shape = resTy.getShape();
        auto elemTy = resTy.getElementType();
        auto enc = resTy.getEncoding();
        auto cluster = dyn_cast_or_null<triton::xpu::ClusterLayoutAttr>(enc);
        if (!cluster)
          continue; // SliceEncoding handled later; unencoded left as-is.
        Attribute newEncoding;
        if (shape.size() == 2) {
          newEncoding = getTiled2D(context, shape);
        } else if (shape.size() == 1) {
          int d = findSliceDimConsumer(opResult);
          if (d < 0)
            newEncoding = flipDefault1D(context, cluster,
                                        static_cast<unsigned>(shape[0]));
          else
            newEncoding = getCluster1DForSliceDim(
                context, static_cast<unsigned>(shape[0]),
                static_cast<unsigned>(d));
        } else {
          continue; // rank > 2 unexpected: skip (safe).
        }
        opResult.setType(RankedTensorType::get(shape, elemTy, newEncoding));
      }
    });
  }

  // Step 2.1. ConstantOp: rebuild the DenseElementsAttr against the new
  // (already-updated) result type. Collect first, then rewrite: erasing the
  // walked op in-place invalidates the walker's iterator (heap corruption).
  void rebuildConstants(ModuleOp &mod, MLIRContext *context) {
    llvm::SmallVector<arith::ConstantOp> constOps;
    mod.walk([&](arith::ConstantOp constOp) {
      auto resTy = dyn_cast<RankedTensorType>(constOp.getType());
      if (resTy && resTy.getEncoding() &&
          isa<triton::xpu::ClusterLayoutAttr>(resTy.getEncoding()))
        constOps.push_back(constOp);
    });
    for (auto constOp : constOps) {
      auto newValue = constOp.getValue();
      if (auto attr = dyn_cast<mlir::DenseElementsAttr>(constOp.getValue())) {
        newValue = DenseElementsAttr::getFromRawBuffer(
            cast<ShapedType>(constOp.getType()), attr.getRawData());
      }
      OpBuilder builder(constOp);
      auto newConstOp = builder.create<mlir::arith::ConstantOp>(
          constOp.getLoc(), constOp.getType(), newValue);
      constOp.replaceAllUsesWith(newConstOp.getResult());
      constOp.erase();
    }
  }

  // Step 2.2. ExpandDimsOp: its preceding convert_layout carries a
  // SliceEncoding whose parent must become the row-tiled 2D encoding.
  void fixExpandDimsParents(ModuleOp &mod, MLIRContext *context) {
    mod.walk([&](triton::ExpandDimsOp expandOp) {
      auto expandTy = dyn_cast<RankedTensorType>(expandOp.getType());
      if (!expandTy)
        return;
      auto parent = dyn_cast_or_null<triton::xpu::ClusterLayoutAttr>(
          expandTy.getEncoding());
      if (!parent)
        return;
      auto cvt =
          expandOp.getSrc().getDefiningOp<triton::xpu::ConvertLayoutOp>();
      if (!cvt)
        return;
      auto cvtTy = dyn_cast<RankedTensorType>(cvt.getType());
      if (!cvtTy)
        return;
      auto slice =
          dyn_cast_or_null<triton::gpu::SliceEncodingAttr>(cvtTy.getEncoding());
      if (!slice)
        return;
      auto newSlice = triton::gpu::SliceEncodingAttr::get(
          context, slice.getDim(),
          cast<triton::gpu::DistributedEncodingTrait>(
              static_cast<Attribute>(parent)));
      cvt->getResult(0).setType(RankedTensorType::get(
          cvtTy.getShape(), cvtTy.getElementType(), newSlice));
    });
  }

  // Step 2.3. scf.ForOp: sync each iter-arg block argument AND result type to
  // its (already-rewritten) init-operand type, for BOTH 1D and 2D. The init
  // operands are op results rewritten in Step 1, but the block args / results
  // cannot be reached by the result walk. Copying the init types keeps the
  // loop well-formed and, crucially, keeps a 1D accumulator's iter-arg in the
  // same (flipped) encoding as its constant init and its yielded value (else
  // an in-loop arith op mixes the un-flipped default iter-arg with a flipped
  // reduce result -> arith same-encoding verifier failure, e.g. kernel_29's
  // acc = maxsi(iter_arg, reduce_result)).
  void syncForOpTypes(ModuleOp &mod) {
    mod.walk([&](scf::ForOp forOp) {
      for (auto [init, iterArg] :
           llvm::zip(forOp.getInitArgs(), forOp.getRegionIterArgs()))
        iterArg.setType(init.getType());
      for (auto [init, res] :
           llvm::zip(forOp.getInitArgs(), forOp.getResults()))
        res.setType(init.getType());
    });
  }

  // Step 2.4. ReduceOp: result SliceEncoding parent must follow the (tiled)
  // source encoding. The tt.reduce verifier (this pass runs while the reduce
  // is still a triton::ReduceOp) enforces result == slice(sourceEncoding), so
  // the parent MUST equal the source layout (now canonical order [0,1]); a
  // different one fails "inferred type incompatible with return type". The
  // rpc>1 result-packing is instead fixed in the lowering
  // (emitOffsetForSliceLayoutXPU); the LargeN cross-core read-back is handled
  // by the standard layout-driven smem+barrier reduce path.
  void fixReduceResultParents(ModuleOp &mod, MLIRContext *context) {
    mod.walk([&](triton::ReduceOp redOp) {
      for (unsigned i = 0; i < redOp->getNumResults(); ++i) {
        auto resTy = dyn_cast<RankedTensorType>(redOp.getResult()[i].getType());
        if (!resTy)
          continue;
        auto slice = dyn_cast_or_null<triton::gpu::SliceEncodingAttr>(
            resTy.getEncoding());
        if (!slice)
          continue;
        auto srcTy = cast<RankedTensorType>(redOp.getOperandTypes()[i]);
        auto srcCluster = dyn_cast_or_null<triton::xpu::ClusterLayoutAttr>(
            srcTy.getEncoding());
        if (!srcCluster)
          continue;
        auto newSlice = triton::gpu::SliceEncodingAttr::get(
            context, slice.getDim(),
            cast<triton::gpu::DistributedEncodingTrait>(
                static_cast<Attribute>(srcCluster)));
        redOp->getResult(i).setType(RankedTensorType::get(
            resTy.getShape(), resTy.getElementType(), newSlice));
      }
    });
  }

  void runOnOperation() override {
    mlir::MLIRContext *context = &getContext();
    mlir::ModuleOp mod = getOperation();

    if (!decideTiling(mod)) {
      LLVM_DEBUG(llvm::dbgs() << "[TLECoreTiling] disabled (no-op fallback)\n");
      return;
    }
    LLVM_DEBUG(llvm::dbgs()
               << "[TLECoreTiling] enabled, groupSize = " << this->groupSize
               << ", numGroups = " << this->numGroups
               << ", coreNum = " << this->coreNum << "\n");

    recoverMakeRange(mod);
    rewriteEncodings(mod, context);
    tagMemoryOps(mod, context);
    encodeComputePtrs(mod, context);
  }

  // Find the actual (already-rewritten) 1D ClusterLayout that COMPUTE uses for
  // a memory buffer, by tracing buffer -> tle_local_ptr -> tt.load result /
  // tt.store value. The memory stamp MUST equal this so copy_g2l fills exactly
  // the slots the load reads (and copy_l2g drains exactly the slots the store
  // wrote). A column-spanning buffer (bias / gamma / beta) is REPLICATED
  // (cpg=[1]/gpc=[1], each core owns all len cols); a row-spanning output
  // (reduce result) is DISTRIBUTED (cpg=[1]/gpc=[ngroup]). The accessed value's
  // encoding is used directly when present (a store value like a reduce result
  // is already a ClusterLayout); when it is unencoded (a tt.load result is left
  // unencoded, its role sits on the downstream convert) we follow the forward
  // chain with findSliceDimConsumer and rebuild via getCluster1DForSliceDim.
  // Returns null if nothing 1D is traceable (caller falls back to the
  // distributed output).
  Attribute get1DAccessLayout(MLIRContext *ctx, Value buffer, unsigned len) {
    for (auto *u : buffer.getUsers()) {
      auto lp = dyn_cast<triton::xpu::TLELocalPtrOp>(u);
      if (!lp)
        continue;
      for (auto *pu : lp.getResult().getUsers()) {
        Value accessed;
        if (auto ld = dyn_cast<triton::LoadOp>(pu))
          accessed = ld.getResult();
        else if (auto st = dyn_cast<triton::StoreOp>(pu))
          accessed = st.getValue();
        else
          continue;
        auto rt = dyn_cast<RankedTensorType>(accessed.getType());
        if (!rt || rt.getShape().size() != 1)
          continue;
        // Encoded (e.g. a store value / reduce result): use it directly.
        if (auto cl = dyn_cast_or_null<triton::xpu::ClusterLayoutAttr>(
                rt.getEncoding()))
          return cl;
        // Unencoded load result: derive the role from the forward chain.
        int d = findSliceDimConsumer(accessed);
        if (d >= 0)
          return getCluster1DForSliceDim(ctx, len, static_cast<unsigned>(d));
      }
    }
    return Attribute();
  }

  // Encode the TLE COMPUTE pointers and their loaded / stored tensors.
  //
  // tle_local_ptr / tt.load results are otherwise left UNENCODED, so the LLVM
  // type converter falls back to the DEFAULT ClusterLayout, which splits the
  // LAST dim across all coreNum cores. That default happens to give the right
  // per-core element COUNT for a 2D tile (ceil(M,coreNum)*ceil(N,1) == the
  // getTiled2D count), so it silently worked. But for a REPLICATED 1D column
  // vector (bias / gamma / beta: cpg=[1]/gpc=[1], every core owns all `len`
  // elements) the default instead DISTRIBUTES -> ceil(len,coreNum) pointers:
  // tle_local_ptr emits only len/coreNum LM pointers and tt.load reads only
  // that many values, then the downstream convert_layout to the replicated
  // layout REPLICATES those few values to fill `len` (ConvertLayoutOpToLLVM
  // src-fewer path) => wrong data on every core.
  //
  // Stamping the compute ptr / load / store result with the buffer's REAL
  // tile_layout (identical to the copy stamp in tagMemoryOps, computed from the
  // same buffer shape) makes numElems == the LM elemsPerCore, so exactly the
  // filled slots are read. For a 2D tile the layout matches the old default
  // count (identity convert, no regression); for a 1D column vector it becomes
  // the replicated `len` count; for a 1D row output (reduce result) it becomes
  // the distributed count the store value already carries.
  void encodeComputePtrs(ModuleOp mod, MLIRContext *context) {
    if (!tilingEnabled())
      return;
    mod.walk([&](triton::xpu::TLELocalPtrOp lp) {
      auto memTy = dyn_cast<triton::gpu::MemDescType>(lp.getBuffer().getType());
      if (!memTy)
        return;
      // Only encode INPUT pointers (consumed by tt.load). An OUTPUT pointer
      // (consumed by tt.store) must keep the encoding that matches its store
      // VALUE operand: that value is the compute result whose final
      // convert_layout leaves it in the DEFAULT (unencoded) layout, and the
      // tt.store verifier requires value-type == ptr-type. Re-encoding the
      // output ptr here would break that verifier. The output count is already
      // correct under the default layout (copy_l2g is stamped with the matching
      // getTiled2D / getOutput1D in tagMemoryOps), so leave output ptrs alone.
      bool feedsStore = false;
      bool feedsLoad = false;
      for (auto *u : lp.getResult().getUsers()) {
        if (isa<triton::StoreOp>(u))
          feedsStore = true;
        else if (isa<triton::LoadOp>(u))
          feedsLoad = true;
      }
      if (feedsStore || !feedsLoad)
        return;
      auto shape = memTy.getShape();
      Attribute layout;
      if (shape.size() == 2) {
        layout = getTiled2D(context, shape);
      } else if (shape.size() == 1) {
        layout = get1DAccessLayout(context, lp.getBuffer(),
                                   static_cast<unsigned>(shape[0]));
        if (!layout)
          layout = getOutput1D(context, static_cast<unsigned>(shape[0]));
      } else {
        return;
      }
      auto setEnc = [&](Value v) {
        auto rt = dyn_cast<RankedTensorType>(v.getType());
        if (!rt)
          return;
        v.setType(
            RankedTensorType::get(rt.getShape(), rt.getElementType(), layout));
      };
      setEnc(lp.getResult());
      // The tt.load result must carry the same encoding so its result struct
      // size matches the pointer count (the load packs one value per pointer).
      // A tt.store's value is produced by compute and is already tiled; its
      // pointer operand is this (now-encoded) local_ptr, so nothing to set.
      for (auto *u : lp.getResult().getUsers())
        if (auto ld = dyn_cast<triton::LoadOp>(u))
          setEnc(ld.getResult());
    });
  }

  // Stamp the chosen ClusterLayout onto the TLE memory ops as the discardable
  // `xpu.tile_layout` attribute. Under the canonical layout g == 1 sets
  // threads-per-warp == 1, so the legacy flat row-major cut (which cuts by
  // threads-per-warp) would hand the whole tile to a single core. Therefore
  // EVERY copy_g2l / copy_l2g / local_alloc must be stamped with its actual
  // ClusterLayout so the layout-driven segment planner (LoadStoreOpToLLVM
  // Branch C) is used. Each op's layout is computed from its OWN buffer shape:
  //   * 2D source tile: getTiled2D.
  //   * 1D buffer: the SAME encoding COMPUTE uses (get1DAccessLayout) so the
  //   DMA
  //     covers exactly the loaded / stored slots -- replicated for a column
  //     vector (bias), distributed for a row output (reduce result). Falls back
  //     to getOutput1D (distributed) when no 1D access is traceable.
  void tagMemoryOps(ModuleOp mod, MLIRContext *context) {
    if (!tilingEnabled())
      return;
    auto stampFor = [&](Operation *op, ArrayRef<int64_t> shape, Value buffer) {
      Attribute layout;
      if (shape.size() == 2) {
        layout = getTiled2D(context, shape);
      } else if (shape.size() == 1) {
        layout =
            get1DAccessLayout(context, buffer, static_cast<unsigned>(shape[0]));
        if (!layout)
          layout = getOutput1D(context, static_cast<unsigned>(shape[0]));
      } else {
        return;
      }
      op->setAttr("xpu.tile_layout", layout);
    };
    mod.walk([&](triton::xpu::TLECopyGlobalToLocalOp op) {
      auto memTy =
          dyn_cast<triton::gpu::MemDescType>(op.getDstBuffer().getType());
      if (memTy)
        stampFor(op, memTy.getShape(), op.getDstBuffer());
    });
    mod.walk([&](triton::xpu::TLECopyLocalToGlobalOp op) {
      auto memTy =
          dyn_cast<triton::gpu::MemDescType>(op.getSrcBuffer().getType());
      if (memTy)
        stampFor(op, memTy.getShape(), op.getSrcBuffer());
    });
    mod.walk([&](triton::gpu::LocalAllocOp op) {
      auto memTy = dyn_cast<triton::gpu::MemDescType>(op.getType());
      if (memTy)
        stampFor(op, memTy.getShape(), op.getResult());
    });
  }
};

} // namespace xpu
} // namespace triton
} // namespace mlir
