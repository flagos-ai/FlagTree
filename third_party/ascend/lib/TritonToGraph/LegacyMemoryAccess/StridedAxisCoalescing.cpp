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

#include "TritonToGraph/LegacyMemoryAccess/StridedAxisCoalescing.h"

#include "TritonToGraph/GraphOptimization.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Matchers.h"
#include "llvm/Support/Debug.h"

#include <functional>

#define DEBUG_TYPE "graph-optimize"

namespace StridedAxisCoalescing {

using namespace mlir;
using namespace triton;

// Detects the FLA per-head strided base `base + (pid % S)` produced by
// splitting the H axis (the contiguous axis folded onto the grid). Returns the
// matching AddPtrOp, or a null AddPtrOp if `base` is not such an ih-split ptr.
static triton::AddPtrOp findIhAddPtr(Value base, int64_t S) {
  Value src = base;
  while (auto addptr = src.getDefiningOp<triton::AddPtrOp>()) {
    if (isa<RankedTensorType>(addptr.getPtr().getType()))
      break;
    if (auto rem = addptr.getOffset().getDefiningOp<arith::RemSIOp>()) {
      APInt cC;
      if (matchPattern(rem.getRhs(), m_ConstantInt(&cC)) &&
          std::abs(cC.getSExtValue()) == S) {
        Value lhs = rem.getLhs();
        while (true) {
          if (auto e = lhs.getDefiningOp<arith::ExtSIOp>()) {
            lhs = e.getIn();
            continue;
          }
          if (auto t = lhs.getDefiningOp<arith::TruncIOp>()) {
            lhs = t.getIn();
            continue;
          }
          break;
        }
        if (lhs.getDefiningOp<triton::GetProgramIdOp>())
          return addptr;
      }
    }
    src = addptr.getPtr();
  }
  return triton::AddPtrOp();
}

// Mirror of findIhAddPtr that returns the program_id axis driving the ih split
// (i.e. the grid dim the host launcher must divide by S), or -1 if `base` is
// not such an ih-split ptr. Whenever findIhAddPtr succeeds this does too.
static int32_t findIhAxis(Value base, int64_t S) {
  Value src = base;
  while (auto addptr = src.getDefiningOp<triton::AddPtrOp>()) {
    if (isa<RankedTensorType>(addptr.getPtr().getType()))
      break;
    if (auto rem = addptr.getOffset().getDefiningOp<arith::RemSIOp>()) {
      APInt cC;
      if (matchPattern(rem.getRhs(), m_ConstantInt(&cC)) &&
          std::abs(cC.getSExtValue()) == S) {
        Value lhs = rem.getLhs();
        while (true) {
          if (auto e = lhs.getDefiningOp<arith::ExtSIOp>()) {
            lhs = e.getIn();
            continue;
          }
          if (auto t = lhs.getDefiningOp<arith::TruncIOp>()) {
            lhs = t.getIn();
            continue;
          }
          break;
        }
        if (auto pid = lhs.getDefiningOp<triton::GetProgramIdOp>())
          return pid.getAxisAsInt();
      }
    }
    src = addptr.getPtr();
  }
  return -1;
}

// Returns the i_h value `pid % S` (the per-head index feeding the ih split), or
// null. Mirror of findIhAddPtr but yields the RemSIOp result itself, used to
// check whether i_h also feeds a per-head scalar load (which coalescing cannot
// lane-expand -- see the correctness guard in rewriteStridedAxisCoalesce).
static Value findIhRem(Value base, int64_t S) {
  Value src = base;
  while (auto addptr = src.getDefiningOp<triton::AddPtrOp>()) {
    if (isa<RankedTensorType>(addptr.getPtr().getType()))
      break;
    if (auto rem = addptr.getOffset().getDefiningOp<arith::RemSIOp>()) {
      APInt cC;
      if (matchPattern(rem.getRhs(), m_ConstantInt(&cC)) &&
          std::abs(cC.getSExtValue()) == S)
        return rem.getResult();
    }
    src = addptr.getPtr();
  }
  return Value();
}

static Value build2DBlockPtr(IRRewriter &rw, triton::MakeTensorPtrOp m1d,
                             int64_t S, int64_t BT) {
  triton::AddPtrOp ih = findIhAddPtr(m1d.getBase(), S);
  if (!ih)
    return Value();
  auto loc = m1d.getLoc();
  rw.setInsertionPoint(m1d);
  Value newBase = ih.getPtr();
  Value cH = rw.create<arith::ConstantOp>(loc, rw.getI64IntegerAttr(S));
  Value c1 = rw.create<arith::ConstantOp>(loc, rw.getI64IntegerAttr(1));
  Value c0 = rw.create<arith::ConstantOp>(loc, rw.getI32IntegerAttr(0));
  SmallVector<Value, 2> shape{m1d.getShape()[0], cH};
  SmallVector<Value, 2> strides{m1d.getStrides()[0], c1};
  SmallVector<Value, 2> offsets{m1d.getOffsets()[0], c0};
  SmallVector<int32_t, 2> blockShape{static_cast<int32_t>(BT),
                                     static_cast<int32_t>(S)};
  SmallVector<int32_t, 2> order{1, 0};
  auto p = rw.create<triton::MakeTensorPtrOp>(loc, newBase, shape, strides,
                                              offsets, blockShape, order);
  return p.getResult();
}

// Lift a rank-1 tensor type tensor<BTxe> to tensor<BTxSxe> (append the folded
// H axis as the inner lane). Scalars / non-rank-1 types pass through unchanged.
static Type lift2D(Type t, int64_t S) {
  auto rt = dyn_cast<RankedTensorType>(t);
  if (!rt || rt.getRank() != 1)
    return t;
  return RankedTensorType::get({rt.getShape()[0], S}, rt.getElementType());
}

// An op is safe to 2D-ify (lane-parallel over the appended H axis) iff every
// lane s computes independently: a pure elementwise arith/math op, a cast, a
// splat, or a scan/reduce ALONG THE T axis (axis 0). On the 2D tile [BT,S] a
// T-axis reduce is per-lane (the S lanes stay independent, output [S]) and a
// T-axis scan likewise, so both lift directly -- no need to pre-collapse the
// reverse-cumsum idiom into a single scan. Ops that mix or move the lane
// (transpose, tt.dot, reshape, reduce/scan along the lane) are NOT here, so the
// caller bails and keeps the original (indirect) path.
static bool is2DSafe(Operation *op) {
  if (isa<arith::AddFOp, arith::SubFOp, arith::MulFOp, arith::DivFOp,
          arith::NegFOp, arith::MaximumFOp, arith::MinimumFOp, arith::MaxNumFOp,
          arith::MinNumFOp, arith::CmpFOp, arith::SelectOp, arith::ExtFOp,
          arith::TruncFOp, arith::SIToFPOp, arith::UIToFPOp, arith::FPToSIOp,
          arith::FPToUIOp>(op))
    return true;
  // Every math dialect op (exp/log/sqrt/tanh/erf/...) is a per-lane scalar
  // elementwise map -> 2D-safe. This covers gate activations expressed as
  // tensor math (softplus = log(1+exp), silu, gelu, ...) without enumerating
  // each op, so such cumsum kernels coalesce without per-idiom pattern match.
  if (isa<math::MathDialect>(op->getDialect()))
    return true;
  if (isa<triton::SplatOp>(op))
    return true;
  if (auto scan = dyn_cast<triton::ScanOp>(op))
    return scan.getAxis() == 0 && scan->getNumResults() == 1;
  if (auto reduce = dyn_cast<triton::ReduceOp>(op))
    return reduce.getAxis() == 0 && reduce->getNumResults() == 1;
  return false;
}

void rewriteStridedAxisCoalesce(ModuleOp moduleOp) {
  IRRewriter rw(moduleOp.getContext());

  // Collect the strided ih-base 1D loads (seeds). All must share one stride S
  // (the folded H axis); BT is the per-chunk tile length.
  SmallVector<triton::LoadOp> seeds;
  int64_t S = 0, BT = 0;
  moduleOp.walk([&](triton::LoadOp l) {
    auto m = l.getPtr().getDefiningOp<triton::MakeTensorPtrOp>();
    if (!m)
      return;
    auto rt = dyn_cast<RankedTensorType>(l.getResult().getType());
    if (!rt || rt.getRank() != 1)
      return;
    auto strides = m.getStrides();
    if (strides.empty())
      return;
    APInt sC;
    if (!matchPattern(strides.back(), m_ConstantInt(&sC)))
      return;
    int64_t s = std::abs(sC.getSExtValue());
    if (s <= 1)
      return;
    if (!findIhAddPtr(m.getBase(), s))
      return;
    if (S == 0) {
      S = s;
      BT = rt.getShape()[0];
    }
    if (s != S)
      return;
    seeds.push_back(l);
  });
  if (seeds.empty())
    return;

  // The grid axis the launcher will divide by S (the pid feeding `pid % S`).
  int32_t coalesceAxis = -1;
  if (auto m0 = seeds.front().getPtr().getDefiningOp<triton::MakeTensorPtrOp>())
    coalesceAxis = findIhAxis(m0.getBase(), S);
  if (coalesceAxis < 0)
    return; // cannot identify the axis -> do not coalesce

  // Full TA path: the launcher divides grid[coalesceAxis] by S, so the
  // kernel-visible num_programs(coalesceAxis) becomes grid/S. If the kernel
  // reads it, coalescing would change that value -> wrong results. Bail (the
  // kernel keeps its original, correct, uncoalesced path).
  bool readsAxisNumPrograms = false;
  moduleOp.walk([&](triton::GetNumProgramsOp np) {
    if (np.getAxisAsInt() == coalesceAxis)
      readsAxisNumPrograms = true;
  });
  if (readsAxisNumPrograms)
    return;

  // The H axis is folded into the inner lane, so every per-head value must be
  // expanded across the S lanes. Block-ptr loads are lane-expanded by
  // build2DBlockPtr; a per-head SCALAR load (A_log[i_h] / dt_bias[i_h] in
  // gdn-style gates) is collected here and later lifted to an [S] per-lane
  // vector load (lane s -> base[s], matching the folded i_h = s). The scalar
  // chain on top of it (exp/neg/...) is lifted to [S] by get2D, and the splat
  // that feeds it into the tile becomes an [S]->[BT,S] broadcast. Bail only if
  // i_h reaches something we cannot lane-expand: a scalar load feeding an
  // address (indirect gather), or a tensor load not via make_tensor_ptr.
  SmallVector<triton::LoadOp> headLoads;
  if (auto m0 =
          seeds.front().getPtr().getDefiningOp<triton::MakeTensorPtrOp>()) {
    if (Value ihRem = findIhRem(m0.getBase(), S)) {
      SmallVector<Operation *> wl2(ihRem.getUsers().begin(),
                                   ihRem.getUsers().end());
      DenseSet<Operation *> seen2;
      while (!wl2.empty()) {
        Operation *u = wl2.pop_back_val();
        if (!seen2.insert(u).second)
          continue;
        if (isa<triton::MakeTensorPtrOp>(u))
          continue; // block-ptr path: fine
        if (auto ld = dyn_cast<triton::LoadOp>(u)) {
          // per-head scalar load: must be scalar and must not itself feed
          // an address (indirect gather). Then it is liftable to [S].
          if (isa<RankedTensorType>(ld.getResult().getType()))
            return;
          for (Operation *ru : ld.getResult().getUsers())
            if (isa<triton::AddPtrOp>(ru))
              return; // indirect -> bail
          headLoads.push_back(ld);
          continue;
        }
        if (isa<triton::AddPtrOp, arith::ExtSIOp, arith::TruncIOp,
                arith::RemSIOp, arith::AddIOp, arith::MulIOp>(u))
          for (Operation *uu : u->getResult(0).getUsers())
            wl2.push_back(uu);
      }
    }
  }

  // Discover the load->store subgraph by forward reachability from the seeds.
  // Every op on the way must be 2D-safe (elementwise / cast / splat / T-axis
  // scan or reduce); stores are the sinks. A T-axis reduce is kept as-is and
  // lifted in place (no idiom-specific pre-collapse) -- see is2DSafe.
  // Any unsafe op (or a value escaping to one) aborts the whole rewrite.
  DenseSet<Operation *> region;
  SmallVector<triton::StoreOp> sinks;
  DenseSet<Operation *> visited;
  SmallVector<Operation *> wl;
  for (auto s : seeds)
    for (Operation *u : s.getResult().getUsers())
      wl.push_back(u);
  while (!wl.empty()) {
    Operation *op = wl.pop_back_val();
    if (!visited.insert(op).second)
      continue;
    if (auto st = dyn_cast<triton::StoreOp>(op)) {
      sinks.push_back(st);
      continue;
    }
    if (!is2DSafe(op))
      return; // bail: unsafe consumer in the chain
    region.insert(op);
    for (Value r : op->getResults())
      for (Operation *u : r.getUsers())
        wl.push_back(u);
  }
  if (sinks.empty())
    return;

  // Every sink store must also be a matching 1D stride-S ih-base block ptr.
  for (auto st : sinks) {
    auto m = st.getPtr().getDefiningOp<triton::MakeTensorPtrOp>();
    if (!m || !findIhAddPtr(m.getBase(), S))
      return;
    auto os = m.getStrides();
    if (os.empty())
      return;
    APInt soC;
    if (!matchPattern(os.back(), m_ConstantInt(&soC)) ||
        std::abs(soC.getSExtValue()) != S)
      return;
  }

  // Map each 1D value to its 2D counterpart, materializing splats/constants
  // on demand. Returns null to signal an un-liftable operand (bail).
  DenseMap<Value, Value> vmap;
  std::function<Value(Value)> get2D = [&](Value v) -> Value {
    auto it = vmap.find(v);
    if (it != vmap.end())
      return it->second;
    if (!isa<RankedTensorType>(v.getType())) {
      // Scalar: if it derives from a per-head load (whose [S] lane vector is
      // already in vmap), lift the elementwise scalar chain to [S] so each
      // lane gets its own value. i_h-independent scalars stay scalar (they
      // splat uniformly across lanes, which is correct).
      Operation *def = v.getDefiningOp();
      bool liftable =
          def && (isa<math::MathDialect>(def->getDialect()) ||
                  isa<arith::AddFOp, arith::SubFOp, arith::MulFOp,
                      arith::DivFOp, arith::NegFOp, arith::MaximumFOp,
                      arith::MinimumFOp, arith::MaxNumFOp, arith::MinNumFOp,
                      arith::ExtFOp, arith::TruncFOp>(def));
      if (liftable) {
        SmallVector<Value> ops2;
        bool anyLane = false;
        for (Value o : def->getOperands()) {
          Value n = get2D(o);
          if (!n)
            return Value();
          if (isa<RankedTensorType>(n.getType()))
            anyLane = true;
          ops2.push_back(n);
        }
        if (anyLane) {
          OpBuilder::InsertionGuard g(rw);
          rw.setInsertionPointAfter(def);
          for (Value &o : ops2)
            if (!isa<RankedTensorType>(o.getType()))
              o = rw.create<triton::SplatOp>(
                  def->getLoc(), RankedTensorType::get({S}, o.getType()), o);
          OperationState st(def->getLoc(), def->getName());
          st.addOperands(ops2);
          st.addAttributes(def->getAttrs());
          for (Value r : def->getResults())
            st.addTypes(RankedTensorType::get({S}, r.getType()));
          Operation *nu = rw.create(st);
          vmap[v] = nu->getResult(0);
          return nu->getResult(0);
        }
      }
      return v; // i_h-independent scalar, splats uniformly
    }
    // Save/restore the insertion point: the materializers below move it next
    // to the original splat/constant (which may sit at the top of the func).
    // Without this the caller's `rw.setInsertionPoint(op)` would be clobbered
    // and the rebuilt op emitted before its operands -> dominance violation.
    OpBuilder::InsertionGuard guard(rw);
    if (auto sp = v.getDefiningOp<triton::SplatOp>()) {
      Value src2 = get2D(sp.getSrc());
      if (!src2)
        return Value();
      rw.setInsertionPointAfter(sp);
      Value n;
      if (isa<RankedTensorType>(src2.getType())) {
        // src2 is the per-lane reduce result [S] (the 1D scalar source
        // became a vector once the reduce was 2D-ified). Broadcast it
        // across T: [S] -> expand_dims(0) -> [1,S] -> broadcast [BT,S].
        Value ex = rw.create<triton::ExpandDimsOp>(sp.getLoc(), src2, 0);
        n = rw.create<triton::BroadcastOp>(sp.getLoc(), lift2D(sp.getType(), S),
                                           ex);
      } else {
        // True scalar splat: lift to a 2D splat over [BT,S].
        n = rw.create<triton::SplatOp>(sp.getLoc(), lift2D(sp.getType(), S),
                                       src2);
      }
      vmap[v] = n;
      return n;
    }
    if (auto c = v.getDefiningOp<arith::ConstantOp>()) {
      if (auto dea = dyn_cast<DenseElementsAttr>(c.getValue())) {
        if (dea.isSplat()) {
          auto nt = cast<RankedTensorType>(lift2D(c.getType(), S));
          rw.setInsertionPointAfter(c);
          Value n = rw.create<arith::ConstantOp>(
              c.getLoc(), nt,
              DenseElementsAttr::get(nt, dea.getSplatValue<Attribute>()));
          vmap[v] = n;
          return n;
        }
      }
    }
    return Value();
  };

  // Build 2D loads for the seeds.
  for (auto l : seeds) {
    auto m = l.getPtr().getDefiningOp<triton::MakeTensorPtrOp>();
    Value p2 = build2DBlockPtr(rw, m, S, BT);
    if (!p2)
      return;
    rw.setInsertionPoint(l);
    auto nl = rw.create<triton::LoadOp>(l.getLoc(), p2, ArrayRef<int32_t>{0, 1},
                                        l.getPadding(), l.getCache(),
                                        l.getEvict(), l.getIsVolatile());
    vmap[l.getResult()] = nl.getResult();
  }

  // Lift each per-head scalar load to an [S] per-lane vector load: base[0:S]
  // (lane s = base[s], matching the folded i_h = s). The offset must be exactly
  // i_h = pid % S so that lane s maps to base[s]; otherwise bail.
  for (auto ld : headLoads) {
    auto ap = ld.getPtr().getDefiningOp<triton::AddPtrOp>();
    if (!ap)
      return;
    Value off = ap.getOffset();
    while (auto e = off.getDefiningOp<arith::ExtSIOp>())
      off = e.getIn();
    while (auto t = off.getDefiningOp<arith::TruncIOp>())
      off = t.getIn();
    if (!off.getDefiningOp<arith::RemSIOp>())
      return; // offset not pure i_h
    Value base = ap.getPtr();
    Type elemTy = ld.getResult().getType();
    rw.setInsertionPoint(ld);
    auto loc = ld.getLoc();
    Value cS = rw.create<arith::ConstantOp>(loc, rw.getI64IntegerAttr(S));
    Value c1 = rw.create<arith::ConstantOp>(loc, rw.getI64IntegerAttr(1));
    Value c0 = rw.create<arith::ConstantOp>(loc, rw.getI32IntegerAttr(0));
    SmallVector<Value, 1> shape{cS}, strides{c1}, offsets{c0};
    SmallVector<int32_t, 1> blockShape{static_cast<int32_t>(S)}, order{0};
    auto p = rw.create<triton::MakeTensorPtrOp>(loc, base, shape, strides,
                                                offsets, blockShape, order);
    auto vl = rw.create<triton::LoadOp>(
        loc, p.getResult(), ArrayRef<int32_t>{0},
        triton::PaddingOption::PAD_ZERO, triton::CacheModifier::NONE,
        triton::EvictionPolicy::NORMAL, false);
    (void)elemTy;
    vmap[ld.getResult()] = vl.getResult();
  }

  // Rebuild the region ops in IR (topological) order as 2D.
  SmallVector<Operation *> ordered;
  moduleOp.walk([&](Operation *op) {
    if (region.count(op))
      ordered.push_back(op);
  });
  for (Operation *op : ordered) {
    rw.setInsertionPoint(op);
    if (auto scan = dyn_cast<triton::ScanOp>(op)) {
      Value in = get2D(scan.getOperand(0));
      if (!in)
        return;
      auto ns = rw.create<triton::ScanOp>(scan.getLoc(), ValueRange{in},
                                          static_cast<int>(scan.getAxis()),
                                          scan.getReverse());
      rw.cloneRegionBefore(scan.getCombineOp(), ns.getCombineOp(),
                           ns.getCombineOp().end());
      vmap[scan->getResult(0)] = ns->getResult(0);
      continue;
    }
    if (auto reduce = dyn_cast<triton::ReduceOp>(op)) {
      Value in = get2D(reduce.getOperand(0));
      if (!in)
        return;
      // T-axis reduce on the 2D tile [BT,S] -> [S]: one independent
      // reduction per lane (the S lanes do not mix). The 1D result was a
      // scalar; its 2D counterpart is the per-lane vector [S], which a
      // downstream splat turns into an expand_dims+broadcast (see get2D).
      auto nr = rw.create<triton::ReduceOp>(reduce.getLoc(), ValueRange{in},
                                            static_cast<int>(reduce.getAxis()));
      rw.cloneRegionBefore(reduce.getCombineOp(), nr.getCombineOp(),
                           nr.getCombineOp().end());
      vmap[reduce->getResult(0)] = nr->getResult(0);
      continue;
    }
    // Splats are materialized on demand by get2D (which lifts a scalar splat
    // to a 2D splat, and a per-lane reduce result [S] to expand_dims +
    // broadcast). Skip here so it is not rebuilt as an invalid 2D splat.
    if (isa<triton::SplatOp>(op))
      continue;
    SmallVector<Value> operands;
    for (Value o : op->getOperands()) {
      Value n = get2D(o);
      if (!n)
        return;
      operands.push_back(n);
    }
    OperationState st(op->getLoc(), op->getName());
    st.addOperands(operands);
    st.addAttributes(op->getAttrs());
    for (Value r : op->getResults())
      st.addTypes(lift2D(r.getType(), S));
    Operation *nu = rw.create(st);
    for (auto [oldR, newR] : llvm::zip(op->getResults(), nu->getResults()))
      vmap[oldR] = newR;
  }

  // Build the 2D stores.
  for (auto st : sinks) {
    Value val = get2D(st.getValue());
    if (!val)
      return;
    auto m = st.getPtr().getDefiningOp<triton::MakeTensorPtrOp>();
    Value p2 = build2DBlockPtr(rw, m, S, BT);
    if (!p2)
      return;
    rw.setInsertionPoint(st);
    rw.create<triton::StoreOp>(st.getLoc(), p2, val, ArrayRef<int32_t>{0, 1},
                               st.getCache(), st.getEvict());
  }

  // i_b = divsi(get_program_id, S): with the H axis folded into the inner
  // tile the per-instance i_b becomes the raw program id. Redirect it (this
  // also fixes the new 2D block ptr bases, which reuse i_b).
  if (auto m0 =
          seeds.front().getPtr().getDefiningOp<triton::MakeTensorPtrOp>()) {
    if (triton::AddPtrOp ihA = findIhAddPtr(m0.getBase(), S)) {
      if (auto rem = ihA.getOffset().getDefiningOp<arith::RemSIOp>()) {
        Value lhs = rem.getLhs();
        while (true) {
          if (auto e = lhs.getDefiningOp<arith::ExtSIOp>()) {
            lhs = e.getIn();
            continue;
          }
          if (auto t = lhs.getDefiningOp<arith::TruncIOp>()) {
            lhs = t.getIn();
            continue;
          }
          break;
        }
        SmallVector<arith::DivSIOp, 2> divs;
        for (Operation *u : lhs.getUsers())
          if (auto dv = dyn_cast<arith::DivSIOp>(u)) {
            APInt dC;
            if (dv.getLhs() == lhs &&
                matchPattern(dv.getRhs(), m_ConstantInt(&dC)) &&
                std::abs(dC.getSExtValue()) == S)
              divs.push_back(dv);
          }
        for (auto dv : divs)
          rw.replaceAllUsesWith(dv.getResult(), lhs);
      }
    }
  }

  LLVM_DEBUG(
      llvm::dbgs() << "[" DEBUG_TYPE "] matched graph optimization rule "
                   << static_cast<unsigned>(
                          cfg::GraphOptimizationRuleId::StridedAxisCoalescing)
                   << " ("
                   << cfg::getGraphOptimizationRuleName(
                          cfg::GraphOptimizationRuleId::StridedAxisCoalescing)
                   << ") at " << seeds.front().getLoc() << ": stride=" << S
                   << " tileLen=" << BT << " axis=" << coalesceAxis << " loads="
                   << seeds.size() << " stores=" << sinks.size() << "\n");

  // Erase the original chain (sinks, then region in reverse order, then seeds).
  for (auto st : sinks)
    rw.eraseOp(st);
  for (auto it = ordered.rbegin(); it != ordered.rend(); ++it)
    rw.eraseOp(*it);
  for (auto l : seeds)
    rw.eraseOp(l);

  auto i32t = IntegerType::get(moduleOp.getContext(), 32);
  moduleOp->setAttr("hacc.coalesce_factor", IntegerAttr::get(i32t, S));
  moduleOp->setAttr("hacc.coalesce_axis", IntegerAttr::get(i32t, coalesceAxis));
}

} // namespace StridedAxisCoalescing
