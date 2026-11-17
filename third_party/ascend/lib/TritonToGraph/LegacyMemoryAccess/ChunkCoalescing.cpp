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

#include "TritonToGraph/LegacyMemoryAccess/ChunkCoalescing.h"

#include "TritonToGraph/GraphOptimization.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Matchers.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Debug.h"

#define DEBUG_TYPE "graph-optimize"

#include <algorithm>
#include <functional>
#include <optional>

namespace ChunkCoalescing {

using namespace mlir;
using namespace triton;

namespace {

constexpr int64_t kMinContigBytes = 512;
constexpr int64_t kUBBytesBudget = 128 * 1024;
constexpr int64_t kMaxCoalesceTilesCeil = 16;

constexpr llvm::StringLiteral kCoalesceFactorAttr = "hacc.coalesce_factor";
constexpr llvm::StringLiteral kCoalesceAxisAttr = "hacc.coalesce_axis";
constexpr llvm::StringLiteral kGridNumTilesAttr = "hacc.grid_num_tiles";

struct TileSeed {
  triton::GetProgramIdOp pid;
  int32_t axis = 0;
  int64_t tileLen = 0;
  int64_t bound = 0;
  Value mask;
};

static bool getConstInt(Value v, int64_t &out) {
  APInt ap;
  if (matchPattern(v, m_ConstantInt(&ap))) {
    out = ap.getSExtValue();
    return true;
  }
  DenseElementsAttr dea;
  if (matchPattern(v, m_Constant(&dea)) && dea.isSplat() &&
      isa<IntegerType>(dea.getElementType())) {
    out = dea.getSplatValue<APInt>().getSExtValue();
    return true;
  }
  return false;
}

// tt.assert is liftable: a group covers exactly H tiles of the original grid,
// so a widened condition traps on the same inputs, only grouped differently.
static bool isLiftable(Operation *op) {
  if (auto *d = op->getDialect()) {
    StringRef ns = d->getNamespace();
    if (ns == arith::ArithDialect::getDialectNamespace() ||
        ns == math::MathDialect::getDialectNamespace())
      return true;
  }
  return isa<triton::SplatOp, triton::AddPtrOp, triton::BroadcastOp,
             triton::ExpandDimsOp, triton::LoadOp, triton::StoreOp,
             triton::ScanOp, triton::ReduceOp, triton::AssertOp>(op);
}

// A pid-dependent predicate that can only trap does not constrain the rewrite:
// tt.assert has no results, so nothing reaches an offset, a mask or a select.
// The permitted sink is whitelisted rather than the forbidden ones blacklisted,
// because the rewrite deletes the seed mask outright and so must fail closed.
// sanitize_overflow is what makes this matter: it is on by default and
// instruments nearly every pid-derived index.
static bool onlyFeedsDeviceAssert(Value predicate) {
  SmallVector<Value> wl{predicate};
  DenseSet<Value> visited{predicate};
  while (!wl.empty()) {
    Value cur = wl.pop_back_val();
    for (Operation *u : cur.getUsers()) {
      if (isa<triton::AssertOp>(u))
        continue;
      // Only the instrumentation's own boolean combines may extend the chain.
      if (!isa<arith::AndIOp, arith::OrIOp, arith::XOrIOp>(u))
        return false;
      for (Value r : u->getResults())
        if (visited.insert(r).second)
          wl.push_back(r);
    }
  }
  return true;
}

static std::optional<TileSeed> findSeed(ModuleOp moduleOp) {
  int32_t maxAxis = -1;
  moduleOp.walk([&](triton::GetProgramIdOp pid) {
    maxAxis = std::max<int32_t>(maxAxis, pid.getAxisAsInt());
  });

  int32_t maxAxisPids = 0;
  moduleOp.walk([&](triton::GetProgramIdOp pid) {
    if (pid.getAxisAsInt() == maxAxis)
      ++maxAxisPids;
  });
  if (maxAxisPids != 1)
    return std::nullopt;

  bool readsMaxAxisNumPrograms = false;
  moduleOp.walk([&](triton::GetNumProgramsOp np) {
    if (np.getAxisAsInt() == maxAxis)
      readsMaxAxisNumPrograms = true;
  });
  if (readsMaxAxisNumPrograms)
    return std::nullopt;

  std::optional<TileSeed> result;
  moduleOp.walk([&](triton::GetProgramIdOp pid) {
    if (result || pid.getAxisAsInt() != maxAxis)
      return;
    for (Operation *u : pid.getResult().getUsers()) {
      auto mul = dyn_cast<arith::MulIOp>(u);
      if (!mul)
        continue;
      Value other =
          (mul.getLhs() == pid.getResult()) ? mul.getRhs() : mul.getLhs();
      int64_t T = 0;
      if (!getConstInt(other, T) || T <= 1)
        continue;

      for (Operation *mu : mul.getResult().getUsers()) {
        auto sp = dyn_cast<triton::SplatOp>(mu);
        if (!sp)
          continue;
        for (Operation *su : sp.getResult().getUsers()) {
          auto add = dyn_cast<arith::AddIOp>(su);
          if (!add)
            continue;
          Value rangeV =
              (add.getLhs() == sp.getResult()) ? add.getRhs() : add.getLhs();
          auto range = rangeV.getDefiningOp<triton::MakeRangeOp>();
          if (!range || range.getStart() != 0 || range.getEnd() != T)
            continue;

          int64_t bound = 0;
          Value mask;
          bool unsafe = false;
          DenseSet<Value> taint;
          SmallVector<Value> twl;
          taint.insert(pid.getResult());
          twl.push_back(pid.getResult());
          while (!twl.empty() && !unsafe) {
            Value cur = twl.pop_back_val();
            for (Operation *tu : cur.getUsers()) {
              if (auto cmp = dyn_cast<arith::CmpIOp>(tu)) {
                if (mask && cmp.getResult() == mask)
                  continue;
                int64_t b = 0;
                if (!mask && cur == add.getResult() && cmp.getLhs() == cur &&
                    cmp.getPredicate() == arith::CmpIPredicate::slt &&
                    getConstInt(cmp.getRhs(), b) && b >= T && b % T == 0) {
                  bound = b;
                  mask = cmp.getResult();
                  continue;
                }
                if (onlyFeedsDeviceAssert(cmp.getResult()))
                  continue;
                unsafe = true;
                break;
              }
              if (isa<arith::MinSIOp, arith::MaxSIOp, arith::MinUIOp,
                      arith::MaxUIOp, arith::RemSIOp, arith::RemUIOp,
                      arith::DivSIOp, arith::DivUIOp, arith::CeilDivSIOp,
                      arith::FloorDivSIOp>(tu)) {
                unsafe = true;
                break;
              }
              bool propagates = isa<triton::SplatOp, triton::ExpandDimsOp,
                                    triton::BroadcastOp, triton::AddPtrOp>(tu);
              if (auto *d = tu->getDialect())
                propagates |= d->getNamespace() ==
                              arith::ArithDialect::getDialectNamespace();
              if (!propagates)
                continue;
              for (Value r : tu->getResults())
                if (taint.insert(r).second)
                  twl.push_back(r);
            }
          }
          if (unsafe)
            return;
          if (!mask) {
            auto hintAttr =
                moduleOp->getAttrOfType<IntegerAttr>(kGridNumTilesAttr);
            if (!hintAttr)
              return;
            int64_t numTiles = hintAttr.getInt();
            if (numTiles <= 0)
              return;
            bound = numTiles * T;
            result = TileSeed{pid, maxAxis, T, bound, /*mask=*/Value()};
            return;
          }

          result = TileSeed{pid, maxAxis, T, bound, mask};
          return;
        }
      }
    }
  });
  return result;
}

static bool collectRegion(TileSeed &seed, ModuleOp moduleOp,
                          DenseSet<Operation *> &region,
                          SmallVectorImpl<Operation *> &ordered) {
  SmallVector<Operation *> wl;
  DenseSet<Operation *> visited;
  bool hasStore = false;
  for (Operation *u : seed.pid.getResult().getUsers())
    wl.push_back(u);
  while (!wl.empty()) {
    Operation *op = wl.pop_back_val();
    if (!visited.insert(op).second)
      continue;
    if (!isLiftable(op))
      return false;
    if (isa<triton::StoreOp>(op))
      hasStore = true;
    region.insert(op);
    for (Value r : op->getResults())
      for (Operation *u : r.getUsers())
        wl.push_back(u);
  }
  if (!hasStore)
    return false;

  for (Operation *op : region) {
    if (isa<triton::StoreOp>(op))
      continue;
    for (Value r : op->getResults())
      for (Operation *u : r.getUsers())
        if (!region.count(u))
          return false;
  }

  moduleOp.walk([&](Operation *op) {
    if (region.count(op))
      ordered.push_back(op);
  });
  return true;
}

static int64_t chooseH(int64_t numTiles, int64_t tileLen, int64_t elemBytes,
                       int64_t maxH) {
  int64_t blockBytes = tileLen * elemBytes;
  int64_t hMin = (kMinContigBytes + blockBytes - 1) / blockBytes;
  if (hMin < 2)
    hMin = 2;
  if (maxH < hMin)
    return 0;
  if (numTiles <= 0)
    return 0;

  for (int64_t c = maxH; c >= hMin; --c)
    if (numTiles % c == 0)
      return c;
  return 0;
}

static void rewriteModule(ModuleOp moduleOp, IRRewriter &rw) {
  if (moduleOp->hasAttr(kCoalesceFactorAttr))
    return;

  auto seed = findSeed(moduleOp);
  if (!seed)
    return;

  DenseSet<Operation *> region;
  SmallVector<Operation *> ordered;
  if (!collectRegion(*seed, moduleOp, region, ordered))
    return;

  int64_t elemBytes = 0;
  for (Operation *op : ordered)
    if (auto ld = dyn_cast<triton::LoadOp>(op)) {
      auto rt = dyn_cast<RankedTensorType>(ld.getResult().getType());
      if (rt)
        elemBytes = rt.getElementTypeBitWidth() / 8;
      break;
    }
  if (elemBytes == 0)
    return;

  auto tensorBytes = [](Type t) -> int64_t {
    auto rt = dyn_cast<RankedTensorType>(t);
    if (!rt)
      return 0;
    Type et = rt.getElementType();
    if (!et.isIntOrFloat())
      return 0;
    return rt.getNumElements() * ((et.getIntOrFloatBitWidth() + 7) / 8);
  };
  int64_t footprintUnit = 0;
  for (Operation *op : ordered) {
    if (auto ld = dyn_cast<triton::LoadOp>(op))
      footprintUnit += tensorBytes(ld.getResult().getType());
    else if (auto st = dyn_cast<triton::StoreOp>(op))
      footprintUnit += tensorBytes(st.getValue().getType());
    else
      for (Type t : op->getResultTypes())
        if (auto rt = dyn_cast<RankedTensorType>(t))
          if (isa<FloatType>(rt.getElementType()))
            footprintUnit += tensorBytes(t);
  }
  int64_t maxH = kMaxCoalesceTilesCeil;
  if (footprintUnit > 0)
    maxH = std::min<int64_t>(maxH, kUBBytesBudget / footprintUnit);
  if (maxH < 2)
    return; // even H=2 would overflow UB

  int64_t numTiles = (seed->bound + seed->tileLen - 1) / seed->tileLen;
  int64_t H = chooseH(numTiles, seed->tileLen, elemBytes, maxH);
  if (H <= 1)
    return;

  Value pidVal = seed->pid.getResult();
  Location ploc = seed->pid.getLoc();
  Value seedMask = seed->mask;
  Operation *seedMaskOp = seedMask ? seedMask.getDefiningOp() : nullptr;

  if (seedMask) {
    for (Operation *u : seedMask.getUsers()) {
      auto ld = dyn_cast<triton::LoadOp>(u);
      auto st = dyn_cast<triton::StoreOp>(u);
      bool okAsLoadMask = ld && ld.getMask() == seedMask;
      bool okAsStoreMask = st && st.getMask() == seedMask;
      if (!okAsLoadMask && !okAsStoreMask)
        return;
    }
  }

  Block *pidBlock = seed->pid->getBlock();
  for (Operation *op : ordered) {
    if (op->getBlock() != pidBlock)
      return;
    if (!isa<triton::LoadOp, triton::StoreOp>(op) && op->getNumOperands() == 0)
      return;
    if (auto sel = dyn_cast<arith::SelectOp>(op)) {
      bool condTensor = isa<RankedTensorType>(sel.getCondition().getType());
      bool valTensor = isa<RankedTensorType>(sel.getTrueValue().getType());
      if (condTensor != valTensor)
        return;
    }
    if (auto ld = dyn_cast<triton::LoadOp>(op))
      if (!ld.getBoundaryCheck().empty())
        return;
    if (auto st = dyn_cast<triton::StoreOp>(op))
      if (!st.getBoundaryCheck().empty())
        return;
  }

  auto liftTy = [&](Type t) -> RankedTensorType {
    if (auto rt = dyn_cast<RankedTensorType>(t)) {
      SmallVector<int64_t> s;
      s.push_back(H);
      s.append(rt.getShape().begin(), rt.getShape().end());
      return RankedTensorType::get(s, rt.getElementType());
    }
    return RankedTensorType::get({H}, t);
  };

  LLVM_DEBUG(llvm::dbgs() << "[" DEBUG_TYPE "] matched graph optimization rule "
                          << static_cast<unsigned>(
                                 cfg::GraphOptimizationRuleId::ChunkCoalescing)
                          << " ("
                          << cfg::getGraphOptimizationRuleName(
                                 cfg::GraphOptimizationRuleId::ChunkCoalescing)
                          << ") at " << ploc << ": tileLen=" << seed->tileLen
                          << " bound=" << seed->bound
                          << " numTiles=" << numTiles << " maxH=" << maxH
                          << " H=" << H << "\n");

  rw.setInsertionPointAfter(seed->pid);
  Value cH = rw.create<arith::ConstantIntOp>(ploc, H, 32);
  Value pidH = rw.create<arith::MulIOp>(ploc, pidVal, cH);
  auto i32 = rw.getI32Type();
  auto hVecTy = RankedTensorType::get({H}, i32);
  Value rangeH = rw.create<triton::MakeRangeOp>(ploc, hVecTy, 0, H);
  Value splatPidH = rw.create<triton::SplatOp>(ploc, hVecTy, pidH);
  Value tileVec = rw.create<arith::AddIOp>(ploc, splatPidH, rangeH);

  DenseMap<Value, Value> vmap;
  std::function<Value(Value)> lift = [&](Value v) -> Value {
    if (v == pidVal)
      return tileVec;
    auto it = vmap.find(v);
    if (it != vmap.end())
      return it->second;
    Operation *def = v.getDefiningOp();
    if (def && region.count(def)) {
      auto rebuilt = vmap.find(v);
      assert(rebuilt != vmap.end() && "region value used before its rebuild");
      return rebuilt->second;
    }
    if (!isa<RankedTensorType>(v.getType()))
      return v;
    Value e = rw.create<triton::ExpandDimsOp>(v.getLoc(), v, 0);
    Value b =
        rw.create<triton::BroadcastOp>(v.getLoc(), liftTy(v.getType()), e);
    vmap[v] = b;
    return b;
  };
  auto liftOpd = [&](Value v) -> Value {
    Value lv = lift(v);
    if (!isa<RankedTensorType>(lv.getType()))
      lv = rw.create<triton::SplatOp>(lv.getLoc(), liftTy(v.getType()), lv);
    return lv;
  };

  auto liftOpdOrNull = [&](Value v) -> Value {
    return v ? liftOpd(v) : Value();
  };
  auto bumpBoundary = [&](ArrayRef<int32_t> bc) {
    return llvm::map_to_vector(bc, [](int32_t i) { return i + 1; });
  };
  auto copyAttrs = [&](Operation *from, Operation *to) {
    for (NamedAttribute a : from->getAttrs())
      if (!to->hasAttr(a.getName()))
        to->setAttr(a.getName(), a.getValue());
  };

  for (Operation *op : ordered) {
    if (op == seedMaskOp)
      continue;

    rw.setInsertionPoint(op);
    Location loc = op->getLoc();

    if (auto sp = dyn_cast<triton::SplatOp>(op)) {
      Value lin = lift(sp.getSrc());
      if (!isa<RankedTensorType>(lin.getType())) {
        vmap[sp.getResult()] =
            rw.create<triton::SplatOp>(loc, liftTy(sp.getType()), lin);
      } else {
        int addDims = cast<RankedTensorType>(sp.getType()).getRank();
        Value cur = lin;
        for (int k = 0; k < addDims; ++k)
          cur = rw.create<triton::ExpandDimsOp>(
              loc, cur, cast<RankedTensorType>(cur.getType()).getRank());
        vmap[sp.getResult()] =
            rw.create<triton::BroadcastOp>(loc, liftTy(sp.getType()), cur);
      }
      continue;
    }
    if (auto ed = dyn_cast<triton::ExpandDimsOp>(op)) {
      vmap[ed.getResult()] = rw.create<triton::ExpandDimsOp>(
          loc, liftOpd(ed.getSrc()), ed.getAxis() + 1);
      continue;
    }
    if (auto bc = dyn_cast<triton::BroadcastOp>(op)) {
      vmap[bc.getResult()] = rw.create<triton::BroadcastOp>(
          loc, liftTy(bc.getType()), liftOpd(bc.getSrc()));
      continue;
    }
    if (auto scan = dyn_cast<triton::ScanOp>(op)) {
      SmallVector<Value> srcs = llvm::map_to_vector(
          scan.getSrcs(), [&](Value s) { return liftOpd(s); });
      auto nu = rw.create<triton::ScanOp>(loc, srcs, scan.getAxis() + 1,
                                          scan.getReverse());
      rw.cloneRegionBefore(scan.getCombineOp(), nu.getCombineOp(),
                           nu.getCombineOp().end());
      copyAttrs(scan, nu);
      for (auto [o, n] : llvm::zip(scan.getResults(), nu.getResults()))
        vmap[o] = n;
      continue;
    }
    if (auto red = dyn_cast<triton::ReduceOp>(op)) {
      SmallVector<Value> srcs = llvm::map_to_vector(
          red.getSrcs(), [&](Value s) { return liftOpd(s); });
      auto nu = rw.create<triton::ReduceOp>(loc, srcs, red.getAxis() + 1);
      rw.cloneRegionBefore(red.getCombineOp(), nu.getCombineOp(),
                           nu.getCombineOp().end());
      copyAttrs(red, nu);
      for (auto [o, n] : llvm::zip(red.getResults(), nu.getResults()))
        vmap[o] = n;
      continue;
    }
    if (auto ld = dyn_cast<triton::LoadOp>(op)) {
      bool dropMask = ld.getMask() == seedMask;
      Value m = dropMask ? Value() : liftOpdOrNull(ld.getMask());
      Value o = dropMask ? Value() : liftOpdOrNull(ld.getOther());
      auto bc = bumpBoundary(ld.getBoundaryCheck());
      auto nu = rw.create<triton::LoadOp>(loc, liftOpd(ld.getPtr()), m, o, bc,
                                          ld.getPadding(), ld.getCache(),
                                          ld.getEvict(), ld.getIsVolatile());
      copyAttrs(ld, nu);
      vmap[ld.getResult()] = nu.getResult();
      continue;
    }
    if (auto st = dyn_cast<triton::StoreOp>(op)) {
      bool dropMask = st.getMask() == seedMask;
      Value m = dropMask ? Value() : liftOpdOrNull(st.getMask());
      auto bc = bumpBoundary(st.getBoundaryCheck());
      rw.create<triton::StoreOp>(loc, liftOpd(st.getPtr()),
                                 liftOpd(st.getValue()), m, bc, st.getCache(),
                                 st.getEvict());
      continue;
    }

    SmallVector<Value> operands = llvm::map_to_vector(
        op->getOperands(), [&](Value o) { return liftOpd(o); });
    SmallVector<Type> resTypes = llvm::map_to_vector(
        op->getResultTypes(), [&](Type t) -> Type { return liftTy(t); });
    Operation *nu = rw.create(loc, op->getName().getIdentifier(), operands,
                              resTypes, op->getAttrs());
    for (auto [oldR, newR] : llvm::zip(op->getResults(), nu->getResults()))
      vmap[oldR] = newR;
  }

  for (auto it = ordered.rbegin(); it != ordered.rend(); ++it)
    rw.eraseOp(*it);

  auto i32Ty = IntegerType::get(moduleOp.getContext(), 32);
  moduleOp->setAttr(kCoalesceFactorAttr, IntegerAttr::get(i32Ty, H));
  moduleOp->setAttr(kCoalesceAxisAttr, IntegerAttr::get(i32Ty, seed->axis));
}

} // namespace

void rewriteChunkCoalesce(ModuleOp moduleOp) {
  IRRewriter rw(moduleOp.getContext());
  rewriteModule(moduleOp, rw);
  moduleOp->removeAttr(kGridNumTilesAttr);
}

} // namespace ChunkCoalescing
