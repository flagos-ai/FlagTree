#include "triton/Analysis/TileAnalysis.h"
#include "triton/Analysis/NewAnalysis/Utility.h"
// For getVectorWidth / vectorizedTyValid: the gates below ask how wide a vector
// of a given element type is, which is the state half's answer. The dependency
// is one-way on purpose -- the state unit must not come to depend on this one.
#include "triton/Analysis/VectorizabilityAnalysis.h"

#include "triton/Tools/Sys/GetEnv.hpp"

namespace mlir {
namespace triton {
namespace xpu {

ClusterLayoutAttr getClusterLayout(RankedTensorType tensorTy) {
  if (auto sliceEncoding =
          dyn_cast<triton::gpu::SliceEncodingAttr>(tensorTy.getEncoding()))
    return dyn_cast<ClusterLayoutAttr>(sliceEncoding.getParent());
  return dyn_cast<ClusterLayoutAttr>(tensorTy.getEncoding());
}

int64_t getNumRegs(Type type, bool *isVector) {
  if (isVector)
    *isVector = false;
  auto tensorTy = dyn_cast<RankedTensorType>(type);
  if (!tensorTy)
    return 0;
  auto clusterEncoding = getClusterLayout(tensorTy);
  if (!clusterEncoding)
    return 0;
  int64_t units = 1;
  for (auto sizePerCore : clusterEncoding.getSizePerCore())
    units *= sizePerCore;
  units = std::max<int64_t>(units, 1);
  auto elemTy = getElementTypeOrSelf(tensorTy);
  if (isa<VectorType>(elemTy)) {
    if (isVector)
      *isVector = true;
    return units;
  }
  // A tensor of pointers is an address computation, not a live data value:
  // gm2lm/alloca own that footprint and it does not scale with the tile.
  if (!elemTy.isIntOrFloat())
    return 0;
  // sizePerCore counts elements here and a scalar register holds one of them.
  return units;
}

// Last line at which `val` is still live, in `getOpLine`'s numbering.
//
// Walking `op->getOperands()` only ever reports the direct user, and for a
// value defined outside a loop body that user is an op *inside* the body: from
// that line on the value looks dead, while the implicit backedge keeps it live
// for every remaining iteration. MLIR has no explicit backedge to propagate
// along -- an `scf.for` body block has no successor edge back to itself -- so
// `mlir::Liveness` does not report such a value as live-out of the body either.
// What does carry the information is the ancestor op that is a sibling of the
// definition: `getOpLine` numbers post-order, so that ancestor's line is larger
// than any line nested inside it, and ending the range there ends it after the
// loop instead of in the middle of it.
//
// This is the mechanism upstream uses for shared memory buffers, where the
// comment at lib/Analysis/Allocation.cpp:335 names the same failure ("%5
// liveness range ends before the child operation's liveness range ends") and
// the fix is likewise post-order ids plus the parent op standing in for its
// region. The range is module-wide and therefore conservative inside a nested
// region: a value read on one branch of an `scf.if` is charged until the whole
// `scf.if` retires. That direction is the safe one for a hard budget.
// `std::nullopt` when the value has no use to end the range at. Line 0 is a
// real op, so reporting 0 there would kill the value at the first replayed op
// instead
// -- the opposite of the caller's convention, which keeps an unused value live
// to the end (and does so for the direct-use walk by leaving it out of the
// map).
static std::optional<unsigned>
liveEndLine(Value val, const DenseMap<Operation *, unsigned> &op2Line) {
  Block *defBlock = val.getParentBlock();
  std::optional<unsigned> end;
  for (auto &use : val.getUses()) {
    Operation *user = use.getOwner();
    if (defBlock)
      if (Operation *ancestor = defBlock->findAncestorOpInBlock(*user))
        user = ancestor;
    if (auto it = op2Line.find(user); it != op2Line.end())
      end = end ? std::max(*end, it->second) : it->second;
  }
  return end;
}

void getRegPressure(ModuleOp m, const llvm::SetVector<Operation *> &opTree,
                    RegPressure &p) {
  DenseMap<Operation *, unsigned> op2Line;
  getOpLine(m, op2Line);

  SmallVector<Operation *> ordered(opTree.begin(), opTree.end());
  llvm::sort(ordered, [&](Operation *lhs, Operation *rhs) {
    return op2Line[lhs] < op2Line[rhs];
  });

  DenseMap<Value, int64_t> vecRegs, scalarRegs;
  DenseMap<Value, unsigned> lastUse, liveEnd;
  auto note = [&](Value val) {
    bool isVector = false;
    int64_t regs = getNumRegs(val.getType(), &isVector);
    if (!regs)
      return false;
    if (!isVector) {
      scalarRegs[val] = regs;
      return true;
    }
    vecRegs[val] = regs;
    if (auto layout = getClusterLayout(cast<RankedTensorType>(val.getType()))) {
      int64_t width = layout.getSizePerCore().back();
      p.maxVecWidth = std::max<int64_t>(p.maxVecWidth, width);
      p.minVecWidth =
          p.minVecWidth ? std::min<int64_t>(p.minVecWidth, width) : width;
    }
    return true;
  };
  for (auto *op : ordered) {
    unsigned line = op2Line[op];
    for (auto res : op->getResults())
      note(res);
    for (auto operand : op->getOperands()) {
      if (!note(operand))
        continue;
      auto it = lastUse.find(operand);
      if (it == lastUse.end() || it->second < line)
        lastUse[operand] = line;
    }
  }
  for (auto *regs : {&vecRegs, &scalarRegs})
    for (auto &[val, n] : *regs)
      if (auto end = liveEndLine(val, op2Line))
        liveEnd[val] = *end;

  // Values defined outside the tree are live on entry. Values with no use
  // inside the tree are conservatively kept live to the end.
  auto peakOf = [&](DenseMap<Value, int64_t> &regs,
                    const DenseMap<Value, unsigned> &ends, int64_t &total,
                    int64_t &invariant) {
    int64_t live = 0;
    total = 0;
    DenseMap<unsigned, int64_t> deathAtLine;
    for (auto &[val, n] : regs) {
      total += n;
      auto *defOp = val.getDefiningOp();
      if (!defOp || !opTree.contains(defOp))
        live += n;
      if (auto it = ends.find(val); it != ends.end())
        deathAtLine[it->second] += n;
    }
    // Exactly the entry-live set: no tile loop makes these smaller.
    invariant = live;
    int64_t peak = live;
    for (auto *op : ordered) {
      for (auto res : op->getResults())
        live += regs.lookup(res);
      peak = std::max(peak, live);
      live -= deathAtLine.lookup(op2Line[op]);
    }
    return peak;
  };
  p.vecPeakUse = peakOf(vecRegs, lastUse, p.vecTotal, p.vecInvariant);
  p.scalarPeakUse =
      peakOf(scalarRegs, lastUse, p.scalarTotal, p.scalarInvariant);
  // Same replay against the liveness range. `total` and `invariant` do not
  // depend on where a value dies, so the second pass recomputes them into
  // scratch and the reported ones stay the ones above.
  int64_t scratchTotal = 0, scratchInvariant = 0;
  p.vecPeakLive = peakOf(vecRegs, liveEnd, scratchTotal, scratchInvariant);
  p.scalarPeakLive =
      peakOf(scalarRegs, liveEnd, scratchTotal, scratchInvariant);
  // The liveness range is what the tile loop actually cannot shrink, so it is
  // the peak the decision runs on. Measured before flipping it (2026-08-14, the
  // five golden probes, 10 sites): `livePeak > peak` at four layernorm sites
  // (32->40, 36->44 twice, 40->48) and equal at the other six, and `target` is
  // unchanged at every one of them -- none of the four crosses a `ceil(peak /
  // 24)` boundary -- so the emitted IR for the whole suite is byte-identical
  // either way. That is what makes this safe to enable without re-calibrating
  // `vrfBudget=24` against the understated peak it was fitted on: on the
  // measured set there is nothing to re-fit. `TRITONXPU_LIVE_RANGE=0` goes back
  // to the direct-use peak, and both numbers stay reported so `[LiveRange]` can
  // still show the gap.
  //
  // Neutral at 24 is not the same as pointless: the understatement is inert
  // only because 24 puts both numbers on the same side of every `ceil`. Move
  // the budget and it bites. `TRITONXPU_VRF_BUDGET=40`, layernorm M=1536
  // N=16384 bufsz=512, same day: the direct-use peak 40 gives `ceil(40/40)=1`,
  // i.e. no reroll, `vspill=14 reload=22 instrs=776`, 877.2 us; the live peak
  // 48 gives `ceil(48/40)=2`, `vspill=0 reload=8 instrs=690`, 834.5 us -- 4.9%
  // faster, reproducible to +-0.1% over three passes. Budget 44 diverges the
  // same way (876.6 -> 822.5 us best, ~6.2%); budget 36 keeps vspill=0 on both
  // sides and differs only in instruction count, 690 -> 675, yet still 840.9 ->
  // 819.8 us. On all four budgets measured the live peak is never the slower
  // one. This correction is what makes `vrfBudget` re-tunable at all.
  bool useLiveRange = true;
  if (auto opt = mlir::triton::tools::isEnvValueBool(
          mlir::triton::tools::getStrEnvXPU("TRITONXPU_LIVE_RANGE")))
    useLiveRange = *opt;
  p.vecPeak = useLiveRange ? p.vecPeakLive : p.vecPeakUse;
  p.scalarPeak = useLiveRange ? p.scalarPeakLive : p.scalarPeakUse;
}

void getBlockRegPressure(ModuleOp m, Operation *insertPt, RegPressure &p) {
  Block *block = insertPt->getBlock();
  if (!block)
    return;
  llvm::SetVector<Operation *> blockOps;
  for (auto &op : *block)
    blockOps.insert(&op);
  getRegPressure(m, blockOps, p);
}

//===----------------------------------------------------------------------===//
// E-dependent gates, moved here from VectorizabilityAnalysis.
//===----------------------------------------------------------------------===//

bool vectorFitsRoot(Type rootOpTy) {
  auto rowsPerCore = 1;
  if (auto rootOpTensorTy = mlir::dyn_cast<RankedTensorType>(rootOpTy)) {
    auto rank = rootOpTensorTy.getShape().size();
    if (rank > 1) {
      rowsPerCore = mlir::cast<triton::xpu::ClusterLayoutAttr>(
                        rootOpTensorTy.getEncoding())
                        .getSizePerCore()[0];
    }
  }

  unsigned numElems = getTotalElemsPerThread(rootOpTy) / rowsPerCore;
  Type vecTy = getElementTypeOrSelf(rootOpTy);
  Type elemTy = getElementTypeOrSelf(vecTy);
  auto vectorWidth = getVectorWidth(elemTy);
  return numElems >= vectorWidth && numElems % vectorWidth == 0 &&
         vectorizedTyValid(elemTy);
}

bool vectorFitsReduceOperand(triton::xpu::ReduceOp redOp, Type operandTy) {
  unsigned numElems = 0;
  auto axis = redOp.getAxis();

  if (auto operandTensorTy = dyn_cast<RankedTensorType>(operandTy)) {
    auto operandShape = operandTensorTy.getShape();
    numElems = operandShape[axis];
  }

  Type vecTy = getElementTypeOrSelf(operandTy);
  Type elemTy = getElementTypeOrSelf(vecTy);
  auto elemWidth = elemTy.getIntOrFloatBitWidth();
  auto vectorWidth = 512 / elemWidth;

  if (numElems < vectorWidth || numElems % vectorWidth > 0 ||
      !vectorizedTyValid(elemTy))
    return false;

  return true;
}

Fit vectorFitsValue(Value value, FitQuery query, unsigned wantWidth) {
  unsigned numElems = getTotalElemsPerThread(value.getType());

  switch (query) {
  case FitQuery::WholeVectors:
    // `wantWidth == 0` is unreachable today (it is 512 / elemWidth for an
    // int-or-float element type), the guard only keeps the division defined.
    return (wantWidth != 0 && numElems != 0 && numElems % wantWidth == 0)
               ? Fit::Yes
               : Fit::No;
  case FitQuery::SingleElem:
    return numElems == 1 ? Fit::Yes : Fit::No;
  case FitQuery::AtLeastWidth:
    return numElems >= wantWidth ? Fit::Yes : Fit::No;
  }
  llvm_unreachable("unhandled FitQuery");
}

//===----------------------------------------------------------------------===//
// M6 -- the tile plan carrier.
//===----------------------------------------------------------------------===//

namespace {

// Kernel plus printed location. The kernel name is part of the key because the
// probe suite compiles the same kernel twice at different shapes (softmax /
// shortrow), so `loc` alone is not even unique across a run.
std::string locKeyOf(Operation *op) {
  std::string key;
  llvm::raw_string_ostream os(key);
  if (auto funcOp = op->getParentOfType<triton::FuncOp>())
    os << funcOp.getName() << "|";
  else
    os << "<unknown>|";
  op->getLoc().print(os);
  return key;
}

// Where the two carriers below meet: both key a root the same way, so the
// resolution is written once. Building the index also strips the id key, which
// is why it runs even when there is no array to resolve.
struct KeyIndex {
  llvm::DenseMap<int64_t, SmallVector<Operation *>> byId;
  llvm::StringMap<SmallVector<Operation *>> byLoc;
};

KeyIndex indexAndStripKeys(ModuleOp mod, StringRef idAttrName) {
  KeyIndex index;
  mod.walk([&](Operation *op) {
    if (auto idAttr = op->getAttrOfType<IntegerAttr>(idAttrName)) {
      index.byId[idAttr.getInt()].push_back(op);
      op->removeAttr(idAttrName);
    }
    index.byLoc[locKeyOf(op)].push_back(op);
  });
  return index;
}

struct KeyHits {
  size_t byId = 0;
  size_t byLoc = 0;
  size_t byLocKind = 0;
  // What the surviving op turned into, when the id found it: an op that was
  // rebuilt under a different name is the interesting failure mode for `loc`.
  StringRef nowName = "-";
};

KeyHits resolveEntry(const KeyIndex &index, int64_t id, StringRef locKey,
                     StringRef rootName) {
  KeyHits hits;
  auto idIt = index.byId.find(id);
  if (idIt != index.byId.end()) {
    hits.byId = idIt->second.size();
    if (hits.byId == 1)
      hits.nowName = idIt->second.front()->getName().getStringRef();
  }
  auto locIt = index.byLoc.find(locKey);
  if (locIt != index.byLoc.end()) {
    hits.byLoc = locIt->second.size();
    // The same key narrowed by op kind, which a consumer legitimately knows: it
    // enumerates stores, or reduces, not arbitrary ops. Reported so that "loc
    // is ambiguous" cannot be answered with "then also match the op name" --
    // the number for that variant is right here.
    for (Operation *op : locIt->second)
      if (op->getName().getStringRef() == rootName)
        ++hits.byLocKind;
  }
  return hits;
}

// Append one entry to `planAttrName` and stamp the root with its index. The
// caller supplies the payload; id / loc / root are the part every carrier needs
// to be resolvable again.
int64_t recordEntry(ModuleOp mod, Operation *root, StringRef planAttrName,
                    StringRef idAttrName, ArrayRef<NamedAttribute> payload) {
  MLIRContext *ctx = mod.getContext();
  SmallVector<Attribute> entries;
  if (auto existing = mod->getAttrOfType<ArrayAttr>(planAttrName))
    entries.assign(existing.begin(), existing.end());

  // The id is the entry's own index, so the two keys stay independent: nothing
  // on the consumer side needs the array to be searched by id to find it.
  int64_t id = entries.size();
  auto i64Ty = IntegerType::get(ctx, 64);
  SmallVector<NamedAttribute> fields(payload.begin(), payload.end());
  fields.push_back(
      NamedAttribute(StringAttr::get(ctx, "id"), IntegerAttr::get(i64Ty, id)));
  fields.push_back(NamedAttribute(StringAttr::get(ctx, "loc"),
                                  StringAttr::get(ctx, locKeyOf(root))));
  fields.push_back(
      NamedAttribute(StringAttr::get(ctx, "root"),
                     StringAttr::get(ctx, root->getName().getStringRef())));
  entries.push_back(DictionaryAttr::get(ctx, fields));

  mod->setAttr(planAttrName, ArrayAttr::get(ctx, entries));
  root->setAttr(idAttrName, IntegerAttr::get(i64Ty, id));
  return id;
}

} // namespace

bool tilePlanProbeEnabled() {
  return mlir::triton::tools::getBoolEnvXPU("TRITONXPU_TILE_PLAN");
}

void tilePlanRecord(ModuleOp mod, Operation *root, StringRef site,
                    bool eligible, int64_t closure) {
  if (!tilePlanProbeEnabled() || !root)
    return;

  MLIRContext *ctx = mod.getContext();
  auto i64Ty = IntegerType::get(ctx, 64);
  auto named = [&](StringRef name, Attribute value) {
    return NamedAttribute(StringAttr::get(ctx, name), value);
  };
  recordEntry(mod, root, kTilePlanAttrName, kTilePlanIdAttrName,
              {named("closure", IntegerAttr::get(i64Ty, closure)),
               named("eligible", BoolAttr::get(ctx, eligible)),
               named("site", StringAttr::get(ctx, site))});
}

void tilePlanCheck(ModuleOp mod) {
  auto plan = mod->getAttrOfType<ArrayAttr>(kTilePlanAttrName);

  // Unconditional, and before any early return: whatever the probe wrote must
  // not reach the emitted IR, or C1 fails for a reason that has nothing to do
  // with the decision being measured.
  mod->removeAttr(kTilePlanAttrName);
  KeyIndex index = indexAndStripKeys(mod, kTilePlanIdAttrName);

  if (!plan)
    return;

  int64_t idHits = 0, locHits = 0, locKindHits = 0;
  for (Attribute entry : plan) {
    auto dict = dyn_cast<DictionaryAttr>(entry);
    if (!dict)
      continue;
    int64_t id = cast<IntegerAttr>(dict.get("id")).getInt();
    StringRef locKey = cast<StringAttr>(dict.get("loc")).getValue();
    StringRef site = cast<StringAttr>(dict.get("site")).getValue();
    StringRef rootName = cast<StringAttr>(dict.get("root")).getValue();

    KeyHits hits = resolveEntry(index, id, locKey, rootName);

    // A hit is exactly one live op. Zero means the key did not survive; more
    // than one means it does not identify a root, which is just as unusable --
    // counting either as a hit is the fallback masking the exit gate forbids.
    if (hits.byId == 1)
      ++idHits;
    if (hits.byLoc == 1)
      ++locHits;
    if (hits.byLocKind == 1)
      ++locKindHits;

    llvm::errs() << "[TilePlan] id=" << id << " site=" << site
                 << " root=" << rootName << " now=" << hits.nowName
                 << " byId=" << hits.byId << " byLoc=" << hits.byLoc
                 << " byLocKind=" << hits.byLocKind << " key=" << locKey
                 << "\n";
  }
  llvm::errs() << "[TilePlan] summary entries=" << plan.size()
               << " idHits=" << idHits << " locHits=" << locHits
               << " locKindHits=" << locKindHits << "\n";
}

bool tileDecideEnabled() {
  return mlir::triton::tools::getBoolEnvXPU("TRITONXPU_TILE_DECIDE");
}

void tileDecisionRecord(ModuleOp mod, Operation *root, StringRef site,
                        StringRef state, int64_t closure, int64_t term,
                        int64_t matIn, int64_t matOut) {
  if (!tileDecideEnabled() || !root)
    return;

  MLIRContext *ctx = mod.getContext();
  auto i64Ty = IntegerType::get(ctx, 64);
  auto named = [&](StringRef name, Attribute value) {
    return NamedAttribute(StringAttr::get(ctx, name), value);
  };
  recordEntry(mod, root, kTileDecisionAttrName, kTileDecisionIdAttrName,
              {named("closure", IntegerAttr::get(i64Ty, closure)),
               named("matIn", IntegerAttr::get(i64Ty, matIn)),
               named("matOut", IntegerAttr::get(i64Ty, matOut)),
               named("site", StringAttr::get(ctx, site)),
               named("state", StringAttr::get(ctx, state)),
               named("term", IntegerAttr::get(i64Ty, term))});
}

void tileDecisionCheck(ModuleOp mod) {
  auto plan = mod->getAttrOfType<ArrayAttr>(kTileDecisionAttrName);
  mod->removeAttr(kTileDecisionAttrName);
  KeyIndex index = indexAndStripKeys(mod, kTileDecisionIdAttrName);

  if (!plan)
    return;

  int64_t idHits = 0, locHits = 0, locKindHits = 0;
  for (Attribute entry : plan) {
    auto dict = dyn_cast<DictionaryAttr>(entry);
    if (!dict)
      continue;
    int64_t id = cast<IntegerAttr>(dict.get("id")).getInt();
    StringRef locKey = cast<StringAttr>(dict.get("loc")).getValue();
    StringRef site = cast<StringAttr>(dict.get("site")).getValue();
    StringRef rootName = cast<StringAttr>(dict.get("root")).getValue();
    StringRef state = cast<StringAttr>(dict.get("state")).getValue();
    int64_t closure = cast<IntegerAttr>(dict.get("closure")).getInt();
    int64_t term = cast<IntegerAttr>(dict.get("term")).getInt();
    int64_t matIn = cast<IntegerAttr>(dict.get("matIn")).getInt();
    int64_t matOut = cast<IntegerAttr>(dict.get("matOut")).getInt();

    KeyHits hits = resolveEntry(index, id, locKey, rootName);
    if (hits.byId == 1)
      ++idHits;
    if (hits.byLoc == 1)
      ++locHits;
    if (hits.byLocKind == 1)
      ++locKindHits;

    llvm::errs() << "[TileDecision] id=" << id << " site=" << site
                 << " root=" << rootName << " state=" << state
                 << " closure=" << closure << " term=" << term
                 << " mat{in=" << matIn << ",out=" << matOut << "}"
                 << " now=" << hits.nowName << " byId=" << hits.byId
                 << " byLoc=" << hits.byLoc << " byLocKind=" << hits.byLocKind
                 << " key=" << locKey << "\n";
  }
  llvm::errs() << "[TileDecision] summary entries=" << plan.size()
               << " idHits=" << idHits << " locHits=" << locHits
               << " locKindHits=" << locKindHits << "\n";
}

} // namespace xpu
} // namespace triton
} // namespace mlir
