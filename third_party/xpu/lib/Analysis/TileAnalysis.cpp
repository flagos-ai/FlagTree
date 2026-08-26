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

void getRegPressure(ModuleOp m, const llvm::SetVector<Operation *> &opTree,
                    RegPressure &p) {
  DenseMap<Operation *, unsigned> op2Line;
  getOpLine(m, op2Line);

  SmallVector<Operation *> ordered(opTree.begin(), opTree.end());
  llvm::sort(ordered, [&](Operation *lhs, Operation *rhs) {
    return op2Line[lhs] < op2Line[rhs];
  });

  DenseMap<Value, int64_t> vecRegs, scalarRegs;
  DenseMap<Value, unsigned> lastUse;
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

  // Values defined outside the tree are live on entry. Values with no use
  // inside the tree are conservatively kept live to the end.
  auto peakOf = [&](DenseMap<Value, int64_t> &regs, int64_t &total) {
    int64_t live = 0;
    total = 0;
    DenseMap<unsigned, int64_t> deathAtLine;
    for (auto &[val, n] : regs) {
      total += n;
      auto *defOp = val.getDefiningOp();
      if (!defOp || !opTree.contains(defOp))
        live += n;
      if (auto it = lastUse.find(val); it != lastUse.end())
        deathAtLine[it->second] += n;
    }
    int64_t peak = live;
    for (auto *op : ordered) {
      for (auto res : op->getResults())
        live += regs.lookup(res);
      peak = std::max(peak, live);
      live -= deathAtLine.lookup(op2Line[op]);
    }
    return peak;
  };
  p.vecPeak = peakOf(vecRegs, p.vecTotal);
  p.scalarPeak = peakOf(scalarRegs, p.scalarTotal);
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

} // namespace

bool tilePlanProbeEnabled() {
  return mlir::triton::tools::getBoolEnv("TRITONXPU_TILE_PLAN");
}

void tilePlanRecord(ModuleOp mod, Operation *root, StringRef site,
                    bool eligible, int64_t closure) {
  if (!tilePlanProbeEnabled() || !root)
    return;

  MLIRContext *ctx = mod.getContext();
  SmallVector<Attribute> entries;
  if (auto existing = mod->getAttrOfType<ArrayAttr>(kTilePlanAttrName))
    entries.assign(existing.begin(), existing.end());

  // The id is the entry's own index, so the two keys stay independent: nothing
  // on the consumer side needs the array to be searched by id to find it.
  int64_t id = entries.size();
  auto i64Ty = IntegerType::get(ctx, 64);
  auto named = [&](StringRef name, Attribute value) {
    return NamedAttribute(StringAttr::get(ctx, name), value);
  };
  entries.push_back(DictionaryAttr::get(
      ctx, {named("closure", IntegerAttr::get(i64Ty, closure)),
            named("eligible", BoolAttr::get(ctx, eligible)),
            named("id", IntegerAttr::get(i64Ty, id)),
            named("loc", StringAttr::get(ctx, locKeyOf(root))),
            named("root", StringAttr::get(ctx, root->getName().getStringRef())),
            named("site", StringAttr::get(ctx, site))}));

  mod->setAttr(kTilePlanAttrName, ArrayAttr::get(ctx, entries));
  root->setAttr(kTilePlanIdAttrName, IntegerAttr::get(i64Ty, id));
}

void tilePlanCheck(ModuleOp mod) {
  auto plan = mod->getAttrOfType<ArrayAttr>(kTilePlanAttrName);

  // Unconditional, and before any early return: whatever the probe wrote must
  // not reach the emitted IR, or C1 fails for a reason that has nothing to do
  // with the decision being measured.
  mod->removeAttr(kTilePlanAttrName);
  llvm::DenseMap<int64_t, SmallVector<Operation *>> byId;
  llvm::StringMap<SmallVector<Operation *>> byLoc;
  mod.walk([&](Operation *op) {
    if (auto idAttr = op->getAttrOfType<IntegerAttr>(kTilePlanIdAttrName)) {
      byId[idAttr.getInt()].push_back(op);
      op->removeAttr(kTilePlanIdAttrName);
    }
    byLoc[locKeyOf(op)].push_back(op);
  });

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

    auto idIt = byId.find(id);
    size_t nById = idIt == byId.end() ? 0 : idIt->second.size();
    auto locIt = byLoc.find(locKey);
    size_t nByLoc = locIt == byLoc.end() ? 0 : locIt->second.size();

    // The same key narrowed by op kind, which a consumer legitimately knows: it
    // enumerates stores, or reduces, not arbitrary ops. Reported so that "loc
    // is ambiguous" cannot be answered with "then also match the op name" --
    // the number for that variant is right here.
    size_t nByLocKind = 0;
    if (locIt != byLoc.end())
      for (Operation *op : locIt->second)
        if (op->getName().getStringRef() == rootName)
          ++nByLocKind;

    // A hit is exactly one live op. Zero means the key did not survive; more
    // than one means it does not identify a root, which is just as unusable --
    // counting either as a hit is the fallback masking the exit gate forbids.
    if (nById == 1)
      ++idHits;
    if (nByLoc == 1)
      ++locHits;
    if (nByLocKind == 1)
      ++locKindHits;

    // What the surviving op turned into, when the id found it: an op that was
    // rebuilt under a different name is the interesting failure mode for `loc`.
    StringRef nowName = nById == 1
                            ? idIt->second.front()->getName().getStringRef()
                            : StringRef("-");
    llvm::errs() << "[TilePlan] id=" << id << " site=" << site
                 << " root=" << rootName << " now=" << nowName
                 << " byId=" << nById << " byLoc=" << nByLoc
                 << " byLocKind=" << nByLocKind << " key=" << locKey << "\n";
  }
  llvm::errs() << "[TilePlan] summary entries=" << plan.size()
               << " idHits=" << idHits << " locHits=" << locHits
               << " locKindHits=" << locKindHits << "\n";
}

} // namespace xpu
} // namespace triton
} // namespace mlir
