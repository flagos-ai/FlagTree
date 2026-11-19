/*
 * Copyright 2025-     FlagOS Contributors
 *
 * Permission is hereby granted, free of charge, to any person obtaining
 * a copy of this software and associated documentation files
 * (the "Software"), to deal in the Software without restriction,
 * including without limitation the rights to use, copy, modify, merge,
 * publish, distribute, sublicense, and/or sell copies of the Software,
 * and to permit persons to whom the Software is furnished to do so,
 * subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be
 * included in all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
 * EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
 * MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
 * IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY
 * CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,
 * TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE
 * SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
 */

// flagtree pipeline enhance: merge the layout duplicates of a load before the
// loop is pipelined.
//
// RemoveLayoutConversions treats a load with fewer elements than threads as
// cheap and re-materializes it in every layout it is used with instead of
// converting the layout. The pipeliner then sees several loads: each gets its
// own ring buffer, async copy and wait, and a copy that cannot be made
// asynchronous is read synchronously from global memory every iteration. This
// pass runs after the latencies are assigned (so the assignment of the other
// loads does not change) and keeps one load per group of loads that have
// equivalent pointers, masks, other values and attributes and no global store
// in between; the other copies become layout conversions of the kept one.

#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/Transforms/Passes.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Debug.h"

#define DEBUG_TYPE "tritongpu-merge-duplicate-loads"
#define DBGS() (llvm::dbgs() << "[" DEBUG_TYPE "]: ")
#define LDBG(X) LLVM_DEBUG(DBGS() << X << "\n")

using namespace mlir;
namespace tt = mlir::triton;
namespace ttg = mlir::triton::gpu;

namespace mlir::triton::gpu {

#define GEN_PASS_DEF_TRITONGPUMERGEDUPLICATELOADS
#include "triton/Dialect/TritonGPU/Transforms/Passes.h.inc"

namespace {

static bool sameTypeModuloEncoding(Type a, Type b) {
  if (a == b)
    return true;
  auto ta = dyn_cast<RankedTensorType>(a);
  auto tb = dyn_cast<RankedTensorType>(b);
  if (!ta || !tb)
    return false;
  return ta.getShape() == tb.getShape() &&
         sameTypeModuloEncoding(ta.getElementType(), tb.getElementType());
}

// Attribute dictionaries are equal when every attribute matches; tensor
// constants only have to match in shape, element type and data.
static bool sameAttributesModuloEncoding(DictionaryAttr da, DictionaryAttr db,
                                         StringRef ignored = "") {
  if (da == db)
    return true;
  auto count = [&](DictionaryAttr d) {
    return llvm::count_if(d, [&](NamedAttribute na) {
      return na.getName().getValue() != ignored;
    });
  };
  if (count(da) != count(db))
    return false;
  for (NamedAttribute na : da) {
    if (na.getName().getValue() == ignored)
      continue;
    Attribute vb = db.get(na.getName());
    if (!vb)
      return false;
    if (na.getValue() == vb)
      continue;
    auto ea = dyn_cast<DenseElementsAttr>(na.getValue());
    auto eb = dyn_cast<DenseElementsAttr>(vb);
    if (!ea || !eb)
      return false;
    if (!sameTypeModuloEncoding(ea.getType(), eb.getType()) ||
        ea.isSplat() != eb.isSplat() || ea.getRawData() != eb.getRawData())
      return false;
  }
  return true;
}

// Two values are equivalent modulo layout when they are computed by the same
// chain of pure ops from the same leaves (block arguments and ops with memory
// effects), looking through convert_layout.
class LayoutEquivalence {
public:
  bool equivalent(Value a, Value b) {
    a = stripConvertLayout(a);
    b = stripConvertLayout(b);
    if (a == b)
      return true;
    auto key = std::make_pair(a, b);
    auto it = memo.find(key);
    if (it != memo.end())
      return it->second;
    memo[key] = false; // a cycle cannot be equivalent
    bool result = compute(a, b);
    memo[key] = result;
    return result;
  }

  static Value stripConvertLayout(Value v) {
    while (auto cvt = v.getDefiningOp<ttg::ConvertLayoutOp>())
      v = cvt.getSrc();
    return v;
  }

private:
  bool compute(Value a, Value b) {
    auto ra = dyn_cast<OpResult>(a);
    auto rb = dyn_cast<OpResult>(b);
    if (!ra || !rb)
      return false;
    if (ra.getResultNumber() != rb.getResultNumber())
      return false;
    Operation *oa = ra.getOwner();
    Operation *ob = rb.getOwner();
    if (oa->getName() != ob->getName() || oa->getNumRegions() != 0 ||
        ob->getNumRegions() != 0 ||
        oa->getNumOperands() != ob->getNumOperands() ||
        oa->getNumResults() != ob->getNumResults())
      return false;
    // Ops with memory effects (loads, ...) are leaves: they must be the same.
    if (!isMemoryEffectFree(oa) || !isMemoryEffectFree(ob))
      return false;
    for (auto [ta, tb] : llvm::zip(oa->getResultTypes(), ob->getResultTypes()))
      if (!sameTypeModuloEncoding(ta, tb))
        return false;
    if (!sameAttributesModuloEncoding(oa->getAttrDictionary(),
                                      ob->getAttrDictionary()))
      return false;
    for (auto [x, y] : llvm::zip(oa->getOperands(), ob->getOperands()))
      if (!equivalent(x, y))
        return false;
    return true;
  }

  DenseMap<std::pair<Value, Value>, bool> memo;
};

// Whether the op (or an op nested in it) may write global memory.
static bool mayWriteGlobalMemory(Operation *op) {
  if (op->getNumRegions() != 0) {
    bool nested = false;
    op->walk([&](Operation *inner) {
      if (inner != op && inner->getNumRegions() == 0 &&
          mayWriteGlobalMemory(inner))
        nested = true;
    });
    return nested;
  }
  if (isMemoryEffectFree(op))
    return false;
  auto iface = dyn_cast<MemoryEffectOpInterface>(op);
  if (!iface)
    return true;
  SmallVector<MemoryEffects::EffectInstance> effects;
  iface.getEffects(effects);
  for (auto &effect : effects) {
    if (!isa<MemoryEffects::Write>(effect.getEffect()))
      continue;
    SideEffects::Resource *resource = effect.getResource();
    // Writes to shared memory do not change what a global load returns.
    if (resource == ttg::SharedMemory::get() ||
        resource == tt::SharedMemory::get())
      continue;
    return true;
  }
  return false;
}

static bool hasLatency(Operation *op, tt::TritonDialect *dialect) {
  return dialect->getLatencyAttrHelper().getAttr(op) != nullptr;
}

// Move `op` and the pure ops of the same block it needs that are placed after
// `before` in front of `before`. Returns false (and changes nothing) when an
// op with memory effects or regions would have to move.
static bool moveWithPureSliceBefore(Operation *op, Operation *before) {
  Block *block = before->getBlock();
  SmallVector<Operation *> toMove;
  llvm::SmallDenseSet<Operation *, 16> visited;
  SmallVector<Operation *> worklist = {op};
  while (!worklist.empty()) {
    Operation *cur = worklist.pop_back_val();
    if (!visited.insert(cur).second)
      continue;
    if (cur != op) {
      if (cur->getNumRegions() != 0 || !isMemoryEffectFree(cur))
        return false;
      toMove.push_back(cur);
    }
    for (Value operand : cur->getOperands()) {
      Operation *def = operand.getDefiningOp();
      if (!def || def->getBlock() != block || def->isBeforeInBlock(before))
        continue;
      worklist.push_back(def);
    }
  }
  llvm::sort(toMove,
             [](Operation *a, Operation *b) { return a->isBeforeInBlock(b); });
  for (Operation *moved : toMove)
    moved->moveBefore(before);
  op->moveBefore(before);
  return true;
}

static bool sameLoadModuloLayout(tt::LoadOp a, tt::LoadOp b,
                                 LayoutEquivalence &eq) {
  if (a.getNumOperands() != b.getNumOperands())
    return false;
  if (!sameTypeModuloEncoding(a.getType(), b.getType()))
    return false;
  if (!sameAttributesModuloEncoding(
          a->getAttrDictionary(), b->getAttrDictionary(),
          tt::TritonDialect::getLoaded(a)->getLatencyAttrHelper().getName()))
    return false;
  for (auto [x, y] : llvm::zip(a->getOperands(), b->getOperands()))
    if (!eq.equivalent(x, y))
      return false;
  return true;
}

// Merge one group of duplicate loads of the loop body. Returns true when the
// IR changed.
static bool mergeGroup(ArrayRef<tt::LoadOp> group, bool mergeAny) {
  auto *dialect = tt::TritonDialect::getLoaded(group.front());
  unsigned numPipelined = llvm::count_if(
      group, [&](tt::LoadOp load) { return hasLatency(load, dialect); });
  if (numPipelined == 0 || (!mergeAny && numPipelined != group.size())) {
    LDBG("skip group of " << group.size() << " loads: " << numPipelined
                          << " pipelined");
    return false;
  }
  for (tt::LoadOp load : group) {
    if (load.getIsVolatile())
      return false;
    // Loads staged into shared memory for a dot keep their own encoding
    // decisions; only register loads are merged.
    if (llvm::any_of(load->getUsers(), [](Operation *user) {
          return isa<ttg::LocalAllocOp>(user);
        }))
      return false;
  }
  // No global store between the first and the last copy.
  for (Operation *op = group.front()->getNextNode(); op != group.back();
       op = op->getNextNode()) {
    if (mayWriteGlobalMemory(op)) {
      LDBG("skip group: " << *op << " may write global memory");
      return false;
    }
  }
  // Keep the first pipelined copy, placed at the position of the first copy
  // so that it dominates every use.
  tt::LoadOp kept = *llvm::find_if(
      group, [&](tt::LoadOp load) { return hasLatency(load, dialect); });
  if (kept != group.front() && !moveWithPureSliceBefore(kept, group.front())) {
    LDBG("skip group: the pipelined copy " << *kept
                                           << " cannot be moved first");
    return false;
  }
  int maxLatency = 0;
  for (tt::LoadOp load : group)
    if (auto attr = dialect->getLatencyAttrHelper().getAttr(load))
      maxLatency = std::max<int>(maxLatency, attr.getInt());
  dialect->getLatencyAttrHelper().setAttr(
      kept, Builder(kept).getI32IntegerAttr(maxLatency));
  for (tt::LoadOp load : group) {
    if (load == kept)
      continue;
    LDBG("replace " << *load << " with a layout conversion of " << *kept);
    if (load.getType() == kept.getType()) {
      load.replaceAllUsesWith(kept.getResult());
    } else {
      OpBuilder builder(load);
      auto cvt = ttg::ConvertLayoutOp::create(builder, load.getLoc(),
                                              load.getType(), kept.getResult());
      load.replaceAllUsesWith(cvt.getResult());
    }
    load.erase();
  }
  return true;
}

static void mergeDuplicateLoads(scf::ForOp forOp, bool mergeAny) {
  bool changed = true;
  while (changed) {
    changed = false;
    SmallVector<tt::LoadOp> loads;
    for (Operation &op : forOp.getBody()->without_terminator()) {
      auto load = dyn_cast<tt::LoadOp>(&op);
      if (load && isa<RankedTensorType>(load.getType()))
        loads.push_back(load);
    }
    LayoutEquivalence eq;
    llvm::SmallDenseSet<Operation *, 8> grouped;
    for (unsigned i = 0; i < loads.size() && !changed; ++i) {
      if (grouped.count(loads[i]))
        continue;
      SmallVector<tt::LoadOp> group = {loads[i]};
      for (unsigned j = i + 1; j < loads.size(); ++j) {
        if (!grouped.count(loads[j]) &&
            sameLoadModuloLayout(loads[i], loads[j], eq))
          group.push_back(loads[j]);
      }
      if (group.size() < 2)
        continue;
      for (tt::LoadOp load : group)
        grouped.insert(load);
      // The merge invalidates the equivalence memo and the load list; start
      // over after every change.
      changed = mergeGroup(group, mergeAny);
    }
  }
}

struct MergeDuplicateLoadsPass
    : public impl::TritonGPUMergeDuplicateLoadsBase<MergeDuplicateLoadsPass> {
  using impl::TritonGPUMergeDuplicateLoadsBase<
      MergeDuplicateLoadsPass>::TritonGPUMergeDuplicateLoadsBase;

  void runOnOperation() override {
    if (mode != "all" && mode != "any") {
      getOperation().emitError("tritongpu-merge-duplicate-loads: mode must be "
                               "\"all\" or \"any\", got \"")
          << mode << "\"";
      return signalPassFailure();
    }
    bool mergeAny = mode == "any";
    SmallVector<scf::ForOp> loops;
    getOperation()->walk([&](scf::ForOp forOp) { loops.push_back(forOp); });
    for (scf::ForOp forOp : loops)
      mergeDuplicateLoads(forOp, mergeAny);
  }
};

} // namespace
} // namespace mlir::triton::gpu
