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

#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Interfaces/ControlFlowInterfaces.h"
#include "mlir/Interfaces/LoopLikeInterface.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "mlir/Pass/Pass.h"
#include "nvidia/include/Dialect/NVWS/IR/Dialect.h"
#include "tle/dialect/include/IR/Dialect.h"
#include "tle/dialect/include/Transforms/Passes.h"
#include "tle/dialect/include/Transforms/TransformAttrs.h"
#include "triton/Analysis/Alias.h"
#include "triton/Analysis/Utility.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "triton/Dialect/TritonNvidiaGPU/IR/Dialect.h"
#include "llvm/ADT/STLExtras.h"
#include <deque>

namespace mlir::triton::tle {

namespace tt = mlir::triton;
namespace ttg = mlir::triton::gpu;
namespace ttng = mlir::triton::nvidia_gpu;
namespace ttnvws = mlir::triton::nvws;

#define GEN_PASS_DEF_TRITONTLESCHEDULETMASTORESYNC
#include "tle/dialect/include/Transforms/Passes.h.inc"

namespace {

// Keep the lattice finite, and restrict generated waits to small immediates.
static constexpr unsigned kDefaultMaxPendingGroups = 1;
static constexpr unsigned kPendingGroupsLimit = 8;

static Value canonicalizeWarpSpecializeCapture(Value value) {
  while (auto blockArg = dyn_cast<BlockArgument>(value)) {
    Block *block = blockArg.getOwner();
    auto partitions =
        dyn_cast_or_null<ttg::WarpSpecializePartitionsOp>(block->getParentOp());
    if (!partitions)
      break;
    auto wsOp = dyn_cast<ttg::WarpSpecializeOp>(partitions->getParentOp());
    if (!wsOp)
      break;
    unsigned argNo = blockArg.getArgNumber();
    OperandRange captures = wsOp.getExplicitCaptures();
    if (argNo >= captures.size())
      break;
    value = captures[argNo];
  }
  return value;
}

static Value getMemDescRoot(Value value) {
  Value current = canonicalizeWarpSpecializeCapture(value);
  while (true) {
    if (auto index = current.getDefiningOp<ttg::MemDescIndexOp>()) {
      current = canonicalizeWarpSpecializeCapture(index.getSrc());
      continue;
    }
    if (auto subslice = current.getDefiningOp<ttg::MemDescSubsliceOp>()) {
      current = canonicalizeWarpSpecializeCapture(subslice.getSrc());
      continue;
    }
    if (auto alias = current.getDefiningOp<MemDescAliasOp>()) {
      current = canonicalizeWarpSpecializeCapture(alias.getSrc());
      continue;
    }
    if (auto trans = current.getDefiningOp<ttg::MemDescTransOp>()) {
      current = canonicalizeWarpSpecializeCapture(trans.getSrc());
      continue;
    }
    if (auto reshape = current.getDefiningOp<ttg::MemDescReshapeOp>()) {
      current = canonicalizeWarpSpecializeCapture(reshape.getSrc());
      continue;
    }
    if (auto reinterpret = current.getDefiningOp<ttg::MemDescReinterpretOp>()) {
      current = canonicalizeWarpSpecializeCapture(reinterpret.getSrc());
      continue;
    }
    if (auto wgmmaView = current.getDefiningOp<MemDescWGMMAViewOp>()) {
      current = canonicalizeWarpSpecializeCapture(wgmmaView.getSrc());
      continue;
    }
    break;
  }
  return current;
}

static bool isTLEExplicitTMAStore(ttng::AsyncTMACopyLocalToGlobalOp op) {
  return op->hasAttr(kTleTMAStoreExplicitCommitAttr);
}

static bool isNonTLEStoreGroupBoundary(Operation *op) {
  if (auto tmaStore = dyn_cast<ttng::AsyncTMACopyLocalToGlobalOp>(op))
    return !isTLEExplicitTMAStore(tmaStore);
  return isa<ttng::AsyncTMAReduceOp, ttng::AsyncTMAScatterOp>(op);
}

// An age is a lower bound on the number of newer committed groups. At a
// merge, take the minimum age on paths where the source may still be read.
// Unlike a queue indexed from its front, this remains sound for unequal path
// lengths, zero-trip loops, and different group orders on different branches.
struct PendingState {
  DenseMap<Value, unsigned> ages;
  unsigned groups = 0;
  unsigned maxGroups = kDefaultMaxPendingGroups;

  bool join(const PendingState &other) {
    bool changed = false;
    for (auto [root, age] : other.ages) {
      auto [it, inserted] = ages.try_emplace(root, age);
      if (inserted || age < it->second) {
        it->second = age;
        changed = true;
      }
    }
    if (other.groups > groups) {
      groups = other.groups;
      changed = true;
    }
    return changed;
  }

  void wait(unsigned pendings) {
    for (auto it = ages.begin(); it != ages.end();) {
      auto current = it++;
      if (current->second >= pendings)
        ages.erase(current);
    }
    groups = std::min(groups, pendings);
  }

  void commit(ArrayRef<Value> roots) {
    for (auto &entry : ages)
      entry.second = std::min(entry.second + 1, maxGroups - 1);
    for (Value root : roots)
      ages[root] = 0;
    groups = std::min(groups + 1, maxGroups);
  }
};

class StoreAliasAnalysis : public SharedMemoryAliasAnalysis {
public:
  using SharedMemoryAliasAnalysis::SharedMemoryAliasAnalysis;

  void setToEntryState(dataflow::Lattice<AliasInfo> *lattice) override {
    Value value = lattice->getAnchor();
    // Standalone pass tests also use shared-memory function arguments.
    AliasInfo info;
    if (isa<ttg::MemDescType>(value.getType()))
      info.insert(getMemDescRoot(value));
    propagateIfChanged(lattice, lattice->join(info));
  }
};

struct StoreGroup {
  SmallVector<Value, 2> sources;
  SmallVector<Value, 2> roots;
};

// Other region operations may change the issuing warp/thread (notably warp
// specialization), or execute concurrently. Give them independent, drained
// scheduling domains rather than treating them as sequential branches.
static bool isSequentialRegion(Operation *op) {
  return isa<scf::ForOp, scf::IfOp, scf::WhileOp, scf::ExecuteRegionOp,
             scf::IndexSwitchOp>(op);
}

// True for an operation that introduces a new shared-memory buffer rather than
// a view of an existing one.
static bool allocatesSharedBuffer(Operation *op);

// The region a buffer is allocated in, resolving loop-carried block arguments
// to the value the loop was entered with.
static Region *getSourceRegion(Value value) {
  SmallPtrSet<Value, 4> visited;
  while (auto arg = dyn_cast<BlockArgument>(value)) {
    if (!visited.insert(value).second)
      break;
    auto loop = dyn_cast_or_null<LoopLikeOpInterface>(arg.getOwner()->getParentOp());
    if (!loop)
      break;
    OpOperand *init = loop.getTiedLoopInit(arg);
    if (!init)
      break;
    value = init->get();
  }
  return value.getParentRegion();
}

static bool isControlFlow(Operation *op) {
  return isa<BranchOpInterface>(op) || isSequentialRegion(op) ||
         (isa<RegionBranchTerminatorOpInterface>(op) &&
          isSequentialRegion(op->getParentOp()));
}

static bool allocatesSharedBuffer(Operation *op) {
  for (Value result : op->getResults()) {
    auto memDescTy = dyn_cast<ttg::MemDescType>(result.getType());
    if (!memDescTy ||
        !isa_and_nonnull<ttg::SharedMemorySpaceAttr>(memDescTy.getMemorySpace()))
      continue;
    if (getMemDescRoot(result) == result)
      return true;
  }
  return false;
}

class StoreScheduler {
public:
  StoreScheduler(ModuleOp module, unsigned maxPendingGroups)
      : module(module), maxPendingGroups(maxPendingGroups) {}

  LogicalResult run() {
    SmallVector<Block *> blocks;
    module.walk([&](Block *block) { blocks.push_back(block); });
    for (Block *block : blocks)
      normalizeGroups(*block);
    if (groups.empty())
      return success();

    solver = createDataFlowSolver();
    solver->load<StoreAliasAnalysis>();
    if (failed(solver->initializeAndRun(module)))
      return failure();
    for (auto &entry : groups)
      for (Value source : entry.second.sources)
        llvm::append_range(entry.second.roots, getRoots(source));

    buildControlFlow();
    while (!worklist.empty()) {
      Operation *op = worklist.front();
      worklist.pop_front();
      queued.erase(op);
      PendingState output = lookupInput(op);
      transfer(op, output);
      for (Operation *successor : successors[op]) {
        auto [it, inserted] = inputs.try_emplace(successor, emptyState());
        bool changed = it->second.join(output);
        if (inserted || changed)
          enqueue(successor);
      }
    }

    // No IR is changed during iteration. Joins only add possibilities or lower
    // ages, so the finite lattice converges without a fixed iteration budget.
    // Transfer can strengthen a wait and drop output facts; retaining earlier
    // facts at successor joins is a conservative over-approximation.
    for (Operation *op : operations) {
      auto it = inputs.find(op);
      if (it == inputs.end())
        continue;
      PendingState state = it->second;
      if (auto wait = transfer(op, state)) {
        OpBuilder builder(op);
        ttng::TMAStoreWaitOp::create(builder, op->getLoc(), *wait);
      }
    }
    return success();
  }

private:
  void normalizeGroups(Block &block) {
    SmallVector<ttng::AsyncTMACopyLocalToGlobalOp> stores;
    DenseSet<Operation *> loweringWaits;
    auto commit = [&](Operation *before) {
      if (stores.empty())
        return;
      OpBuilder builder(block.getParentOp());
      if (before)
        builder.setInsertionPoint(before);
      else
        builder.setInsertionPointToEnd(&block);
      auto groupOp =
          TMAStoreCommitGroupOp::create(builder, stores.front().getLoc());
      StoreGroup &group = groups[groupOp];
      for (auto store : stores)
        group.sources.push_back(store.getSrc());
      groupStarts.insert(stores.front());
      stores.clear();
    };
    for (auto it = block.begin(); it != block.end();) {
      Operation *op = &*it++;
      if (loweringWaits.erase(op)) {
        op->erase();
        continue;
      }
      if (auto store = dyn_cast<ttng::AsyncTMACopyLocalToGlobalOp>(op)) {
        if (isTLEExplicitTMAStore(store)) {
          stores.push_back(store);
          continue;
        }
      }
      if (!stores.empty() && isa<TMAStoreCommitGroupOp>(op)) {
        // Only replace the immediate wait emitted with this TLE commit.
        // In particular, never erase a wait merely because an outer region
        // has pending stores: that loses nested-region completion guarantees.
        if (auto wait =
                dyn_cast_or_null<ttng::TMAStoreWaitOp>(op->getNextNode()))
          if (wait.getPendings() == 0)
            loweringWaits.insert(wait);
        op->erase();
        continue;
      }
      if (isa<ttng::FenceAsyncSharedOp>(op))
        continue;
      commit(op);
    }
    commit(nullptr);
  }

  SmallVector<Value, 2> getRoots(Value value) const {
    SmallVector<Value, 2> roots;
    if (auto *lattice =
            solver->lookupState<dataflow::Lattice<AliasInfo>>(value))
      llvm::append_range(roots, lattice->getValue().getAllocs());
    if (roots.empty()) {
      auto pointer =
          dyn_cast<tt::PointerType>(getElementTypeOrSelf(value.getType()));
      if (isa<ttg::MemDescType>(value.getType()) ||
          (pointer && pointer.getAddressSpace() == 3))
        roots.push_back(
            Value()); // Unknown shared alias: may refer to any source.
    }
    return roots;
  }

  std::optional<unsigned> transfer(Operation *op, PendingState &state) const {
    std::optional<unsigned> required;
    auto wait = [&](unsigned n) {
      if (!state.groups)
        return;
      required = required ? std::min(*required, n) : n;
      state.wait(n);
    };
    if (auto existing = dyn_cast<ttng::TMAStoreWaitOp>(op)) {
      state.wait(existing.getPendings());
      return required;
    }
    if (auto group = groups.find(op); group != groups.end()) {
      state.commit(group->second.roots);
      return required;
    }
    if (auto store = dyn_cast<ttng::AsyncTMACopyLocalToGlobalOp>(op)) {
      if (isTLEExplicitTMAStore(store)) {
        if (groupStarts.contains(op) && state.groups == maxPendingGroups)
          wait(maxPendingGroups - 1);
        return required;
      }
    }
    if (isNonTLEStoreGroupBoundary(op)) {
      wait(0);
      SmallVector<Value> roots;
      for (Value operand : op->getOperands())
        if (isa<ttg::MemDescType>(operand.getType()))
          llvm::append_range(roots, getRoots(operand));
      // These operations lower with an implicit commit of their own.
      state.commit(roots);
      return required;
    }
    if (isa<TMAStoreCommitGroupOp>(op)) {
      wait(0);
      state.commit({});
      return required;
    }
    if (auto terminator = dyn_cast<RegionBranchTerminatorOpInterface>(op);
        terminator && isSequentialRegion(op->getParentOp())) {
      // Buffers allocated inside this region die when it exits. Allocation may
      // then hand their offsets to later buffers, whose writes race with the
      // TMA engine still reading the source; no barrier can order that. Such
      // groups must complete before the region exits, unless the terminator
      // carries the buffer out and keeps it live.
      Region *region = op->getParentRegion();
      std::optional<unsigned> escaping;
      for (auto [root, age] : state.ages) {
        if (root && (llvm::is_contained(op->getOperands(), root) ||
                     !region->isAncestor(getSourceRegion(root))))
          continue;
        escaping = escaping ? std::min(*escaping, age) : age;
      }
      if (escaping)
        wait(*escaping);
    }
    if (isControlFlow(op))
      return required;
    if (op->hasTrait<OpTrait::IsTerminator>() || op->getNumRegions() ||
        isa<ttnvws::ConsumerReleaseOp, ttng::ArriveBarrierOp>(op)) {
      wait(0);
      return required;
    }
    if (isa<ttng::FenceAsyncSharedOp>(op))
      return required;
    // WGMMA commit/wait only order the WGMMA queue. They neither overwrite a
    // TMA source nor release it to another warp. Keep their general effects
    // intact for other passes, but do not treat them as unknown shared writes.
    if (isa<ttng::WarpGroupDotCommitOp, ttng::WarpGroupDotWaitOp>(op))
      return required;

    // Allocation may give a new buffer the offset of one whose live range has
    // just ended, a source still being read by the TMA engine among them, so
    // reuse is not visible as an alias of any pending root.
    if (allocatesSharedBuffer(op))
      wait(0);

    auto reuse = [&](Value value) {
      auto roots = getRoots(value);
      if (roots.empty())
        return;
      if (llvm::is_contained(roots, Value())) {
        wait(0);
        return;
      }
      if (auto unknown = state.ages.find(Value()); unknown != state.ages.end())
        wait(unknown->second);
      for (Value root : roots)
        if (auto it = state.ages.find(root); it != state.ages.end())
          wait(it->second);
    };
    if (auto release = dyn_cast<PipeReaderReleaseOp>(op)) {
      for (Value field : release.getFields())
        reuse(field);
    } else if (auto effects = dyn_cast<MemoryEffectOpInterface>(op)) {
      SmallVector<MemoryEffects::EffectInstance> instances;
      effects.getEffects(instances);
      for (const auto &effect : instances) {
        if (!isa<MemoryEffects::Write, MemoryEffects::Free>(effect.getEffect()))
          continue;
        if (Value value = effect.getValue())
          reuse(value);
        else
          wait(0);
      }
    } else if (!isMemoryEffectFree(op)) {
      wait(0);
    }
    return required;
  }

  PendingState emptyState() const {
    PendingState state;
    state.maxGroups = maxPendingGroups;
    return state;
  }

  PendingState lookupInput(Operation *op) const {
    auto it = inputs.find(op);
    return it == inputs.end() ? emptyState() : it->second;
  }

  void enqueue(Operation *op) {
    if (queued.insert(op).second)
      worklist.push_back(op);
  }

  void buildControlFlow() {
    module.walk<WalkOrder::PreOrder>([&](Operation *op) {
      if (!op->getBlock())
        return;
      operations.push_back(op);
      auto add = [&](Operation *next) {
        if (next)
          successors[op].push_back(next);
      };
      auto addRegion = [&](RegionSuccessor successor, Operation *parent) {
        if (successor.isParent())
          add(parent->getNextNode());
        else if (!successor.getSuccessor()->empty())
          add(&successor.getSuccessor()->front().front());
      };
      if (isa<BranchOpInterface>(op)) {
        for (Block *successor : op->getSuccessors())
          add(&successor->front());
      } else if (isSequentialRegion(op)) {
        SmallVector<RegionSuccessor> regions;
        cast<RegionBranchOpInterface>(op).getSuccessorRegions(
            RegionBranchPoint::parent(), regions);
        for (auto successor : regions)
          addRegion(successor, op);
      } else if (auto term = dyn_cast<RegionBranchTerminatorOpInterface>(op);
                 term && isSequentialRegion(op->getParentOp())) {
        SmallVector<RegionSuccessor> regions;
        SmallVector<Attribute> operands(op->getNumOperands());
        term.getSuccessorRegions(operands, regions);
        for (auto successor : regions)
          addRegion(successor, op->getParentOp());
      } else if (!op->hasTrait<OpTrait::IsTerminator>()) {
        add(op->getNextNode());
      }
      if (!isSequentialRegion(op)) {
        for (Region &region : op->getRegions()) {
          if (!region.empty() && !region.front().empty()) {
            Operation *entry = &region.front().front();
            inputs.try_emplace(entry, emptyState());
            enqueue(entry);
          }
        }
      }
    });
  }

  ModuleOp module;
  unsigned maxPendingGroups;
  std::unique_ptr<DataFlowSolver> solver;
  DenseMap<Operation *, StoreGroup> groups;
  DenseSet<Operation *> groupStarts;
  SmallVector<Operation *> operations;
  DenseMap<Operation *, SmallVector<Operation *, 2>> successors;
  DenseMap<Operation *, PendingState> inputs;
  std::deque<Operation *> worklist;
  DenseSet<Operation *> queued;
};

class TritonTleScheduleTmaStoreSyncPass
    : public impl::TritonTleScheduleTmaStoreSyncBase<
          TritonTleScheduleTmaStoreSyncPass> {
public:
  using impl::TritonTleScheduleTmaStoreSyncBase<
      TritonTleScheduleTmaStoreSyncPass>::TritonTleScheduleTmaStoreSyncBase;

  void runOnOperation() override {
    unsigned groups = std::clamp<int32_t>(maxPendingGroups, 1, kPendingGroupsLimit);
    if (failed(StoreScheduler(getOperation(), groups).run()))
      signalPassFailure();
  }
};

} // namespace

} // namespace mlir::triton::tle
