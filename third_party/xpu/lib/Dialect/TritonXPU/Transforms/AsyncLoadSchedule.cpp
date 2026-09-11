#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/Dominance.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "triton/Dialect/LLVMXPU/IR/Dialect.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/TritonXPU/IR/Dialect.h"
#include "triton/Dialect/TritonXPU/Transforms/Passes.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/Debug.h"

#define DEBUG_TYPE "tritonxpu-async-load-schedule"

using namespace mlir;
using namespace mlir::triton;

namespace mlir {
namespace triton {
namespace xpu {

#define GEN_PASS_DEF_TRITONXPUASYNCLOADSCHEDULE
#include "triton/Dialect/TritonXPU/Transforms/Passes.h.inc"

namespace {

struct ForwardingChain {
  SmallVector<Operation *> ops;
  Operation *firstRealUser = nullptr;
};

class TritonXPUAsyncLoadSchedulePass
    : public impl::TritonXPUAsyncLoadScheduleBase<
          TritonXPUAsyncLoadSchedulePass> {
public:
  using impl::TritonXPUAsyncLoadScheduleBase<
      TritonXPUAsyncLoadSchedulePass>::TritonXPUAsyncLoadScheduleBase;

  TritonXPUAsyncLoadSchedulePass() = default;
  TritonXPUAsyncLoadSchedulePass(bool dumpFlag) { this->dumpFlag = dumpFlag; }

  void runOnOperation() override {
    ModuleOp mod = getOperation();

    // Phase 0: Hoist gm2lm ops upward past non-dependent ops to enable
    // earlier DMA issue and better overlap.
    SmallVector<Operation *> gm2lmOps;
    mod.walk([&](Operation *op) {
      if (isa<triton::xpu::GM2LMOp, triton::xpu::GM2LMMaskOp>(op))
        gm2lmOps.push_back(op);
    });
    for (auto *op : gm2lmOps)
      tryHoistGM2LM(op);

    SmallVector<triton::xpu::LoadOp> loadOps;
    mod.walk([&](triton::xpu::LoadOp loadOp) { loadOps.push_back(loadOp); });

    llvm::DenseMap<Operation *, Operation *> insertedMfences;
    for (auto loadOp : loadOps)
      trySchedule(loadOp, insertedMfences);
    eraseDominatedMfences(insertedMfences);
  }

private:
  // Check if op depends on (uses a result of) target.
  bool dependsOn(Operation *op, Operation *target) const {
    for (Value operand : op->getOperands()) {
      if (Operation *def = operand.getDefiningOp()) {
        if (def == target)
          return true;
      }
    }
    return false;
  }

  // Check if it's safe to hoist gm2lm above op.
  bool canHoistAcross(Operation *gm2lmOp, Operation *op) const {
    // gm2lm cannot move above its own operand definitions.
    for (Value operand : gm2lmOp->getOperands()) {
      if (Operation *def = operand.getDefiningOp()) {
        if (def == op)
          return false;
      }
    }
    // Don't cross terminators.
    if (op->hasTrait<OpTrait::IsTerminator>())
      return false;
    // Don't cross other gm2lm/store/lm2gm (memory ops).
    if (isa<triton::xpu::GM2LMOp, triton::xpu::GM2LMMaskOp,
            triton::xpu::StoreOp, triton::xpu::LM2GMOp,
            triton::xpu::LM2GMMaskOp>(op))
      return false;
    // Don't cross mfence.
    if (isa<LLVM::XPU::MfenceOp>(op))
      return false;
    // Don't cross load that aliases with the gm2lm's LM buffer.
    if (auto loadOp = dyn_cast<triton::xpu::LoadOp>(op)) {
      Value gm2lmResult = gm2lmOp->getResult(0);
      return !mayAliasLM(gm2lmResult, loadOp.getPtr());
    }
    // Region ops (reduce/scan/if...) are safe to cross *iff* they have no
    // memory effects — a pure reduction touches only SSA values, never the LM
    // buffer our DMA lands in, so issuing the DMA before it just overlaps the
    // transfer with the reduction. Ops the gm2lm actually depends on are held
    // back separately (see collectLocalDeps `barriers`), so this only lets us
    // cross *unrelated* region ops. Impure ops (side effects) stay barriers.
    return isMemoryEffectFree(op);
  }

  // Collect the set of same-block ops that must move together with `gm2lmOp`
  // when it is hoisted to just before `insertBefore`: the transitive
  // operand-closure ops that are defined at or after insertBefore and can be
  // relocated (memory-effect-free, region-free address computation). Operands
  // defined strictly before insertBefore already dominate and stay put.
  //
  // Any dependency that *cannot* move — an op with memory effects (e.g. a
  // gather index load feeding the address) or a region op (reduce/scan) whose
  // result the address depends on — is recorded in `barriers` instead. The
  // gm2lm must never be hoisted above a barrier; the caller uses them as hard
  // lower bounds for the insertion point (so we still hoist as far up as the
  // nearest barrier rather than giving up entirely). `deps` is returned sorted
  // in block order.
  void collectLocalDeps(Operation *gm2lmOp, Operation *insertBefore,
                        SmallVectorImpl<Operation *> &deps,
                        DenseSet<Operation *> &barriers) const {
    Block *block = gm2lmOp->getBlock();
    SmallVector<Operation *> worklist;
    DenseSet<Operation *> visited;
    worklist.push_back(gm2lmOp);
    visited.insert(gm2lmOp);
    while (!worklist.empty()) {
      Operation *cur = worklist.pop_back_val();
      for (Value operand : cur->getOperands()) {
        Operation *def = operand.getDefiningOp();
        // Block args / values from enclosing regions always dominate.
        if (!def || def->getBlock() != block)
          continue;
        // Already dominates the insertion point: no need to move it.
        if (def->isBeforeInBlock(insertBefore))
          continue;
        if (!visited.insert(def).second)
          continue;
        // This def sits in (insertBefore, gm2lmOp]. Pure, region-free ops are
        // address computation we carry along; anything else (memory effects or
        // a nested region) cannot be relocated and becomes a hard lower bound.
        if (isMemoryEffectFree(def) && def->getNumRegions() == 0) {
          deps.push_back(def);
          worklist.push_back(def);
        } else {
          barriers.insert(def);
        }
      }
    }

    // Sort in original block order so we move them in topological order.
    llvm::sort(
        deps, [](Operation *a, Operation *b) { return a->isBeforeInBlock(b); });
  }

  // Verify that relocating every op in `moveSet` to just before `insertBefore`
  // preserves SSA dominance. For each moving op, every same-block operand
  // definition must either be part of the move set (relative order is kept) or
  // already sit before insertBefore (so it still dominates after the move).
  // Operands defined outside the block (block args / enclosing-region values)
  // always dominate and are safe.
  bool hoistPreservesDominance(ArrayRef<Operation *> moveSet,
                               Operation *insertBefore) const {
    DenseSet<Operation *> moving(moveSet.begin(), moveSet.end());
    // insertBefore is the fixed anchor the whole move set is relocated in
    // front of. If it is itself part of the move set, relocating the other
    // ops "before insertBefore" reorders them around their own dependency
    // (insertBefore) and produces a use-before-def. This happens when the
    // backward scan stops on a dependency op (e.g. the topmost address
    // computation, right below a region op we cannot cross). Bail out — the
    // load simply stays synchronous.
    if (moving.contains(insertBefore))
      return false;
    Block *block = insertBefore->getBlock();
    for (Operation *op : moveSet) {
      for (Value operand : op->getOperands()) {
        Operation *def = operand.getDefiningOp();
        if (!def || def->getBlock() != block)
          continue;
        if (moving.contains(def))
          continue;
        if (!def->isBeforeInBlock(insertBefore))
          return false;
      }
    }
    return true;
  }

  // Try to hoist a gm2lm op upward past non-dependent ops in the same block,
  // bringing its local dependencies along.
  void tryHoistGM2LM(Operation *gm2lmOp) const {
    Block *block = gm2lmOp->getBlock();
    if (!block)
      return;

    // First, collect the full set of local deps (address computation) that
    // would need to move with gm2lm, plus the hard lower-bound barriers it
    // depends on (impure/region ops that cannot be relocated). Use block begin
    // as initial bound to discover everything.
    SmallVector<Operation *> allDeps;
    DenseSet<Operation *> barriers;
    collectLocalDeps(gm2lmOp, &block->front(), allDeps, barriers);

    DenseSet<Operation *> depSet(allDeps.begin(), allDeps.end());
    depSet.insert(gm2lmOp);

    // Scan backwards from gm2lm to find the earliest safe insertion point,
    // skipping over ops that are part of the dep set (they'll move too) and
    // stopping at barriers (dependencies we cannot move above) or any op we
    // cannot legally cross.
    Operation *insertBefore = gm2lmOp;
    for (Operation *op = gm2lmOp->getPrevNode(); op; op = op->getPrevNode()) {
      if (depSet.contains(op)) {
        insertBefore = op;
        continue;
      }
      if (barriers.contains(op))
        break;
      if (!canHoistAcross(gm2lmOp, op))
        break;
      insertBefore = op;
    }
    if (insertBefore == gm2lmOp)
      return;
    // Check that insertBefore is actually before the earliest dep. If the
    // backward scan stopped right on top of the earliest dependency, there is
    // no non-dep op to anchor in front of — hoisting would place ops before
    // their own operands.
    if (!allDeps.empty() &&
        (insertBefore == allDeps.front() || depSet.contains(insertBefore)))
      return;

    // Re-collect deps constrained to the actual insertion point.
    SmallVector<Operation *> deps;
    DenseSet<Operation *> unusedBarriers;
    collectLocalDeps(gm2lmOp, insertBefore, deps, unusedBarriers);

    // Final dominance-safety gate: verify that moving the whole set to just
    // before insertBefore keeps every operand definition dominating its uses.
    // collectLocalDeps already guarantees the set is operand-closed, but this
    // is a cheap, explicit correctness backstop against invalid IR
    // ("operand does not dominate this use").
    SmallVector<Operation *> moveSet(deps.begin(), deps.end());
    moveSet.push_back(gm2lmOp);
    if (!hoistPreservesDominance(moveSet, insertBefore))
      return;

    // Move deps first (in order), then gm2lm.
    for (Operation *dep : deps) {
      if (dep->isBeforeInBlock(insertBefore))
        continue;
      dep->moveBefore(insertBefore);
    }
    if (!gm2lmOp->isBeforeInBlock(insertBefore))
      gm2lmOp->moveBefore(insertBefore);
    LLVM_DEBUG(llvm::dbgs() << "[AsyncLoadSchedule] hoisted gm2lm (with "
                            << deps.size() << " deps): " << *gm2lmOp << "\n");
  }

  bool isAsyncProducer(Operation *op) const {
    if (auto gm2lmOp = dyn_cast<triton::xpu::GM2LMOp>(op))
      return gm2lmOp.getSyncMode() == MemorySyncMode::ASYNC;
    if (auto gm2lmOp = dyn_cast<triton::xpu::GM2LMMaskOp>(op))
      return gm2lmOp.getSyncMode() == MemorySyncMode::ASYNC;
    return false;
  }

  bool setAsyncProducer(Operation *op) const {
    if (!isa<triton::xpu::GM2LMOp, triton::xpu::GM2LMMaskOp>(op))
      return false;
    auto async =
        MemorySyncModeAttr::get(op->getContext(), MemorySyncMode::ASYNC);
    op->setAttr("syncMode", async);
    return true;
  }

  bool hasSingleLoadUser(Operation *producer,
                         triton::xpu::LoadOp loadOp) const {
    if (!producer || producer->getNumResults() != 1)
      return false;
    Value result = producer->getResult(0);
    if (!result.hasOneUse())
      return false;
    return *result.user_begin() == loadOp.getOperation();
  }

  Operation *getEarliestUserInBlock(Value value, Block *block) const {
    Operation *earliest = nullptr;
    for (Operation *user : value.getUsers()) {
      if (user->getBlock() != block)
        return nullptr;
      if (!earliest || user->isBeforeInBlock(earliest))
        earliest = user;
    }
    return earliest;
  }

  FailureOr<ForwardingChain>
  getForwardingChain(triton::xpu::LoadOp loadOp) const {
    ForwardingChain chain;
    Block *block = loadOp->getBlock();
    Operation *firstUser = getEarliestUserInBlock(loadOp.getResult(), block);
    if (!firstUser)
      return failure();

    if (auto broadcastOp = dyn_cast<triton::xpu::BroadcastOp>(firstUser)) {
      if (!broadcastOp->getResult(0).hasOneUse())
        return failure();
      Operation *realUser = *broadcastOp->getResult(0).user_begin();
      if (realUser->getBlock() != block)
        return failure();
      chain.ops.push_back(broadcastOp.getOperation());
      chain.firstRealUser = realUser;
      return chain;
    }

    chain.firstRealUser = firstUser;
    return chain;
  }

  Value getLMBase(Value ptr) const {
    Operation *def = ptr.getDefiningOp();
    if (auto gm2lmOp = dyn_cast_or_null<triton::xpu::GM2LMOp>(def))
      return gm2lmOp.getBufPtr();
    if (auto gm2lmOp = dyn_cast_or_null<triton::xpu::GM2LMMaskOp>(def))
      return gm2lmOp.getBufPtr();
    return ptr;
  }

  bool mayAliasLM(Value lhs, Value rhs) const {
    Value lhsBase = getLMBase(lhs);
    Value rhsBase = getLMBase(rhs);
    return !lhsBase || !rhsBase || lhsBase == rhsBase;
  }

  bool canMoveAcross(Operation *op, Value movingPtr) const {
    if (op->hasTrait<OpTrait::IsTerminator>() || op->getNumRegions() != 0)
      return false;
    if (auto loadOp = dyn_cast<triton::xpu::LoadOp>(op))
      return !mayAliasLM(movingPtr, loadOp.getPtr());
    if (auto gm2lmOp = dyn_cast<triton::xpu::GM2LMOp>(op))
      return !mayAliasLM(movingPtr, gm2lmOp.getResult());
    if (auto gm2lmOp = dyn_cast<triton::xpu::GM2LMMaskOp>(op))
      return !mayAliasLM(movingPtr, gm2lmOp.getResult());
    if (isa<triton::xpu::StoreOp, triton::xpu::LM2GMOp,
            triton::xpu::LM2GMMaskOp>(op))
      return false;
    if (isa<LLVM::XPU::MfenceOp>(op))
      return true;
    return isMemoryEffectFree(op);
  }

  bool canMoveTo(triton::xpu::LoadOp loadOp,
                 ArrayRef<Operation *> forwardingOps,
                 Operation *insertBefore) const {
    if (!insertBefore || loadOp->getBlock() != insertBefore->getBlock())
      return false;
    if (!loadOp->isBeforeInBlock(insertBefore))
      return false;

    DenseSet<Operation *> movingOps;
    movingOps.insert(loadOp.getOperation());
    for (Operation *op : forwardingOps)
      movingOps.insert(op);

    // Collect all values produced by moving ops.
    DenseSet<Value> movingValues;
    movingValues.insert(loadOp.getResult());
    for (Operation *op : forwardingOps)
      for (Value result : op->getResults())
        movingValues.insert(result);

    Value movingPtr = loadOp.getPtr();
    for (Operation *op = loadOp->getNextNode(); op && op != insertBefore;
         op = op->getNextNode()) {
      if (movingOps.contains(op))
        continue;
      if (!canMoveAcross(op, movingPtr))
        return false;
      // Check if this op uses any result of the moving ops — if so, we'd
      // break dominance by sinking past it.
      for (Value operand : op->getOperands()) {
        if (movingValues.contains(operand))
          return false;
      }
    }
    return true;
  }

  bool hasMfenceBefore(Operation *op) const {
    Operation *prev = op->getPrevNode();
    return prev && isa<LLVM::XPU::MfenceOp>(prev);
  }

  void insertMfenceBefore(
      Operation *op, Operation *producer,
      llvm::DenseMap<Operation *, Operation *> &insertedMfences) const {
    if (hasMfenceBefore(op))
      return;
    OpBuilder builder(op);
    auto loc = op->getLoc();
    auto i32Ty = builder.getIntegerType(32);
    // Fence LM only (mask bit0=1): the async producer is a GM2LM DMA whose
    // data lands in LM, and the op we fence before consumes that LM buffer.
    auto fenceValue = builder.create<arith::ConstantIntOp>(loc, i32Ty, 1);
    auto fenceOp = builder.create<LLVM::XPU::MfenceOp>(loc, fenceValue);
    insertedMfences[fenceOp.getOperation()] = producer;
  }

  void eraseDominatedMfences(
      llvm::DenseMap<Operation *, Operation *> &insertedMfences) {
    if (insertedMfences.empty())
      return;
    DominanceInfo dominance(getOperation());

    // Collect the fences we inserted, in program (walk) order.
    SmallVector<Operation *> mfences;
    getOperation()->walk([&](Operation *op) {
      if (insertedMfences.count(op))
        mfences.push_back(op);
    });

    // An `mfence 1` is a full LM barrier: once it executes, every GM->LM DMA
    // issued before it has completed. So a fence `m` guarding the load of
    // producer `p_m` is redundant iff some *retained* fence `keep` is
    // guaranteed to execute strictly between `p_m` (DMA issue) and `m` (the
    // consuming load): dominates(p_m, keep) && dominates(keep, m).
    //
    // Crucially we only ever credit a fence that we KEEP. The previous
    // implementation credited any dominating fence, which could itself be
    // erased later against a fence sitting *before* p_m — collapsing the chain
    // onto a fence that no longer covers p_m's buffer and letting the load read
    // the buffer before its DMA landed (data race). Sweeping top-down and only
    // trusting the most recent retained fence keeps the retained set a valid
    // cover for every producer.
    DenseSet<Operation *> eraseSet;
    Operation *lastKept = nullptr;
    for (Operation *m : mfences) {
      Operation *producer = insertedMfences.lookup(m);
      if (lastKept && producer && dominance.dominates(producer, lastKept) &&
          dominance.dominates(lastKept, m)) {
        eraseSet.insert(m);
      } else {
        lastKept = m;
      }
    }

    for (Operation *mfence : eraseSet) {
      Value fenceValue = mfence->getOperand(0);
      mfence->erase();
      Operation *constantOp = fenceValue.getDefiningOp();
      if (constantOp && constantOp->use_empty())
        constantOp->erase();
    }
  }

  // Find the furthest op in [loadOp+1, insertBefore) that loadOp can move just
  // before. Returns nullptr if no valid position exists beyond loadOp itself.
  Operation *findFurthestMoveTarget(triton::xpu::LoadOp loadOp,
                                    ArrayRef<Operation *> forwardingOps,
                                    Operation *bound) const {
    if (!bound || loadOp->getBlock() != bound->getBlock())
      return nullptr;
    Operation *best = nullptr;
    DenseSet<Operation *> movingOps;
    movingOps.insert(loadOp.getOperation());
    for (Operation *op : forwardingOps)
      movingOps.insert(op);

    // Collect all values produced by moving ops.
    DenseSet<Value> movingValues;
    movingValues.insert(loadOp.getResult());
    for (Operation *op : forwardingOps)
      for (Value result : op->getResults())
        movingValues.insert(result);

    Value movingPtr = loadOp.getPtr();
    for (Operation *op = loadOp->getNextNode(); op && op != bound;
         op = op->getNextNode()) {
      if (movingOps.contains(op))
        continue;
      if (!canMoveAcross(op, movingPtr))
        break;
      // Check if this op uses any result of the moving ops.
      bool usesMovingValue = false;
      for (Value operand : op->getOperands()) {
        if (movingValues.contains(operand)) {
          usesMovingValue = true;
          break;
        }
      }
      if (usesMovingValue)
        break;
      best = op->getNextNode();
    }
    return best;
  }

  void
  trySchedule(triton::xpu::LoadOp loadOp,
              llvm::DenseMap<Operation *, Operation *> &insertedMfences) const {
    Operation *producer = loadOp.getPtr().getDefiningOp();
    if (!isa_and_nonnull<triton::xpu::GM2LMOp, triton::xpu::GM2LMMaskOp>(
            producer))
      return;
    if (!hasSingleLoadUser(producer, loadOp))
      return;
    bool producerIsAsync = isAsyncProducer(producer);

    auto chainOr = getForwardingChain(loadOp);
    if (failed(chainOr))
      return;
    ForwardingChain chain = *chainOr;
    if (!canMoveTo(loadOp, chain.ops, chain.firstRealUser)) {
      // Fallback: move as far as possible short of firstRealUser.
      Operation *fallback =
          findFurthestMoveTarget(loadOp, chain.ops, chain.firstRealUser);
      if (!fallback || fallback == loadOp->getNextNode()) {
        if (producerIsAsync)
          insertMfenceBefore(loadOp.getOperation(), producer, insertedMfences);
        return;
      }
      loadOp->moveBefore(fallback);
      Operation *insertAfter = loadOp.getOperation();
      for (Operation *forwardingOp : chain.ops) {
        forwardingOp->moveAfter(insertAfter);
        insertAfter = forwardingOp;
      }
      setAsyncProducer(producer);
      insertMfenceBefore(loadOp.getOperation(), producer, insertedMfences);
      if (dumpFlag) {
        LLVM_DEBUG(llvm::dbgs() << "[AsyncLoadSchedule] fallback move load: "
                                << *loadOp << "\n");
      }
      return;
    }

    Operation *insertBefore = chain.firstRealUser;
    loadOp->moveBefore(insertBefore);
    Operation *insertAfter = loadOp.getOperation();
    for (Operation *forwardingOp : chain.ops) {
      forwardingOp->moveAfter(insertAfter);
      insertAfter = forwardingOp;
    }

    setAsyncProducer(producer);
    insertMfenceBefore(loadOp.getOperation(), producer, insertedMfences);

    if (dumpFlag) {
      LLVM_DEBUG(llvm::dbgs()
                 << "[AsyncLoadSchedule] move load before first user: "
                 << *loadOp << "\n");
    }
  }
};

} // namespace

} // namespace xpu
} // namespace triton
} // namespace mlir
