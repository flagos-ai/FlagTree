#include "Analysis/TkoAliasAnalysis.h"
#include "Analysis/TkoDependenceAnalysis.h"
#include "Analysis/TkoOpSemantics.h"
#include "Transform/Passes.h"
#include "TritonToTileIR/Utils.h"
#include "Utils/Utils.h"

#include "cuda_tile/Dialect/CudaTile/IR/Ops.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallSet.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/Debug.h"

using namespace mlir;

// MLIR pass TableGen requires per-pass GEN_PASS_DEF_* macros.
namespace mlir {
namespace triton {
#define GEN_PASS_DEF_AUTOGENMEMORYTOKEN
#include "Transform/Passes.h.inc"
} // namespace triton
} // namespace mlir

#define DEBUG_TYPE "add-memory-token"

/*
 * AutoGenMemoryToken: insert memToken operands on TKO memory ops so the
 * token graph preserves as-if-sequential source semantics.
 *
 * Goal: given a function with un-tokenized TKO memory ops, emit the token
 * graph that preserves required memory ordering without unnecessary edges.
 * Ordinary memory ops get a token edge iff they may-alias and at least one
 * writes. Fence-like and signal ops add ordering described below.
 *
 * Conceptual split:
 *   - TkoAliasAnalysis maps TKO memory accesses to provenance roots: which
 *     allocation roots can this op touch?
 *   - TkoDependenceAnalysis proves access-footprint facts after root aliasing
 *     is known, for example loop-carried disjoint stores on one root.
 *   - This pass turns the required memory-ordering facts into explicit token
 *     SSA and threads those tokens through control flow.
 *
 * Alias granularity is decided by TkoAliasAnalysis: each op's access value
 * (ptr / view / tensor handle) is walked back to a set of terminal root values;
 * ops that share a root require dependence analysis or token ordering.
 * Default: distinct SSA roots are no-alias (Triton's implicit function-arg
 * convention).
 *
 * Per-memory-root token state:
 *   - lastOp[c]    — output of the most-recent mem op touching memory root c.
 *                    Eagerly extended by reads via JOIN so that subsequent
 *                    writes WAR against all pending reads in one edge.
 *   - lastStore[c] — output of the most-recent write/atomic on memory root c.
 *                    Reads take this as their RAW input.
 *   - acquireToken — output of the most-recent acquire-side fence. Every
 *                    subsequent tokenized memory op joins with this so
 *                    post-acquire ops observe the fence's output even across
 *                    iterations.
 *   - parallelStoreInputToken[c]
 *                  — loop-invariant input token for a conservative
 *                    single-store cuda_tile.for optimization. The store takes
 *                    this token as input every iteration, while its output is
 *                    accumulated into the loop-carried lastStore token.
 *
 * Per-op rules:
 *   Read  S: in = JOIN({lastStore[c] : c ∈ S} ∪ {acquireToken})
 *            out → update lastOp[c] via eager-join for each c ∈ S.
 *   Write S: in = JOIN({lastOp[c] : c ∈ S} ∪ {acquireToken})
 *            out becomes lastOp[c] = lastStore[c] for each c ∈ S.
 *   Atomic S: same as Write.
 *   Acquire fence (gdc_wait):
 *               in = none; out → acquireToken.
 *   Dependent launch signal (gdc_launch_dependents):
 *               in = JOIN({lastStore[*]} ∪ {acquireToken}); output is not
 *               published into TokenState.
 *   gpu.barrier: acq_rel / total local fence.
 *               Its output broadcasts to lastOp / lastStore for every memory
 *               root and to acquireToken.
 *
 * Control flow: pre-pass computes per-region memory summaries per memory root.
 * cuda_tile.for / cuda_tile.loop thread one iter_arg per (root, role)
 * touched by the body: LOAD → 1 arg (lastOp), STORE/ATOMIC → 2 args
 * (lastOp, lastStore), plus 1 arg for acquireToken if any body op has an
 * acquire side.
 * cuda_tile.if yields one result per (root, role) advanced in
 * either branch.
 *
 * User bail-out: if any TKO mem op already carries a memToken operand, the
 * pass does nothing — assumes the user owns the ordering. Unsupported
 * region ops (fixed-result ops, ops with no terminator) that contain mem
 * ops produce a pass-level error rather than a silently incomplete graph.
 *
 * Future work:
 *   - Broader footprint-disjoint proofs beyond the current dependence subset.
 *   - Per-op acquire/release memory_ordering_semantics on TKO memory ops
 *     (treated as relaxed for now). GDC wait is modeled as an acquire fence;
 *     GDC launch is modeled as a dependent-launch signal, not as a full
 *     release fence.
 *   - Interprocedural alias.
 */

namespace {

using namespace mlir::triton;
using tileir::TkoAliasAnalysis;
using tileir::TkoDependenceAnalysis;
using MemoryRootId = TkoAliasAnalysis::MemoryRootId;

// Coarsened memory effect for region summary.
enum class MemEffect : unsigned char { None = 0, Load = 1, Store = 2 };

static MemEffect maxEffect(MemEffect a, MemEffect b) {
  return static_cast<MemEffect>(
      std::max(static_cast<unsigned char>(a), static_cast<unsigned char>(b)));
}

/// Summary of memory-ordering behavior inside one MLIR region.
///
/// AutoGenMemoryToken computes this bottom-up before rewriting structured
/// control flow. Loop/branch handlers use it to decide which token state must
/// be represented at region boundaries: per-root lastOp/lastStore tokens for
/// memory effects, plus the acquire token when an acquire-side fence inside
/// the region can affect ordering outside the region.
struct RegionMemorySummary {
  // Strongest load/store effect seen for each memory root, including effects
  // from nested regions.
  DenseMap<MemoryRootId, MemEffect> perRoot;

  // The region contains an acquire-side fence such as gdc_wait or gpu.barrier.
  // Loop handlers thread acquireToken through the loop when this is set.
  bool hasAcquireFence = false;

  // The region contains a standard release-side fence. Today this is only the
  // release side of gpu.barrier; gdc_launch_dependents is modeled separately as
  // a dependent-launch signal.
  bool hasReleaseFence = false;

  // The region contains a total fence such as gpu.barrier. Total fences publish
  // an ordering edge to all roots, so loop handlers must carry that fence token
  // out even for roots not directly touched by the loop body.
  bool hasTotalFence = false;
};

// Per-scope token state. lastOp / lastStore / acquireToken hold the current
// token values. `advanced*` is bookkeeping for control-flow rewrites: it
// records which token slots changed in this scope so enclosing if/loop ops
// know which values must be yielded or threaded out.
struct TokenState {
  DenseMap<MemoryRootId, Value> lastOp;
  DenseMap<MemoryRootId, Value> lastStore;
  DenseMap<MemoryRootId, Value> parallelStoreInputToken;
  Value acquireToken;

  llvm::SmallSet<MemoryRootId, 8> advancedLastOp;
  llvm::SmallSet<MemoryRootId, 8> advancedLastStore;
  bool advancedAcquireToken = false;

  void advanceLastOp(MemoryRootId root, Value token) {
    lastOp[root] = token;
    advancedLastOp.insert(root);
  }

  void advanceLastStore(MemoryRootId root, Value token) {
    lastStore[root] = token;
    advancedLastStore.insert(root);
  }

  void advanceStoreState(MemoryRootId root, Value token) {
    advanceLastOp(root, token);
    advanceLastStore(root, token);
  }

  void advanceAcquire(Value token) {
    acquireToken = token;
    advancedAcquireToken = true;
  }

  void clearAdvancedMarkers() {
    advancedLastOp.clear();
    advancedLastStore.clear();
    advancedAcquireToken = false;
  }

  bool empty() const {
    return lastOp.empty() && lastStore.empty() &&
           parallelStoreInputToken.empty() && !acquireToken;
  }
};

// For collecting Break/Continue terminators inside scf.while body when
// nested under an scf.if — these need to be threaded at the LoopOp level.
using OpToStates = SmallVector<std::pair<Operation *, TokenState>, 4>;

// Token-state demand from code that executes after the current point in the
// enclosing scope. Loop handlers use this to decide whether a LOAD-only root's
// lastOp must escape the loop.
struct DownstreamTokenDemand {
  // A later write/atomic on root c consumes lastOp[c] for WAR ordering.
  llvm::SmallSet<MemoryRootId, 8> rootsConsumedByLaterWrite;
  // A later standard release-side fence consumes every live lastOp slot.
  bool hasLaterReleaseFence = false;
};

/// Planner output for the token slots that must cross a loop boundary.
///
/// This struct contains only ordering decisions, not IR values. Loop handlers
/// consume it to perform the mechanical rewrite: append operands/block args,
/// extend terminators, and install rewritten loop results back into TokenState.
struct LoopTokenThreadingPlan {
  SmallVector<MemoryRootId, 4> threadLastOp;
  SmallVector<MemoryRootId, 4> threadLastStore;
  bool threadAcquire = false;
  llvm::SmallSet<MemoryRootId, 8> parallelStoreRoots;

  bool empty() const {
    return threadLastOp.empty() && threadLastStore.empty() && !threadAcquire;
  }

  size_t numTokenSlots() const {
    return threadLastOp.size() + threadLastStore.size() +
           (threadAcquire ? 1 : 0);
  }
};

/// Planner output for token results/yield operands added to an if-like op or
/// single-region wrapper.
/// Slots are ordered lastOp, lastStore, then optional acquire, matching the
/// loop token slot order.
struct TokenYieldPlan {
  SmallVector<MemoryRootId, 4> yieldLastOp;
  SmallVector<MemoryRootId, 4> yieldLastStore;
  bool yieldAcquire = false;

  bool empty() const {
    return yieldLastOp.empty() && yieldLastStore.empty() && !yieldAcquire;
  }

  size_t numTokenSlots() const {
    return yieldLastOp.size() + yieldLastStore.size() + (yieldAcquire ? 1 : 0);
  }
};

/// Block-level token facts consumed by the entry policy.
struct BlockTokenSummary {
  bool hasUserToken = false;
  bool hasDebugBarrier = false;
  bool hasFenceOrSignal = false;
};

/// Pure token-ordering decisions. This helper consumes already-computed region
/// summaries and downstream token demand, then produces small plan structs;
/// it does not inspect or mutate IR. AutoGenMemoryTokenPass remains
/// responsible for materializing those plans as token SSA.
struct TokenOrderPlanner {
  static LoopTokenThreadingPlan
  planLoopThreading(const RegionMemorySummary &summary,
                    const DownstreamTokenDemand &downstreamDemand) {
    LoopTokenThreadingPlan plan;
    // Acquire-token slot is threaded only for acquire-side fences. A
    // release-only fence consumes prior state but does not publish a token that
    // later memory ops must consume.
    plan.threadAcquire = summary.hasAcquireFence;
    for (auto &[c, eff] : summary.perRoot) {
      if (eff == MemEffect::Store) {
        plan.threadLastOp.push_back(c);
        plan.threadLastStore.push_back(c);
      } else if (eff == MemEffect::Load) {
        // Pure-read skip: for a LOAD-only root inside this loop, avoid
        // threading lastOp[c] as iter_arg when no downstream op will ever
        // consume it. Skipping keeps loop iterations independent for the the
        // compiler pipeliner.
        //
        // Consumers that DO need lastOp[c] to escape the loop: a write on root
        // c *after* this loop (WAR hazard), or a later standard release-side
        // fence that must join all prior memory state. gdc_launch_dependents is
        // only a dependent-launch signal, so it does not force read-only loop
        // tokens to escape.
        if (isLastOpLiveAfter(c, downstreamDemand))
          plan.threadLastOp.push_back(c);
      }
    }
    return plan;
  }

  static TokenYieldPlan planBranchYieldTokens(const TokenState &thenExit,
                                              bool includeThen,
                                              const TokenState &elseExit,
                                              bool includeElse) {
    llvm::SmallSet<MemoryRootId, 8> unionLastOp;
    llvm::SmallSet<MemoryRootId, 8> unionLastStore;
    bool unionAcquire = false;

    auto mergeBranch = [&](const TokenState &branch) {
      unionAcquire |= branch.advancedAcquireToken;
      for (MemoryRootId c : branch.advancedLastOp)
        unionLastOp.insert(c);
      for (MemoryRootId c : branch.advancedLastStore)
        unionLastStore.insert(c);
    };

    if (includeThen)
      mergeBranch(thenExit);
    if (includeElse)
      mergeBranch(elseExit);

    TokenYieldPlan plan;
    plan.yieldLastOp.append(unionLastOp.begin(), unionLastOp.end());
    plan.yieldLastStore.append(unionLastStore.begin(), unionLastStore.end());
    plan.yieldAcquire = unionAcquire;
    return plan;
  }

  static TokenYieldPlan planSingleRegionYieldTokens(const TokenState &exit) {
    TokenState empty;
    return planBranchYieldTokens(exit, /*includeThen=*/true, empty,
                                 /*includeElse=*/false);
  }

private:
  static bool isLastOpLiveAfter(MemoryRootId c,
                                const DownstreamTokenDemand &downstreamDemand) {
    return downstreamDemand.hasLaterReleaseFence ||
           downstreamDemand.rootsConsumedByLaterWrite.contains(c);
  }
};

// ---------------------------------------------------------------------------
// Op category predicates
// ---------------------------------------------------------------------------

static bool isReadMemOp(Operation *op) {
  return tileir::tko_op_semantics::isReadOnly(op);
}

static bool isWriteMemOp(Operation *op) {
  return tileir::tko_op_semantics::isWriteLike(op);
}

static bool isMemOp(Operation *op) {
  return tileir::tko_op_semantics::isMemOp(op);
}

static bool isAcquireFence(Operation *op) {
  return tileir::tko_op_semantics::isAcquireFence(op);
}

static bool isReleaseFence(Operation *op) {
  return tileir::tko_op_semantics::isReleaseFence(op);
}

static bool isDependentLaunchSignal(Operation *op) {
  return tileir::tko_op_semantics::isDependentLaunchSignal(op);
}

static bool isTotalFence(Operation *op) {
  return tileir::tko_op_semantics::isTotalFence(op);
}

static bool isFenceOrSignal(Operation *op) {
  return isAcquireFence(op) || isReleaseFence(op) ||
         isDependentLaunchSignal(op);
}

static bool isTokenOrderingOp(Operation *op) {
  return isMemOp(op) || isFenceOrSignal(op);
}

// Return true iff `op` already has a non-null memToken operand — user has
// taken ownership of the token chain, so auto-gen should bail out.
static bool memOpHasUserToken(Operation *op) {
  return tileir::tko_op_semantics::hasUserToken(op);
}

static bool containsNestedMemLikeOp(Operation *op) {
  for (Region &region : op->getRegions()) {
    auto interrupted = region.walk([&](Operation *nested) {
      if (isTokenOrderingOp(nested))
        return WalkResult::interrupt();
      return WalkResult::advance();
    });
    if (interrupted.wasInterrupted())
      return true;
  }
  return false;
}

// ---------------------------------------------------------------------------
// AutoGenMemoryTokenPass
// ---------------------------------------------------------------------------

class AutoGenMemoryTokenPass
    : public ::mlir::triton::impl::AutoGenMemoryTokenBase<
          AutoGenMemoryTokenPass> {
  // Per-function state (set up at the start of processing each function body).
  std::optional<TkoAliasAnalysis> aliasInfo;
  std::optional<TkoDependenceAnalysis> dependenceInfo;
  DenseMap<Region *, RegionMemorySummary> regionSummaries;

  // -------------------------------------------------------------------------
  // Helpers
  // -------------------------------------------------------------------------

  /// Return the memory roots touched by `op`, or empty if it is not a
  /// recognized TKO memory op.
  SmallVector<MemoryRootId, 2> getAccessRoots(Operation *op) {
    return aliasInfo->getAccessRoots(op);
  }

  BlockTokenSummary summarizeBlockTokens(Block *body) {
    BlockTokenSummary summary;
    body->walk([&](Operation *op) {
      if (isTokenOrderingOp(op) && memOpHasUserToken(op))
        summary.hasUserToken = true;
      if (isa<mlir::gpu::BarrierOp>(op))
        summary.hasDebugBarrier = true;
      if (isFenceOrSignal(op))
        summary.hasFenceOrSignal = true;
    });
    return summary;
  }

  bool handleUserTokenBailout(Operation *funcOp, Block *body,
                              const BlockTokenSummary &summary,
                              IRRewriter &rewriter) {
    if (!summary.hasUserToken)
      return false;

    // User-provided tokens on TKO memory/fence ops mean the caller owns the
    // whole token chain. Leave it untouched, but still consume gpu.barrier in
    // the body: downstream cuda_tile lowering expects this pass to erase debug
    // barriers, and the warning below promises that they are dropped.
    if (summary.hasDebugBarrier) {
      funcOp->emitWarning(
          "debug_barrier should not be added when memory tokens are "
          "added manually; debug_barrier ops will be erased.");
      SmallVector<Operation *> barriersToErase;
      body->walk([&](Operation *op) {
        if (isa<mlir::gpu::BarrierOp>(op))
          barriersToErase.push_back(op);
      });
      for (Operation *barrier : barriersToErase)
        rewriter.eraseOp(barrier);
    }
    return true;
  }

  bool hasAliasOrderingHazard(Block *body) {
    DenseMap<MemoryRootId, unsigned> opCount;
    DenseMap<MemoryRootId, unsigned> writeCount;
    body->walk([&](Operation *op) {
      if (!isMemOp(op))
        return;
      for (MemoryRootId root : getAccessRoots(op)) {
        opCount[root]++;
        if (isWriteMemOp(op))
          writeCount[root]++;
      }
    });
    for (auto &[root, count] : opCount) {
      if (count > 1 && writeCount[root] > 0)
        return true;
    }
    return false;
  }

  /// Find the store candidate for the parallel-store token optimization on
  /// `root`. This is a token-planning precondition, not a dependence proof:
  /// the loop body must have exactly one supported same-root store and no
  /// same-root reads, ordering fences/signals, or nested memory-like regions.
  Operation *findParallelStoreCandidate(cuda_tile::ForOp forOp,
                                        MemoryRootId root) {
    Operation *uniqueStore = nullptr;
    Block *body = forOp.getBody();
    for (Operation &child : *body) {
      if (&child == body->getTerminator())
        continue;
      if (isFenceOrSignal(&child))
        return nullptr;
      if (child.getNumRegions() > 0 && containsNestedMemLikeOp(&child))
        return nullptr;
      if (!isMemOp(&child))
        continue;
      if (!llvm::is_contained(getAccessRoots(&child), root))
        continue;
      if (!isWriteMemOp(&child))
        return nullptr;
      if (uniqueStore)
        return nullptr;
      if (!isa<cuda_tile::StoreViewTkoOp, cuda_tile::StorePtrTkoOp>(&child))
        return nullptr;
      uniqueStore = &child;
    }
    return uniqueStore;
  }

  /// Mark roots whose for-loop store can use the parallel-store token
  /// optimization. This is deliberately separate from the generic loop
  /// threading plan because the proof is cuda_tile.for specific: it asks
  /// TkoDependenceAnalysis whether one concrete store op is disjoint from
  /// itself across for-loop iterations.
  void markParallelStoreRoots(cuda_tile::ForOp forOp,
                              LoopTokenThreadingPlan &plan) {
    for (MemoryRootId root : plan.threadLastStore) {
      Operation *storeOp = findParallelStoreCandidate(forOp, root);
      if (storeOp && dependenceInfo->isLoopCarriedStoreDisjoint(forOp, storeOp))
        plan.parallelStoreRoots.insert(root);
    }
  }

  /// Produce a single Value equal to the JOIN of `toks` (deduped, null-pruned),
  /// creating a JoinTokensOp immediately before `beforeOp` if needed.
  Value joinOrSingle(ArrayRef<Value> toks, Operation *beforeOp,
                     IRRewriter &rewriter) {
    SmallVector<Value, 4> unique;
    llvm::SmallPtrSet<Value, 4> seen;
    for (Value v : toks) {
      if (!v)
        continue;
      if (seen.insert(v).second)
        unique.push_back(v);
    }
    if (unique.empty())
      return Value();
    if (unique.size() == 1)
      return unique.front();
    OpBuilder::InsertionGuard g(rewriter);
    rewriter.setInsertionPoint(beforeOp);
    return cuda_tile::JoinTokensOp::create(rewriter, beforeOp->getLoc(), unique)
        .getResult();
  }

  /// Produce a single Value equal to the JOIN of `toks` (deduped, null-pruned),
  /// creating a JoinTokensOp immediately after `afterOp` if needed.
  Value joinOrSingleAfter(ArrayRef<Value> toks, Operation *afterOp,
                          IRRewriter &rewriter) {
    SmallVector<Value, 4> unique;
    llvm::SmallPtrSet<Value, 4> seen;
    for (Value v : toks) {
      if (!v)
        continue;
      if (seen.insert(v).second)
        unique.push_back(v);
    }
    if (unique.empty())
      return Value();
    if (unique.size() == 1)
      return unique.front();
    OpBuilder::InsertionGuard g(rewriter);
    rewriter.setInsertionPointAfter(afterOp);
    return cuda_tile::JoinTokensOp::create(rewriter, afterOp->getLoc(), unique)
        .getResult();
  }

  void appendLastOpTokens(const TokenState &state,
                          SmallVectorImpl<Value> &tokens) {
    appendTokenMapValues(state.lastOp, tokens);
  }

  void appendLastStoreTokens(const TokenState &state,
                             SmallVectorImpl<Value> &tokens) {
    appendTokenMapValues(state.lastStore, tokens);
  }

  void appendTokenMapValues(const DenseMap<MemoryRootId, Value> &tokenMap,
                            SmallVectorImpl<Value> &tokens) {
    SmallVector<std::pair<MemoryRootId, Value>, 8> ordered;
    ordered.reserve(tokenMap.size());
    for (auto &entry : tokenMap)
      ordered.push_back({entry.first, entry.second});
    llvm::sort(ordered, [](const auto &lhs, const auto &rhs) {
      return lhs.first < rhs.first;
    });
    for (auto &entry : ordered)
      tokens.push_back(entry.second);
  }

  void appendAcquireToken(const TokenState &state,
                          SmallVectorImpl<Value> &tokens) {
    if (state.acquireToken)
      tokens.push_back(state.acquireToken);
  }

  /// GDC dependent-launch signals wait for producer writes and acquireToken.
  /// They do not consume read-only lastOp state; otherwise producer-side
  /// prologue/pipeline loads make the dependent launch wait for unrelated
  /// read-token accumulation and destroy PDL overlap.
  ///
  /// This is a deliberate tradeoff: modeling launch as a full release fence
  /// would also wait for prior read-only lastOp state, which fixes possible
  /// inter-kernel WAR cases (a dependent kernel writes an address the producer
  /// read before launch), but it over-serializes PDL producer read pipelines.
  /// Such WAR cases need an explicit stronger ordering point.
  Value buildDependentLaunchSignalInputToken(Operation *op,
                                             const TokenState &state,
                                             IRRewriter &rewriter) {
    SmallVector<Value, 8> deps;
    appendLastStoreTokens(state, deps);
    appendAcquireToken(state, deps);
    return joinOrSingle(deps, op, rewriter);
  }

  /// A total fence is both release-side and acquire-side. It needs a concrete
  /// output token even when there is no prior state, because later memory ops
  /// may need a token to express "after the barrier".
  Value buildTotalFenceOutputToken(Operation *op, const TokenState &state,
                                   IRRewriter &rewriter) {
    SmallVector<Value, 8> deps;
    appendLastOpTokens(state, deps);
    appendAcquireToken(state, deps);
    Value prior = joinOrSingle(deps, op, rewriter);
    if (prior)
      return prior;

    OpBuilder::InsertionGuard g(rewriter);
    rewriter.setInsertionPoint(op);
    return cuda_tile::MakeTokenOp::create(rewriter, op->getLoc()).getResult();
  }

  /// Publish a total-fence token to every root known in this function.
  /// This is what turns a barrier's single token into per-root ordering state.
  void broadcastFenceOutputToAllRoots(Value token, TokenState &state) {
    unsigned n = aliasInfo->getNumRoots();
    for (unsigned root = 0; root < n; root++)
      state.advanceStoreState(root, token);
  }

  /// Append a memToken operand to `op` and return its result token.
  /// Handles operandSegmentSizes for AttrSizedOperandSegments ops.
  /// `op` is taken by value — MLIR Op classes are light wrappers and
  /// `op->` forwards to the underlying Operation*.
  template <typename OpTy>
  Value updateMemOpWithToken(OpTy op, Value token, IRRewriter &rewriter) {
    assert(!memOpHasUserToken(op.getOperation()) &&
           "auto-tokenization should not rewrite already-tokenized TKO ops");
    SmallVector<Value> newOperands = llvm::to_vector(op->getOperands());
    if (token) {
      newOperands.push_back(token);
      if (auto segmentSizesAttr = op->getAttr("operandSegmentSizes")) {
        auto arrayAttr = cast<DenseI32ArrayAttr>(segmentSizesAttr);
        SmallVector<int32_t> newSegmentSizes =
            llvm::to_vector(arrayAttr.asArrayRef());
        assert(!newSegmentSizes.empty() &&
               "operandSegmentSizes must include the token segment");
        assert(newSegmentSizes.back() == 0 &&
               "expected an empty trailing token operand segment");
        newSegmentSizes.back() = 1;
        op->setAttr("operandSegmentSizes",
                    rewriter.getDenseI32ArrayAttr(newSegmentSizes));
      }
    }
    op->setOperands(newOperands);
    return op->getResults().back();
  }

  struct LoopTokenInputs {
    DenseMap<MemoryRootId, Value> lastOp;
    DenseMap<MemoryRootId, Value> lastStore;
    Value acquire;
  };

  struct TokenYieldInputs {
    DenseMap<MemoryRootId, Value> lastOp;
    DenseMap<MemoryRootId, Value> lastStore;
    Value acquire;
  };

  /// Return `token` if present, otherwise materialize a MakeTokenOp before
  /// `anchor`. Loop threading needs a concrete init value for every token
  /// slot, even when no previous op has initialized that TokenState field.
  Value ensureTokenBefore(Value token, Operation *anchor,
                          IRRewriter &rewriter) {
    if (token)
      return token;
    OpBuilder::InsertionGuard g(rewriter);
    rewriter.setInsertionPoint(anchor);
    return cuda_tile::MakeTokenOp::create(rewriter, anchor->getLoc())
        .getResult();
  }

  /// Materialize the initial token values for a loop threading plan. The
  /// returned maps are keyed by memory root and follow the same slot groups as
  /// LoopTokenThreadingPlan: lastOp, lastStore, and optional acquire.
  LoopTokenInputs materializeLoopTokenInputs(const LoopTokenThreadingPlan &plan,
                                             const TokenState &state,
                                             Operation *anchor,
                                             IRRewriter &rewriter) {
    LoopTokenInputs inputs;
    for (MemoryRootId c : plan.threadLastOp)
      inputs.lastOp[c] =
          ensureTokenBefore(state.lastOp.lookup(c), anchor, rewriter);
    for (MemoryRootId c : plan.threadLastStore)
      inputs.lastStore[c] =
          ensureTokenBefore(state.lastStore.lookup(c), anchor, rewriter);
    if (plan.threadAcquire)
      inputs.acquire = ensureTokenBefore(state.acquireToken, anchor, rewriter);
    return inputs;
  }

  void appendLoopTokenInputs(const LoopTokenThreadingPlan &plan,
                             const LoopTokenInputs &inputs,
                             SmallVectorImpl<Value> &values) {
    for (MemoryRootId c : plan.threadLastOp)
      values.push_back(inputs.lastOp.lookup(c));
    for (MemoryRootId c : plan.threadLastStore)
      values.push_back(inputs.lastStore.lookup(c));
    if (plan.threadAcquire)
      values.push_back(inputs.acquire);
  }

  /// Materialize fallback entry tokens for every token-yield slot. If a
  /// yielding branch/region did not advance a slot, it yields this entry token
  /// to keep the rewritten op's token result signature consistent.
  TokenYieldInputs materializeTokenYieldInputs(const TokenYieldPlan &plan,
                                               const TokenState &entryState,
                                               Operation *anchor,
                                               IRRewriter &rewriter) {
    TokenYieldInputs inputs;
    for (MemoryRootId c : plan.yieldLastOp)
      inputs.lastOp[c] =
          ensureTokenBefore(entryState.lastOp.lookup(c), anchor, rewriter);
    for (MemoryRootId c : plan.yieldLastStore)
      inputs.lastStore[c] =
          ensureTokenBefore(entryState.lastStore.lookup(c), anchor, rewriter);
    if (plan.yieldAcquire)
      inputs.acquire =
          ensureTokenBefore(entryState.acquireToken, anchor, rewriter);
    return inputs;
  }

  /// Build the extra operands a yielding branch/region must append to its
  /// terminator. Values advanced on that exit path win; otherwise use the
  /// materialized entry fallback for the slot.
  SmallVector<Value> buildTokenYieldOperands(const TokenYieldPlan &plan,
                                             const TokenYieldInputs &inputs,
                                             const TokenState &exitState) {
    SmallVector<Value> values;
    values.reserve(plan.numTokenSlots());
    for (MemoryRootId c : plan.yieldLastOp) {
      Value v = exitState.lastOp.lookup(c);
      values.push_back(v ? v : inputs.lastOp.lookup(c));
    }
    for (MemoryRootId c : plan.yieldLastStore) {
      Value v = exitState.lastStore.lookup(c);
      values.push_back(v ? v : inputs.lastStore.lookup(c));
    }
    if (plan.yieldAcquire) {
      values.push_back(exitState.acquireToken ? exitState.acquireToken
                                              : inputs.acquire);
    }
    return values;
  }

  void appendYieldOperands(Operation *term, ArrayRef<Value> extras) {
    SmallVector<Value> ops = llvm::to_vector(term->getOperands());
    ops.append(extras.begin(), extras.end());
    term->setOperands(ops);
  }

  SmallVector<Type> resultTypesWithTokenResults(TypeRange originalResultTypes,
                                                size_t numTokenResults,
                                                MLIRContext *context) {
    SmallVector<Type> resultTypes = llvm::to_vector(originalResultTypes);
    auto tokTy = cuda_tile::TokenType::get(context);
    for (size_t k = 0; k < numTokenResults; k++)
      resultTypes.push_back(tokTy);
    return resultTypes;
  }

  void replaceWithOriginalResults(Operation *oldOp, Operation *newOp,
                                  IRRewriter &rewriter) {
    SmallVector<Value> replacementResults;
    for (size_t k = 0; k < oldOp->getNumResults(); k++)
      replacementResults.push_back(newOp->getResult(k));
    rewriter.replaceOp(oldOp, replacementResults);
  }

  /// Install the threaded token block arguments into the loop body entry
  /// state. Argument order follows LoopTokenThreadingPlan slot order.
  void installLoopBodyTokenArgs(const LoopTokenThreadingPlan &plan, Block *body,
                                size_t firstTokArg, TokenState &bodyEntry) {
    size_t i = 0;
    for (MemoryRootId c : plan.threadLastOp)
      bodyEntry.lastOp[c] = body->getArgument(firstTokArg + i++);
    for (MemoryRootId c : plan.threadLastStore)
      bodyEntry.lastStore[c] = body->getArgument(firstTokArg + i++);
    if (plan.threadAcquire)
      bodyEntry.acquireToken = body->getArgument(firstTokArg + i++);
  }

  /// Redirect in-body uses of materialized loop init tokens to the rewritten
  /// loop's token block arguments. Outside-loop uses keep the init token, since
  /// they pre-date the loop.
  void redirectLoopInitTokenUses(const LoopTokenThreadingPlan &plan,
                                 const LoopTokenInputs &inputs,
                                 Operation *rewrittenLoop, Block *body,
                                 size_t firstTokArg) {
    auto redirect = [&](Value oldTok, Value newArg) {
      oldTok.replaceUsesWithIf(newArg, [&](OpOperand &use) {
        return rewrittenLoop->isProperAncestor(use.getOwner());
      });
    };

    size_t i = 0;
    for (MemoryRootId c : plan.threadLastOp)
      redirect(inputs.lastOp.lookup(c), body->getArgument(firstTokArg + i++));
    for (MemoryRootId c : plan.threadLastStore)
      redirect(inputs.lastStore.lookup(c),
               body->getArgument(firstTokArg + i++));
    if (plan.threadAcquire)
      redirect(inputs.acquire, body->getArgument(firstTokArg + i++));
  }

  /// Build the entry token state for a rewritten loop body. Each threaded slot
  /// is bound to its appended block argument, and uses of materialized init
  /// tokens inside the loop are redirected to those block arguments. Uses
  /// outside the loop keep referring to the pre-loop init tokens.
  TokenState initializeLoopBodyTokenState(const LoopTokenThreadingPlan &plan,
                                          const LoopTokenInputs &inputs,
                                          Operation *rewrittenLoop, Block *body,
                                          size_t firstTokArg,
                                          const TokenState &outerState) {
    TokenState bodyEntry = outerState;
    bodyEntry.clearAdvancedMarkers();
    installLoopBodyTokenArgs(plan, body, firstTokArg, bodyEntry);
    redirectLoopInitTokenUses(plan, inputs, rewrittenLoop, body, firstTokArg);
    return bodyEntry;
  }

  SmallVector<DownstreamTokenDemand>
  computeDownstreamTokenDemand(ArrayRef<Operation *> ops,
                               DownstreamTokenDemand inheritedDownstream) {
    size_t n = ops.size();
    SmallVector<DownstreamTokenDemand> downstream(n);
    DownstreamTokenDemand acc = inheritedDownstream;
    for (int i = (int)n - 1; i >= 0; i--) {
      // downstream[i] is strictly after ops[i], including the enclosing
      // continuation passed in through inheritedDownstream.
      downstream[i] = acc;
      Operation *op = ops[i];
      if (isReleaseFence(op))
        acc.hasLaterReleaseFence = true;
      if (isWriteMemOp(op)) {
        for (MemoryRootId root : getAccessRoots(op))
          acc.rootsConsumedByLaterWrite.insert(root);
      }
      // Nested regions contribute through their pre-computed recursive summary.
      for (Region &region : op->getRegions()) {
        auto it = regionSummaries.find(&region);
        if (it == regionSummaries.end())
          continue;
        const RegionMemorySummary &summary = it->second;
        if (summary.hasReleaseFence)
          acc.hasLaterReleaseFence = true;
        for (auto &[root, eff] : summary.perRoot) {
          if (eff == MemEffect::Store)
            acc.rootsConsumedByLaterWrite.insert(root);
        }
      }
    }
    return downstream;
  }

  /// If a loop body contains a total fence (gpu.barrier), propagate the
  /// post-loop acquireToken to every memory root the loop did not thread
  /// directly. The fence's in-loop fan-out touched those roots, and post-loop
  /// code on them must happens-after the fence. Threaded roots already carry
  /// their own per-iteration chain via dedicated token slots, so do not
  /// overwrite them.
  void propagateLoopTotalFenceFanout(const RegionMemorySummary &summary,
                                     ArrayRef<MemoryRootId> threadLastOp,
                                     ArrayRef<MemoryRootId> threadLastStore,
                                     TokenState &out) {
    if (!summary.hasTotalFence)
      return;

    assert(out.acquireToken &&
           "loop with a total fence should thread acquireToken");

    llvm::SmallSet<MemoryRootId, 8> threaded;
    for (MemoryRootId c : threadLastOp)
      threaded.insert(c);
    for (MemoryRootId c : threadLastStore)
      threaded.insert(c);
    unsigned n = aliasInfo->getNumRoots();
    for (unsigned root = 0; root < n; root++) {
      if (threaded.contains(root))
        continue;
      out.advanceStoreState(root, out.acquireToken);
    }
  }

  /// Build the post-loop token state from the rewritten loop's appended token
  /// results. The incoming state is preserved first so enclosing branch scopes
  /// still see advances that happened before this loop. Result order follows
  /// LoopTokenThreadingPlan slot order: lastOp slots, lastStore slots, then
  /// optional acquireToken.
  TokenState buildPostLoopTokenState(const LoopTokenThreadingPlan &plan,
                                     const RegionMemorySummary &summary,
                                     Operation *rewrittenLoop,
                                     size_t firstTokResult,
                                     const TokenState &preLoopState) {
    TokenState out = preLoopState;
    size_t i = firstTokResult;
    for (MemoryRootId c : plan.threadLastOp)
      out.advanceLastOp(c, rewrittenLoop->getResult(i++));
    for (MemoryRootId c : plan.threadLastStore)
      out.advanceLastStore(c, rewrittenLoop->getResult(i++));
    if (plan.threadAcquire)
      out.advanceAcquire(rewrittenLoop->getResult(i++));
    assert(i == firstTokResult + plan.numTokenSlots() &&
           "loop token result count must match threading plan");

    propagateLoopTotalFenceFanout(summary, plan.threadLastOp,
                                  plan.threadLastStore, out);
    return out;
  }

  /// Build the post-op token state from token results produced by a rewritten
  /// yield-style region op, such as cuda_tile.if. Preserve the incoming state
  /// so enclosing branch scopes still see advances that happened before this
  /// op.
  TokenState buildPostTokenYieldState(const TokenYieldPlan &plan,
                                      Operation *rewrittenOp,
                                      size_t firstTokResult,
                                      const TokenState &preOpState) {
    TokenState out = preOpState;
    size_t i = firstTokResult;
    for (MemoryRootId c : plan.yieldLastOp)
      out.advanceLastOp(c, rewrittenOp->getResult(i++));
    for (MemoryRootId c : plan.yieldLastStore)
      out.advanceLastStore(c, rewrittenOp->getResult(i++));
    if (plan.yieldAcquire)
      out.advanceAcquire(rewrittenOp->getResult(i++));
    assert(i == firstTokResult + plan.numTokenSlots() &&
           "token result count must match yield plan");
    return out;
  }

  /// Pick the token a rewritten loop terminator should yield for a threaded
  /// slot. If an exit path did not advance the slot, yield the entry token so
  /// every Break/Continue matches the loop's expanded token signature.
  Value pickLoopTermToken(const TokenState &exitState,
                          const TokenState &bodyEntry, MemoryRootId c,
                          bool isStore) {
    Value v =
        isStore ? exitState.lastStore.lookup(c) : exitState.lastOp.lookup(c);
    if (v)
      return v;
    return isStore ? bodyEntry.lastStore.lookup(c) : bodyEntry.lastOp.lookup(c);
  }

  void appendLoopTermTokenOperands(const LoopTokenThreadingPlan &plan,
                                   const TokenState &bodyEntry, Operation *term,
                                   const TokenState &exitState) {
    // Only loop exits carry the rewritten loop token signature.
    if (!isa<cuda_tile::BreakOp, cuda_tile::ContinueOp>(term))
      return;

    SmallVector<Value> ops = llvm::to_vector(term->getOperands());
    for (MemoryRootId c : plan.threadLastOp)
      ops.push_back(pickLoopTermToken(exitState, bodyEntry, c, false));
    for (MemoryRootId c : plan.threadLastStore)
      ops.push_back(pickLoopTermToken(exitState, bodyEntry, c, true));
    if (plan.threadAcquire) {
      Value acquire = exitState.acquireToken ? exitState.acquireToken
                                             : bodyEntry.acquireToken;
      ops.push_back(acquire);
    }
    term->setOperands(ops);
  }

  /// Walk a rewritten loop body from its threaded entry token state and patch
  /// every loop exit terminator to yield the threaded token slots. This is the
  /// shared "body token state escapes the loop" step for cuda_tile.for and
  /// cuda_tile.loop after each handler has performed its op-specific signature
  /// rewrite.
  TokenState
  walkLoopBodyAndPatchExits(const LoopTokenThreadingPlan &plan, Block *body,
                            const TokenState &bodyEntry, IRRewriter &rewriter,
                            const DownstreamTokenDemand &downstreamDemand) {
    OpToStates bodyTermOps;
    TokenState bodyExit = addMemTokenForBlock(body, bodyEntry, rewriter,
                                              &bodyTermOps, downstreamDemand);
    appendLoopTermTokenOperands(plan, bodyEntry, body->getTerminator(),
                                bodyExit);
    for (auto &pair : bodyTermOps)
      appendLoopTermTokenOperands(plan, bodyEntry, pair.first, pair.second);
    return bodyExit;
  }

  /// Classification of an if branch terminator for token-yield rewriting.
  /// Loop exits are forwarded to the enclosing loop, returns stop control flow,
  /// and fallthrough branches participate in the if's token results. An empty
  /// else region is modeled as an implicit fallthrough yield with no
  /// branch-local token advances.
  struct IfBranchControlFlow {
    Operation *thenTerminator = nullptr;
    Operation *elseTerminator = nullptr;
    bool thenIsLoopExit = false;
    bool elseIsLoopExit = false;
    bool thenFallsThrough = false;
    bool elseFallsThrough = false;

    bool anyBranchFallsThrough() const {
      return thenFallsThrough || elseFallsThrough;
    }

    bool elseHasExplicitTerminator() const { return elseTerminator != nullptr; }
  };

  bool isLoopExitTerminator(Operation *op) {
    return op && isa<cuda_tile::BreakOp, cuda_tile::ContinueOp>(op);
  }

  bool isReturnTerminator(Operation *op) {
    return op && isa<cuda_tile::ReturnOp>(op);
  }

  IfBranchControlFlow classifyIfBranchControlFlow(cuda_tile::IfOp ifOp) {
    IfBranchControlFlow flow;
    flow.thenTerminator = ifOp.getThenTerminator();
    flow.elseTerminator =
        ifOp.getElseRegion().empty() ? nullptr : ifOp.getElseTerminator();

    flow.thenIsLoopExit = isLoopExitTerminator(flow.thenTerminator);
    flow.elseIsLoopExit = isLoopExitTerminator(flow.elseTerminator);

    // A branch falls through to post-if code when its terminator is a
    // cuda_tile.yield-like terminator. Empty else is the MLIR implicit-yield
    // case: it falls through but contributes no branch-local state.
    flow.thenFallsThrough =
        !flow.thenIsLoopExit && !isReturnTerminator(flow.thenTerminator);
    flow.elseFallsThrough =
        !flow.elseTerminator ||
        (!flow.elseIsLoopExit && !isReturnTerminator(flow.elseTerminator));
    return flow;
  }

  void materializeEmptyElseYield(cuda_tile::IfOp ifOp, IRRewriter &rewriter) {
    if (!ifOp.getElseRegion().empty())
      return;
    rewriter.createBlock(&ifOp.getElseRegion());
    rewriter.setInsertionPointToEnd(ifOp.getElseBlock());
    cuda_tile::YieldOp::create(rewriter, ifOp.getLoc());
  }

  /// Build the input token for one memory op. Reads wait for the latest store
  /// on each touched root (RAW). Writes and atomics wait for the latest op on
  /// each touched root (WAW + WAR), except for parallel-store roots where the
  /// loop body deliberately uses a loop-invariant input token. All memory ops
  /// also wait for the current acquire token, if one exists.
  Value buildMemOpInputToken(Operation *op, ArrayRef<MemoryRootId> roots,
                             const TokenState &state, IRRewriter &rewriter) {
    SmallVector<Value, 4> deps;
    if (isWriteMemOp(op)) {
      for (MemoryRootId root : roots) {
        if (auto it = state.parallelStoreInputToken.find(root);
            it != state.parallelStoreInputToken.end()) {
          deps.push_back(it->second);
          continue;
        }
        if (auto it = state.lastOp.find(root); it != state.lastOp.end())
          deps.push_back(it->second);
      }
    } else {
      for (MemoryRootId root : roots)
        if (auto it = state.lastStore.find(root); it != state.lastStore.end())
          deps.push_back(it->second);
    }
    if (state.acquireToken)
      deps.push_back(state.acquireToken);

    return joinOrSingle(deps, op, rewriter);
  }

  /// Attach the computed input token to a concrete TKO memory op and return the
  /// op's output token.
  Value attachMemOpInputToken(Operation *op, Value inputTok,
                              IRRewriter &rewriter) {
    return llvm::TypeSwitch<Operation *, Value>(op)
        .Case<cuda_tile::LoadPtrTkoOp>(
            [&](auto o) { return updateMemOpWithToken(o, inputTok, rewriter); })
        .Case<cuda_tile::StorePtrTkoOp>(
            [&](auto o) { return updateMemOpWithToken(o, inputTok, rewriter); })
        .Case<cuda_tile::LoadViewTkoOp>(
            [&](auto o) { return updateMemOpWithToken(o, inputTok, rewriter); })
        .Case<cuda_tile::StoreViewTkoOp>(
            [&](auto o) { return updateMemOpWithToken(o, inputTok, rewriter); })
        .Case<cuda_tile::AtomicRMWTkoOp>(
            [&](auto o) { return updateMemOpWithToken(o, inputTok, rewriter); })
        .Case<cuda_tile::AtomicCASTkoOp>(
            [&](auto o) { return updateMemOpWithToken(o, inputTok, rewriter); })
        .Case<cuda_tile::AtomicRedViewTkoOp>(
            [&](auto o) { return updateMemOpWithToken(o, inputTok, rewriter); })
        .Default(Value());
  }

  /// Publish a write/atomic output token into TokenState. Normal writes replace
  /// both lastOp and lastStore for each root. Parallel-store roots accumulate
  /// output tokens into lastStore/lastOp so post-loop users still wait for all
  /// stores even though loop iterations do not WAW-chain through the store's
  /// input token.
  void publishWriteMemOpOutputToken(Operation *op, ArrayRef<MemoryRootId> roots,
                                    Value outputTok, TokenState &state,
                                    IRRewriter &rewriter) {
    for (MemoryRootId root : roots) {
      Value nextTok = outputTok;
      if (state.parallelStoreInputToken.find(root) !=
          state.parallelStoreInputToken.end()) {
        SmallVector<Value, 2> accDeps{state.lastStore.lookup(root), outputTok};
        nextTok = joinOrSingleAfter(accDeps, op, rewriter);
      }
      state.advanceStoreState(root, nextTok);
    }
  }

  /// Publish a read output token into TokenState. The read advances lastOp but
  /// not lastStore. If the root already has a lastOp token, eagerly join the
  /// read output after the op so a future write can WAR against all pending
  /// reads through one token edge.
  void publishReadMemOpOutputToken(Operation *op, ArrayRef<MemoryRootId> roots,
                                   Value outputTok, TokenState &state,
                                   IRRewriter &rewriter) {
    for (MemoryRootId root : roots) {
      Value prior = state.lastOp.lookup(root);
      if (!prior) {
        state.advanceLastOp(root, outputTok);
        continue;
      }

      OpBuilder::InsertionGuard g(rewriter);
      rewriter.setInsertionPointAfter(op);
      auto joinOp = cuda_tile::JoinTokensOp::create(
          rewriter, op->getLoc(), ValueRange{prior, outputTok});
      state.advanceLastOp(root, joinOp.getResult());
    }
  }

  // -------------------------------------------------------------------------
  // Per-op emitters
  // -------------------------------------------------------------------------

  void emitMemOp(Operation *op, TokenState &state, IRRewriter &rewriter) {
    SmallVector<MemoryRootId, 2> roots = getAccessRoots(op);
    Value inputTok = buildMemOpInputToken(op, roots, state, rewriter);
    Value outputTok = attachMemOpInputToken(op, inputTok, rewriter);
    if (isWriteMemOp(op))
      publishWriteMemOpOutputToken(op, roots, outputTok, state, rewriter);
    else
      publishReadMemOpOutputToken(op, roots, outputTok, state, rewriter);
  }

  /// Acquire fence: publish the fence output as acquireToken for subsequent
  /// memory ops. Pure acquire does not consume prior local memory state.
  void emitAcquireFence(Operation *op, TokenState &state,
                        IRRewriter &rewriter) {
    auto waitOp = cast<cuda_tile::GdcWaitTkoOp>(op);
    Value outputTok = updateMemOpWithToken(waitOp, Value(), rewriter);

    state.advanceAcquire(outputTok);
  }

  /// Dependent launch signal: consume producer writes. The result token is not
  /// fed back into TokenState, so later same-kernel memory ops are not forced
  /// to wait for this signal.
  void emitDependentLaunchSignal(Operation *op, TokenState &state,
                                 IRRewriter &rewriter) {
    Value inputTok = buildDependentLaunchSignalInputToken(op, state, rewriter);
    auto launchOp = cast<cuda_tile::GdcLaunchDependentsTkoOp>(op);
    (void)updateMemOpWithToken(launchOp, inputTok, rewriter);
  }

  /// gpu.barrier: total sync. JOIN all prior state and install the joined
  /// token as lastOp / lastStore for every memory root discovered in this
  /// function, plus as acquireToken. After barrier, every subsequent op
  /// chains after all prior ops on every root.
  void emitBarrier(Operation *op, TokenState &state, IRRewriter &rewriter) {
    Value joined = buildTotalFenceOutputToken(op, state, rewriter);
    broadcastFenceOutputToAllRoots(joined, state);
    state.advanceAcquire(joined);

    // Erase the gpu.barrier op — its semantics are now carried by the token
    // graph.
    rewriter.eraseOp(op);
  }

  bool emitMemoryOrFenceOp(Operation *op, TokenState &state,
                           IRRewriter &rewriter) {
    if (isMemOp(op)) {
      emitMemOp(op, state, rewriter);
      return true;
    }

    if (isTotalFence(op)) {
      emitBarrier(op, state, rewriter);
      return true;
    }
    if (isAcquireFence(op)) {
      emitAcquireFence(op, state, rewriter);
      return true;
    }
    if (isDependentLaunchSignal(op)) {
      emitDependentLaunchSignal(op, state, rewriter);
      return true;
    }
    return false;
  }

  // -------------------------------------------------------------------------
  // Control flow handlers
  // -------------------------------------------------------------------------

  /// Rebuild cuda_tile.for with extra iter_args for token state that must be
  /// carried from one iteration to the next and returned after the loop.
  TokenState handleForOp(cuda_tile::ForOp forOp, TokenState state,
                         IRRewriter &rewriter,
                         const DownstreamTokenDemand &downstreamDemand) {
    // Keep markers from earlier ops in the enclosing scope; the caller still
    // needs them after this handler returns. Loop-body markers start fresh when
    // initializeLoopBodyTokenState builds the body entry state.
    RegionMemorySummary &summary = regionSummaries[&forOp.getRegion()];

    LoopTokenThreadingPlan plan =
        TokenOrderPlanner::planLoopThreading(summary, downstreamDemand);
    markParallelStoreRoots(forOp, plan);

    if (plan.empty()) {
      // Body touches nothing that needs threading. Still walk the body so
      // any nested mem ops get tokens, but no iter_arg surgery. Pass
      // downstream demand so nested loops can still see post-loop consumers.
      TokenState bodyState = addMemTokenForBlock(
          forOp.getBody(), state, rewriter, nullptr, downstreamDemand);
      // Discard body token state — without yielded tokens nothing escapes.
      (void)bodyState;
      return state;
    }

    LoopTokenInputs initTokens =
        materializeLoopTokenInputs(plan, state, forOp.getOperation(), rewriter);

    // Build new init-args vector.
    SmallVector<Value> newInitArgs = llvm::to_vector(forOp.getInitValues());
    appendLoopTokenInputs(plan, initTokens, newInitArgs);

    // Create the new ForOp with expanded iter_args.
    rewriter.setInsertionPointAfter(forOp);
    auto newForOp = cuda_tile::ForOp::create(
        rewriter, forOp.getLoc(), forOp.getLowerBound(), forOp.getUpperBound(),
        forOp.getStep(), newInitArgs, /*bodyBuilder=*/nullptr,
        forOp.getUnsignedCmp());
    newForOp->setAttrs(forOp->getAttrs());

    // Move the old body block into the new op's region and preserve
    // existing block-arg mapping for original iter_args. Snapshot arg
    // count BEFORE mergeBlocks (after merge, forOp's body is empty).
    unsigned firstTokArg = forOp.getBody()->getNumArguments();
    Block *newBlock = &newForOp.getRegion().front();
    SmallVector<Value> blockArgs;
    for (auto arg : forOp.getBody()->getArguments())
      blockArgs.push_back(newBlock->getArgument(arg.getArgNumber()));
    rewriter.mergeBlocks(forOp.getBody(), newBlock, blockArgs);

    TokenState bodyEntry =
        initializeLoopBodyTokenState(plan, initTokens, newForOp.getOperation(),
                                     newBlock, firstTokArg, state);
    for (MemoryRootId root : plan.parallelStoreRoots)
      bodyEntry.parallelStoreInputToken[root] = initTokens.lastOp.lookup(root);

    (void)walkLoopBodyAndPatchExits(plan, newBlock, bodyEntry, rewriter,
                                    downstreamDemand);

    // Build post-loop state: result slots become the new lastOp/lastStore.
    // Start from the incoming state (which preserves the caller's
    // advanced* markers — those advances happened before this for-op
    // and must still be visible to the enclosing scope), then add the
    // loop's threaded advances on top.
    TokenState out = buildPostLoopTokenState(
        plan, summary, newForOp.getOperation(), forOp.getNumResults(), state);

    // Replace original results (non-token) with the new ones.
    replaceWithOriginalResults(forOp.getOperation(), newForOp.getOperation(),
                               rewriter);

    return out;
  }

  /// Handle cuda_tile.if. Walk both branches with the entry state; for each
  /// (root, role) that was advanced in either branch, add a token result
  /// to the if-op and have each branch yield its own token (or the entry
  /// value if not advanced). Under scf.while body where the if-op has
  /// Break/Continue terminators, collect branches into `termOps`.
  TokenState handleIfOp(cuda_tile::IfOp ifOp, TokenState state,
                        IRRewriter &rewriter, OpToStates *termOps,
                        const DownstreamTokenDemand &downstreamDemand) {
    // Preserve incoming advanced* markers. They carry prior advances from the
    // enclosing branch, and the caller needs to see them in this handler's
    // return value.

    // Walk both branches with cleared markers so each exit state reports only
    // changes made inside that branch. The yield plan unions those
    // branch-local changes into the if's token results.
    TokenState thenEntry = state;
    TokenState elseEntry = state;
    thenEntry.clearAdvancedMarkers();
    elseEntry.clearAdvancedMarkers();

    TokenState thenExit = addMemTokenForBlock(
        ifOp.getThenBlock(), thenEntry, rewriter, termOps, downstreamDemand);
    TokenState elseExit;
    if (!ifOp.getElseRegion().empty())
      elseExit = addMemTokenForBlock(ifOp.getElseBlock(), elseEntry, rewriter,
                                     termOps, downstreamDemand);

    IfBranchControlFlow branchFlow = classifyIfBranchControlFlow(ifOp);

    // Step 1: collect loop-exit terminators into termOps for the
    // enclosing loop to extend. Must be done before the yield rewrite
    // runs, because the yield rewrite only touches yielding branches.
    if (termOps) {
      if (branchFlow.thenIsLoopExit)
        termOps->push_back({branchFlow.thenTerminator, thenExit});
      if (branchFlow.elseIsLoopExit)
        termOps->push_back({branchFlow.elseTerminator, elseExit});
    }

    // Step 2: if neither branch yields through, post-if code is
    // unreachable. Still patch an empty else with an empty yield if
    // needed so the if op itself is well-formed.
    if (!branchFlow.anyBranchFallsThrough()) {
      materializeEmptyElseYield(ifOp, rewriter);
      return TokenState();
    }

    // Union of (root, role) advanced in the yielding branches only. A
    // loop-exit or return branch doesn't carry its state to post-if. An empty
    // else yields through but has no branch-local advances to merge.
    TokenYieldPlan plan = TokenOrderPlanner::planBranchYieldTokens(
        thenExit, branchFlow.thenFallsThrough, elseExit,
        branchFlow.elseFallsThrough && branchFlow.elseHasExplicitTerminator());

    if (plan.empty())
      return state;

    TokenYieldInputs yieldInputs =
        materializeTokenYieldInputs(plan, state, ifOp.getOperation(), rewriter);
    SmallVector<Value> thenYields, elseYields;
    if (branchFlow.thenFallsThrough)
      thenYields = buildTokenYieldOperands(plan, yieldInputs, thenExit);
    if (branchFlow.elseFallsThrough)
      elseYields = buildTokenYieldOperands(plan, yieldInputs, elseExit);

    // Fill empty else with an empty yield so we have a yield terminator
    // to extend. (A non-empty else was classified above; if it yields
    // through, its terminator is already a YieldOp.)
    materializeEmptyElseYield(ifOp, rewriter);

    // Append yield operands to yielding-branch terminators only. Non-
    // yielding branches (loop-exit, return) keep their original operand
    // count — the verifier only type-checks YieldOp terminators against
    // the if's results, so a mix of yield-in-one-branch + non-yield-in-
    // other is valid.
    if (branchFlow.thenFallsThrough)
      appendYieldOperands(ifOp.getThenTerminator(), thenYields);
    if (branchFlow.elseFallsThrough)
      appendYieldOperands(ifOp.getElseTerminator(), elseYields);

    // Rebuild the IfOp with expanded result types. Count comes from the
    // union (not per-branch) since a return branch yields nothing but
    // still contributes to the if's output via the other branch.
    rewriter.setInsertionPointAfter(ifOp);
    SmallVector<Type> resultTypes = resultTypesWithTokenResults(
        ifOp.getResultTypes(), plan.numTokenSlots(), rewriter.getContext());

    auto newIfOp =
        cuda_tile::IfOp::create(rewriter, ifOp.getLoc(), resultTypes,
                                ifOp.getCondition(), ifOp->getAttrs());
    rewriter.inlineRegionBefore(ifOp.getThenRegion(), newIfOp.getThenRegion(),
                                newIfOp.getThenRegion().begin());
    rewriter.inlineRegionBefore(ifOp.getElseRegion(), newIfOp.getElseRegion(),
                                newIfOp.getElseRegion().begin());

    TokenState out = buildPostTokenYieldState(plan, newIfOp.getOperation(),
                                              ifOp.getNumResults(), state);

    // Replace old IfOp with new (non-token results).
    replaceWithOriginalResults(ifOp.getOperation(), newIfOp.getOperation(),
                               rewriter);

    return out;
  }

  /// Handle cuda_tile::LoopOp (scf.while analog). Like handleForOp but
  /// threads via operand + block-arg (LoopOp has no iter_args; initial
  /// values come from operands) and must also extend any Break/Continue
  /// terminators collected from the body.
  TokenState handleLoopOp(cuda_tile::LoopOp loopOp, TokenState state,
                          IRRewriter &rewriter,
                          const DownstreamTokenDemand &downstreamDemand) {
    // See handleForOp: preserve incoming advanced* markers.
    RegionMemorySummary &summary = regionSummaries[&loopOp.getRegion()];

    LoopTokenThreadingPlan plan =
        TokenOrderPlanner::planLoopThreading(summary, downstreamDemand);

    if (plan.empty()) {
      OpToStates bodyTermOps;
      TokenState bodyEntry = state;
      bodyEntry.clearAdvancedMarkers();
      TokenState bodyExit =
          addMemTokenForBlock(loopOp.getBody(), bodyEntry, rewriter,
                              &bodyTermOps, downstreamDemand);
      (void)bodyExit;
      return state;
    }

    LoopTokenInputs initTokens = materializeLoopTokenInputs(
        plan, state, loopOp.getOperation(), rewriter);

    // New operands = old operands + new token init values.
    SmallVector<Value> newOperands = llvm::to_vector(loopOp->getOperands());
    appendLoopTokenInputs(plan, initTokens, newOperands);

    size_t numTokResults = plan.numTokenSlots();
    auto tokTy = cuda_tile::TokenType::get(rewriter.getContext());
    SmallVector<Type> resultTypes = resultTypesWithTokenResults(
        loopOp.getResultTypes(), numTokResults, rewriter.getContext());

    rewriter.setInsertionPointAfter(loopOp);
    auto newLoopOp =
        cuda_tile::LoopOp::create(rewriter, loopOp.getLoc(), resultTypes,
                                  newOperands, loopOp->getAttrs());

    // Build new body block with added token block-args.
    Block *newBlock = rewriter.createBlock(&newLoopOp.getRegion());
    for (Type type : loopOp.getBody()->getArgumentTypes())
      newBlock->addArgument(type, loopOp.getLoc());
    for (size_t k = 0; k < numTokResults; k++)
      newBlock->addArgument(tokTy, loopOp.getLoc());

    size_t firstTokArg = loopOp.getBody()->getNumArguments();
    SmallVector<Value> blockArgs;
    for (auto arg : loopOp.getBody()->getArguments())
      blockArgs.push_back(newBlock->getArgument(arg.getArgNumber()));
    rewriter.mergeBlocks(loopOp.getBody(), newBlock, blockArgs);

    TokenState bodyEntry =
        initializeLoopBodyTokenState(plan, initTokens, newLoopOp.getOperation(),
                                     newBlock, firstTokArg, state);

    (void)walkLoopBodyAndPatchExits(plan, newBlock, bodyEntry, rewriter,
                                    downstreamDemand);

    // Post-loop state. Start from incoming state (preserves caller's
    // advanced* markers for pre-loop advances in the same branch),
    // then add this loop's threaded advances.
    TokenState out = buildPostLoopTokenState(
        plan, summary, newLoopOp.getOperation(), loopOp.getNumResults(), state);

    // Replace original non-token results.
    replaceWithOriginalResults(loopOp.getOperation(), newLoopOp.getOperation(),
                               rewriter);

    return out;
  }

  LogicalResult diagnoseUnsupportedRegionOp(Operation *op) {
    if (op->getNumRegions() == 0 ||
        op->hasTrait<mlir::OpTrait::IsIsolatedFromAbove>())
      return success();
    if (!containsNestedMemLikeOp(op))
      return success();

    op->emitError() << "AutoGenMemoryToken: cannot tokenize memory ops inside "
                       "unsupported region op '"
                    << op->getName().getStringRef()
                    << "'. Add a specific handler or "
                       "move the memory ops outside the region.";
    signalPassFailure();
    return failure();
  }

  // -------------------------------------------------------------------------
  // Block walker
  // -------------------------------------------------------------------------

  TokenState
  addMemTokenForBlock(Block *block, TokenState state, IRRewriter &rewriter,
                      OpToStates *termOps = nullptr,
                      const DownstreamTokenDemand &downstreamDemand = {}) {
    // Preserve incoming advanced* markers. Callers pass a state that's either
    // freshly cleared (branch / body entries) or a continuation from a
    // recursive call into a nested region. In the latter case the incoming
    // markers describe advances already made in the enclosing block.
    if (!block)
      return state;

    // Collect the original top-level ops before rewriting. Handlers may erase
    // or replace the current op, and suffix token-use analysis must see the
    // original sibling order rather than newly inserted token ops.
    SmallVector<Operation *> ops;
    for (Operation &op : *block)
      ops.push_back(&op);

    auto downstreamByOp = computeDownstreamTokenDemand(ops, downstreamDemand);

    for (size_t idx = 0; idx < ops.size(); idx++) {
      Operation *op = ops[idx];

      DownstreamTokenDemand opDownstream = downstreamByOp[idx];

      if (auto forOp = dyn_cast<cuda_tile::ForOp>(op)) {
        state = handleForOp(forOp, std::move(state), rewriter, opDownstream);
        continue;
      }
      if (auto ifOp = dyn_cast<cuda_tile::IfOp>(op)) {
        state =
            handleIfOp(ifOp, std::move(state), rewriter, termOps, opDownstream);
        continue;
      }
      if (auto loopOp = dyn_cast<cuda_tile::LoopOp>(op)) {
        state = handleLoopOp(loopOp, std::move(state), rewriter, opDownstream);
        continue;
      }
      if (emitMemoryOrFenceOp(op, state, rewriter))
        continue;
      // Other ops (arith, control flow terminators, etc.) without
      // regions — ignore. An op with regions that we don't specifically
      // recognize cannot be safely tokenized when their result count is fixed
      // or their regions lack terminators. We cannot extend their signatures
      // with token results
      // to propagate in-region mem-op advances to post-region code.
      // Silently recursing and discarding the in-region state would
      // leave a post-region store unordered with an in-region load on
      // the same root (the store's input token would still be the
      // pre-region value), which is worse than failing loudly.
      // Diagnose and abort.
      if (failed(diagnoseUnsupportedRegionOp(op)))
        return state;
    }

    return state;
  }

  // -------------------------------------------------------------------------
  // Pre-pass: region memory summaries
  // -------------------------------------------------------------------------

  void computeRegionSummaries(Operation *funcLikeOp) {
    // Post-order walk so nested regions are filled before their parents.
    funcLikeOp->walk<WalkOrder::PostOrder>([&](Operation *op) {
      for (Region &region : op->getRegions()) {
        RegionMemorySummary &info = regionSummaries[&region];
        for (Block &block : region) {
          for (Operation &child : block) {
            if (isMemOp(&child)) {
              MemEffect eff =
                  isWriteMemOp(&child) ? MemEffect::Store : MemEffect::Load;
              for (MemoryRootId root : getAccessRoots(&child))
                info.perRoot[root] = maxEffect(info.perRoot[root], eff);
            }
            if (isAcquireFence(&child))
              info.hasAcquireFence = true;
            if (isReleaseFence(&child))
              info.hasReleaseFence = true;
            if (isTotalFence(&child))
              info.hasTotalFence = true;
            // Propagate from nested regions.
            for (Region &sub : child.getRegions()) {
              auto it = regionSummaries.find(&sub);
              if (it != regionSummaries.end()) {
                for (auto &[c, eff] : it->second.perRoot)
                  info.perRoot[c] = maxEffect(info.perRoot[c], eff);
                if (it->second.hasAcquireFence)
                  info.hasAcquireFence = true;
                if (it->second.hasReleaseFence)
                  info.hasReleaseFence = true;
                if (it->second.hasTotalFence)
                  info.hasTotalFence = true;
              }
            }
          }
        }
      }
    });
  }

  // -------------------------------------------------------------------------
  // Entry helpers
  // -------------------------------------------------------------------------

  Block *getFuncBlock(Operation *op, std::string &funcName) {
    if (auto entryOp = dyn_cast<cuda_tile::EntryOp>(op)) {
      funcName = entryOp.getSymName().str();
      return &entryOp.getBody().front();
    }
    return nullptr;
  }

public:
  AutoGenMemoryTokenPass() = default;
  AutoGenMemoryTokenPass(bool enable_autogen_alias_mem_token) {
    this->enable_autogen_alias_mem_token = enable_autogen_alias_mem_token;
  }

  void runOnOperation() override {
    MLIRContext *context = &getContext();
    Operation *funcOp = getOperation();
    IRRewriter rewriter(context);

    std::string fname;
    Block *body = getFuncBlock(funcOp, fname);
    if (!body)
      return;

    BlockTokenSummary summary = summarizeBlockTokens(body);

    if (handleUserTokenBailout(funcOp, body, summary, rewriter))
      return;

    if (!this->enable_autogen_alias_mem_token && !summary.hasDebugBarrier)
      return;

    aliasInfo.emplace(funcOp);
    dependenceInfo.emplace();
    regionSummaries.clear();

    bool hasHazard = hasAliasOrderingHazard(body);
    if (!hasHazard && !summary.hasFenceOrSignal && !summary.hasDebugBarrier) {
      LLVM_DEBUG(llvm::errs() << "[AutoGenMemoryTokenPass] will not modify IR: "
                              << fname << ".\n");
      return;
    }
    LLVM_DEBUG(llvm::errs() << "[AutoGenMemoryTokenPass] will modify IR: "
                            << fname << ".\n");

    computeRegionSummaries(funcOp);

    TokenState entryState;
    (void)addMemTokenForBlock(body, std::move(entryState), rewriter);
  }
};

} // namespace

std::unique_ptr<Pass> mlir::triton::createAutoGenMemoryTokenPass() {
  return std::make_unique<AutoGenMemoryTokenPass>();
}

std::unique_ptr<Pass> mlir::triton::createAutoGenMemoryTokenPass(
    bool enable_autogen_alias_mem_token) {
  return std::make_unique<AutoGenMemoryTokenPass>(
      enable_autogen_alias_mem_token);
}
