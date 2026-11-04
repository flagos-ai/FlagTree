#include "Analysis/TkoAliasAnalysis.h"
#include "Analysis/TkoOpSemantics.h"

#include "cuda_tile/Dialect/CudaTile/IR/Ops.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/raw_ostream.h"

#define DEBUG_TYPE "tko-alias-analysis"

using namespace mlir;
using namespace mlir::triton::tileir;

//===----------------------------------------------------------------------===//
// Root walk
//===----------------------------------------------------------------------===//

/*
 * Root walk implementation details.
 *
 * The public analysis contract is Operation -> MemoryRootId. Internally, the
 * root walk starts from the access value selected by TkoOpSemantics and walks
 * backward through addressing-only producers and control-flow joins until it
 * reaches terminal SSA values.
 *
 * Addressing-only producers:
 *   - make_tensor_view:        base
 *   - make_partition_view:     tensor_view
 *   - make_strided_view:       tensor_view
 *   - make_gather_scatter_view: tensor_view
 *   - ptr_to_ptr:              source
 *   - broadcast, reshape:      first operand
 *
 * Join points:
 *   - arith.select, cuda_tile.select: true + false operands
 *   - scf.for iter_arg:        init + body yield
 *   - scf.for result:          init + body yield
 *   - scf.if result:           then yield + else yield
 *   - scf.while block arg:     inits + condition forward
 *   - scf.while result:        condition op args
 *
 * Terminal root policy:
 *   - repeated cuda_tile.get_global ops for the same symbol share one root;
 *   - any other unrecognized SSA producer becomes its own terminal root.
 *
 * The last rule matches Triton's convention that distinct function-arg
 * pointers denote distinct buffers unless explicit memToken ordering is already
 * present. AutoGenMemoryToken bails out on those explicit chains.
 */
ArrayRef<Value> TkoAliasAnalysis::computeRoots(Value v,
                                               DenseSet<Value> &visiting) {
  auto it = rootsCache.find(v);
  if (it != rootsCache.end())
    return it->second;

  // Cycle detection: if `v` is already on the current walk stack, bail
  // out with an empty contribution — the caller that originally placed
  // `v` on the stack will collect roots from the other branches.
  if (!visiting.insert(v).second)
    return {};

  SmallVector<Value, 2> roots;

  auto addAll = [&](ArrayRef<Value> rs) {
    for (Value r : rs)
      if (llvm::find(roots, r) == roots.end())
        roots.push_back(r);
  };

  auto recurse = [&](Value operand) {
    addAll(computeRoots(operand, visiting));
  };

  // Walk every cuda_tile.continue / cuda_tile.break that targets `loop`
  // (i.e. that sits at top level of its body, or nested inside if/unrelated
  // scopes — but NOT inside a deeper cuda_tile loop, which would bind to
  // that inner loop instead).
  auto recurseLoopExitsAt = [&](Operation *loop, Region &region, unsigned idx,
                                bool includeContinue, bool includeBreak) {
    region.walk([&](Operation *op) {
      if (op == loop)
        return WalkResult::advance();
      if (isa<cuda_tile::ForOp, cuda_tile::LoopOp>(op))
        return WalkResult::skip();
      bool isExit = (includeContinue && isa<cuda_tile::ContinueOp>(op)) ||
                    (includeBreak && isa<cuda_tile::BreakOp>(op));
      if (isExit && idx < op->getNumOperands())
        recurse(op->getOperand(idx));
      return WalkResult::advance();
    });
  };

  if (auto blockArg = dyn_cast<BlockArgument>(v)) {
    Operation *parent = blockArg.getOwner()->getParentOp();
    if (auto forOp = dyn_cast<scf::ForOp>(parent)) {
      unsigned argNum = blockArg.getArgNumber();
      // arg 0 is the induction variable (i32/index, not an access value);
      // remaining args are iter_args, 1:1 with initArgs / yield operands.
      if (argNum == 0) {
        roots.push_back(v); // terminal, shouldn't be an access value anyway
      } else {
        unsigned idx = argNum - 1;
        recurse(forOp.getInitArgs()[idx]);
        Operation *term = forOp.getBody()->getTerminator();
        if (idx < term->getNumOperands())
          recurse(term->getOperand(idx));
      }
    } else if (auto ctForOp = dyn_cast<cuda_tile::ForOp>(parent)) {
      // cuda_tile.for body arg layout mirrors scf.for: [induction, iter0,
      // iter1, ...]. iter_arg values come from initValues + every
      // cuda_tile.continue's operand at `idx` (including nested continues
      // inside `if` branches).
      unsigned argNum = blockArg.getArgNumber();
      if (argNum == 0) {
        roots.push_back(v);
      } else {
        unsigned idx = argNum - 1;
        if (idx < ctForOp.getInitValues().size())
          recurse(ctForOp.getInitValues()[idx]);
        recurseLoopExitsAt(ctForOp, ctForOp.getRegion(), idx,
                           /*includeContinue=*/true, /*includeBreak=*/false);
      }
    } else if (auto ctLoopOp = dyn_cast<cuda_tile::LoopOp>(parent)) {
      // cuda_tile.loop body args = loop-carried vars (no induction). Sources
      // are initValues + every continue's operand at `argNum`.
      unsigned idx = blockArg.getArgNumber();
      if (idx < ctLoopOp->getNumOperands())
        recurse(ctLoopOp->getOperand(idx));
      recurseLoopExitsAt(ctLoopOp, ctLoopOp.getRegion(), idx,
                         /*includeContinue=*/true, /*includeBreak=*/false);
    } else if (auto whileOp = dyn_cast<scf::WhileOp>(parent)) {
      unsigned argNum = blockArg.getArgNumber();
      if (blockArg.getParentBlock() == whileOp.getBeforeBody()) {
        // Before-region block args: sources are the inits and the
        // after-region's yield operands.
        if (argNum < whileOp.getInits().size())
          recurse(whileOp.getInits()[argNum]);
        Block *after = whileOp.getAfterBody();
        if (after) {
          Operation *yield = after->getTerminator();
          if (argNum < yield->getNumOperands())
            recurse(yield->getOperand(argNum));
        }
      } else {
        // After-region block args: sources are the condition op's args.
        auto condOp =
            cast<scf::ConditionOp>(whileOp.getBeforeBody()->getTerminator());
        if (argNum < condOp.getArgs().size())
          recurse(condOp.getArgs()[argNum]);
      }
    } else {
      // Function arg, unknown region's block arg, etc. — terminal root.
      roots.push_back(v);
    }
  } else if (Operation *def = v.getDefiningOp()) {
    if (Value pt = tko_op_semantics::getPassthroughAccessValue(def)) {
      recurse(pt);
    } else if (auto getGlobal = dyn_cast<cuda_tile::GetGlobalOp>(def)) {
      auto it = globalRootByName.try_emplace(getGlobal.getNameAttr(), v).first;
      roots.push_back(it->second);
    } else if (auto sel = dyn_cast<cuda_tile::SelectOp>(def)) {
      recurse(sel.getValIfTrue());
      recurse(sel.getValIfFalse());
    } else if (auto sel = dyn_cast<arith::SelectOp>(def)) {
      recurse(sel.getTrueValue());
      recurse(sel.getFalseValue());
    } else if (auto forOp = dyn_cast<scf::ForOp>(def)) {
      unsigned idx = cast<OpResult>(v).getResultNumber();
      if (idx < forOp.getInitArgs().size())
        recurse(forOp.getInitArgs()[idx]);
      Operation *term = forOp.getBody()->getTerminator();
      if (idx < term->getNumOperands())
        recurse(term->getOperand(idx));
    } else if (auto ctForOp = dyn_cast<cuda_tile::ForOp>(def)) {
      // cuda_tile.for result[i] = final iter_arg[i]: same sources as the
      // iter_arg block-arg branch above (init + every continue.operand[i]).
      unsigned idx = cast<OpResult>(v).getResultNumber();
      if (idx < ctForOp.getInitValues().size())
        recurse(ctForOp.getInitValues()[idx]);
      recurseLoopExitsAt(ctForOp, ctForOp.getRegion(), idx,
                         /*includeContinue=*/true, /*includeBreak=*/false);
    } else if (auto ctLoopOp = dyn_cast<cuda_tile::LoopOp>(def)) {
      // cuda_tile.loop result[i] = value yielded by cuda_tile.break at
      // operand[i] (every break that targets this loop contributes).
      unsigned idx = cast<OpResult>(v).getResultNumber();
      recurseLoopExitsAt(ctLoopOp, ctLoopOp.getRegion(), idx,
                         /*includeContinue=*/false, /*includeBreak=*/true);
    } else if (auto ctIfOp = dyn_cast<cuda_tile::IfOp>(def)) {
      // cuda_tile.if result[i] comes from cuda_tile.yield in then/else.
      unsigned idx = cast<OpResult>(v).getResultNumber();
      for (Region *region :
           {&ctIfOp.getThenRegion(), &ctIfOp.getElseRegion()}) {
        if (region->empty())
          continue;
        Operation *term = region->front().getTerminator();
        if (isa<cuda_tile::YieldOp>(term) && idx < term->getNumOperands())
          recurse(term->getOperand(idx));
      }
    } else if (auto ifOp = dyn_cast<scf::IfOp>(def)) {
      unsigned idx = cast<OpResult>(v).getResultNumber();
      if (auto thenYield = ifOp.thenYield())
        if (idx < thenYield->getNumOperands())
          recurse(thenYield->getOperand(idx));
      if (!ifOp.getElseRegion().empty())
        if (auto elseYield = ifOp.elseYield())
          if (idx < elseYield->getNumOperands())
            recurse(elseYield->getOperand(idx));
    } else if (auto whileOp = dyn_cast<scf::WhileOp>(def)) {
      unsigned idx = cast<OpResult>(v).getResultNumber();
      auto condOp =
          cast<scf::ConditionOp>(whileOp.getBeforeBody()->getTerminator());
      if (idx < condOp.getArgs().size())
        recurse(condOp.getArgs()[idx]);
    } else {
      // Unrecognized producer: treat as a terminal root.
      roots.push_back(v);
    }
  } else {
    // Value with no defining op and no block-arg owner — shouldn't
    // happen in well-formed IR, but play safe.
    roots.push_back(v);
  }

  visiting.erase(v);

  auto &cached = rootsCache[v];
  cached = std::move(roots);
  return cached;
}

//===----------------------------------------------------------------------===//
// Construction / finalization / queries
//===----------------------------------------------------------------------===//

TkoAliasAnalysis::TkoAliasAnalysis(Operation *funcLike) : funcLike(funcLike) {
  finalize();
}

void TkoAliasAnalysis::finalize() {
  if (finalized)
    return;
  finalized = true;

  // Every terminal root encountered during walks becomes its own root id
  // (default "no-alias on distinct SSA roots"). Root ids are assigned in the
  // order memory accesses first expose them during the function walk, so token
  // materialization does not depend on DenseMap or pointer iteration order.
  SmallVector<Value, 8> leaders;
  DenseSet<Value> seen;
  DenseSet<Value> visiting;
  funcLike->walk([&](Operation *op) {
    Value accessValue = tko_op_semantics::getAccessValue(op);
    if (!accessValue)
      return;
    for (Value r : computeRoots(accessValue, visiting)) {
      if (seen.insert(r).second)
        leaders.push_back(r);
    }
  });

  rootIds.clear();
  rootSetCache.clear();
  numRoots = leaders.size();
  for (auto [i, v] : llvm::enumerate(leaders))
    rootIds[v] = i;
}

TkoAliasAnalysis::MemoryRootSet TkoAliasAnalysis::getValueRoots(Value v) {
  auto cached = rootSetCache.find(v);
  if (cached != rootSetCache.end())
    return cached->second;

  // If `v` wasn't seen during construction, walk on demand.
  if (!rootsCache.count(v)) {
    DenseSet<Value> visiting;
    (void)computeRoots(v, visiting);
    // A new root may have appeared. Do a minimal re-finalize: only assign root
    // ids for newly seen roots.
    bool addedRoot = false;
    for (Value r : rootsCache[v]) {
      if (!rootIds.count(r)) {
        rootIds[r] = numRoots++;
        addedRoot = true;
      }
    }
    if (addedRoot)
      rootSetCache.clear();
  }

  MemoryRootSet out(numRoots, false);
  auto it = rootsCache.find(v);
  if (it != rootsCache.end()) {
    for (Value r : it->second) {
      auto rootIt = rootIds.find(r);
      if (rootIt != rootIds.end())
        out.set(rootIt->second);
    }
  }
  rootSetCache[v] = out;
  return out;
}

SmallVector<TkoAliasAnalysis::MemoryRootId, 2>
TkoAliasAnalysis::getAccessRoots(Operation *memOp) {
  Value accessValue = tko_op_semantics::getAccessValue(memOp);
  if (!accessValue)
    return {};

  MemoryRootSet roots = getValueRoots(accessValue);
  SmallVector<MemoryRootId, 2> out;
  for (unsigned root : roots.set_bits())
    out.push_back(root);
  return out;
}

void TkoAliasAnalysis::print(raw_ostream &os) const {
  os << "TkoAliasAnalysis: " << numRoots << " memory roots\n";
  SmallVector<std::pair<MemoryRootId, Value>, 8> rootEntries;
  for (auto &entry : rootIds)
    rootEntries.emplace_back(entry.second, entry.first);
  llvm::sort(rootEntries, [](auto &a, auto &b) { return a.first < b.first; });
  for (auto &[root, val] : rootEntries) {
    os << "  root " << root << ": ";
    val.printAsOperand(os, OpPrintingFlags());
    os << "\n";
  }
}
