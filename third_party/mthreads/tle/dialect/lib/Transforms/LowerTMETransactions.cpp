#ifdef __TLE__

#include "Dialect/MUSA/IR/Dialect.h"
#include "TritonMUSACommon/TMEUtils.h"
#include "TritonMUSAGPUTransforms/Passes.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/PatternMatch.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/STLExtras.h"

#include <cstdint>
#include <algorithm>
#include <iterator>
#include <limits>
#include <optional>

namespace mlir {

#define GEN_PASS_DEF_TRITONMUSAGPUTLELOWERTMETRANSACTIONS
#include "TritonMUSAGPUTransforms/Passes.h.inc"

namespace {

namespace ttg = triton::gpu;
namespace ttmg = triton::musa;

static FailureOr<int32_t>
resolveIssueThread(ttmg::AsyncTMECopyGlobalToLocalOp copy) {
  auto ws = copy->getParentOfType<ttg::WarpSpecializeOp>();
  if (!ws)
    return 0;

  Region *copyRegion = copy->getParentRegion();
  bool defaultProducer = ws->hasAttr("musa_tle.default_producer");
  if (defaultProducer) {
    if (!ws.getDefaultRegion().isAncestor(copyRegion)) {
      copy.emitOpError(
          "mthreads TLE default producer TME copy must be in the default "
          "partition");
      return failure();
    }
    return 0;
  }
  if (ws.getDefaultRegion().isAncestor(copyRegion)) {
    copy.emitOpError(
        "mthreads TLE completion TME copy must be in the producer partition");
    return failure();
  }

  std::optional<unsigned> partitionIndex;
  auto partitionRegions = ws.getPartitionRegions();
  for (auto [index, region] : llvm::enumerate(partitionRegions)) {
    if (region->isAncestor(copyRegion)) {
      partitionIndex = index;
      break;
    }
  }
  if (!partitionIndex || *partitionIndex + 1 != partitionRegions.size()) {
    copy.emitOpError(
        "mthreads TLE completion TME copy must be in the final producer "
        "partition");
    return failure();
  }

  ModuleOp module = copy->getParentOfType<ModuleOp>();
  auto numWarps = module->getAttrOfType<IntegerAttr>(ttg::AttrNumWarpsName);
  auto threadsPerWarp =
      module->getAttrOfType<IntegerAttr>(ttg::AttrNumThreadsPerWarp);
  if (!numWarps || !threadsPerWarp || numWarps.getInt() <= 0 ||
      threadsPerWarp.getInt() <= 0) {
    copy.emitOpError("mthreads TLE producer issue thread requires "
                     "ttg.num-warps and ttg.threads-per-warp");
    return failure();
  }

  int64_t producerStartWarp = numWarps.getInt();
  ArrayRef<int32_t> workerNumWarps = ws.getPartitionNumWarps();
  if (workerNumWarps.size() != partitionRegions.size()) {
    copy.emitOpError(
        "mthreads TLE producer issue thread requires matching worker warp "
        "counts");
    return failure();
  }
  for (unsigned index = 0; index < *partitionIndex; ++index)
    producerStartWarp += workerNumWarps[index];
  int64_t issueThread = producerStartWarp * threadsPerWarp.getInt();
  if (issueThread > std::numeric_limits<int32_t>::max()) {
    copy.emitOpError("mthreads TLE producer issue thread exceeds int32 range");
    return failure();
  }
  return static_cast<int32_t>(issueThread);
}

// A multi-field pipe commit may be wrapped by one predicated prefetch
// `scf.if`.  The payload copies are still one logical transaction, but the
// producer arrival must be emitted after the region so it is not stranded in
// a branch block.  Keep this relaxation narrow: all fields must be in the same
// branch of one `scf.if` directly nested in the same `scf.for` body.  A group
// split across sibling branches, loops, or arbitrary regions remains
// rejected because an unconditional arrival would be able to deadlock the
// opposite path.
struct GroupBranchContext {
  scf::IfOp ifOp;
  bool thenBranch;
  scf::ForOp forOp;
};

static bool isConstantTrue(Value value) {
  if (auto constant = value.getDefiningOp<arith::ConstantIntOp>())
    return constant.value() == 1;
  return false;
}

static std::optional<GroupBranchContext>
getGroupedBranchContext(ArrayRef<ttmg::AsyncTMECopyGlobalToLocalOp> copies) {
  if (copies.empty())
    return std::nullopt;

  auto findNearestIf = [](Operation *op)
      -> std::optional<std::pair<scf::IfOp, bool>> {
    for (Region *region = op->getParentRegion(); region;) {
      Operation *parent = region->getParentOp();
      if (!parent)
        break;
      if (auto ifOp = dyn_cast<scf::IfOp>(parent)) {
        if (region == &ifOp.getThenRegion())
          return std::make_pair(ifOp, true);
        if (ifOp.elseBlock() && region == &ifOp.getElseRegion())
          return std::make_pair(ifOp, false);
        return std::nullopt;
      }
      region = parent->getParentRegion();
    }
    return std::nullopt;
  };

  auto firstCopy = copies.front();
  auto first = findNearestIf(firstCopy.getOperation());
  if (!first)
    return std::nullopt;
  auto [ifOp, thenBranch] = *first;
  // The branch operation itself must be in the body of one loop.  This avoids
  // treating a warp-specialize/SCF region or a nested loop as a single phase
  // domain when the barrier state is actually carried elsewhere.
  auto forOp = ifOp->getParentOfType<scf::ForOp>();
  if (!forOp || ifOp->getParentRegion() != &forOp.getRegion())
    return std::nullopt;

  Block *copyBlock = copies.front()->getBlock();
  if (!copyBlock)
    return std::nullopt;
  for (auto copy : copies) {
    auto context = findNearestIf(copy.getOperation());
    if (!context || context->first != ifOp || context->second != thenBranch ||
        copy->getBlock() != copyBlock)
      return std::nullopt;
    if (!copy->getBlock()->getParent() ||
        copy->getParentOfType<scf::ForOp>() != forOp)
      return std::nullopt;
  }
  return GroupBranchContext{ifOp, thenBranch, forOp};
}

static Value combineBranchAndCopyPred(OpBuilder &builder, Location loc,
                                      GroupBranchContext context, Value pred) {
  Value branchPred = context.ifOp.getCondition();
  if (!context.thenBranch) {
    Value one = arith::ConstantIntOp::create(builder, loc, 1, 1);
    branchPred = arith::XOrIOp::create(builder, loc, branchPred, one);
  }
  if (isConstantTrue(branchPred))
    return pred;
  if (isConstantTrue(pred))
    return branchPred;
  return arith::AndIOp::create(builder, loc, branchPred, pred);
}

// Return the contiguous grouped payloads immediately preceding `final` in
// the same lexical copy stream.  Barrier SSA values are commonly reused by
// every loop iteration, so grouping all copies by barId alone would merge
// distinct commits.  Contiguity plus the final marker is the existing pipe
// protocol's group boundary.
static SmallVector<ttmg::AsyncTMECopyGlobalToLocalOp>
collectLegacyGroupedCopies(
    ttmg::AsyncTMECopyGlobalToLocalOp final,
    ArrayRef<ttmg::AsyncTMECopyGlobalToLocalOp> copies) {
  SmallVector<ttmg::AsyncTMECopyGlobalToLocalOp> group;
  if (!final->hasAttr(ttmg::kTLEGroupedCompletionFinalAttr))
    return group;
  auto finalIt = llvm::find(copies, final);
  if (finalIt == copies.end())
    return group;
  group.push_back(final);
  while (finalIt != copies.begin()) {
    auto previous = std::prev(finalIt);
    auto previousCopy = *previous;
    if (!previousCopy->hasAttr(ttmg::kTLEGroupedCompletionAttr) ||
        previousCopy.getBarId() != final.getBarId())
      break;
    group.push_back(*previous);
    finalIt = previous;
  }
  std::reverse(group.begin(), group.end());
  return group;
}

class LowerTMETransactionsPass
    : public impl::TritonMUSAGPUTLELowerTMETransactionsBase<
          LowerTMETransactionsPass> {
  void runOnOperation() override {
    ModuleOp module = getOperation();
    IRRewriter rewriter(&getContext());

    SmallVector<ttmg::AsyncTMECopyGlobalToLocalOp> copies;
    SmallVector<ttmg::AsyncTMECopyGlobalToLocalOp> perCopyCopies;
    llvm::SmallPtrSet<Operation *, 8> loweredPerCopyCopies;
    module.walk([&](ttmg::AsyncTMECopyGlobalToLocalOp copy) {
      if (copy->hasAttr(ttmg::kTLEExpectBytesAttr) ||
          copy->hasAttr(ttmg::kTLEGroupedCompletionAttr))
        copies.push_back(copy);
      if (copy->hasAttr(ttmg::kTLEGroupedPerCopyCompletionAttr))
        perCopyCopies.push_back(copy);
    });
    // The experimental per-copy protocol is intentionally handled as a
    // separate pass over the copies.  This keeps the established grouped
    // completion contract (one aggregate add.trans emitted by its leader)
    // byte-for-byte unchanged.  A group is identified by the SSA barrier
    // value; LowerPipe reuses one barrier.index value for all payload fields.
    // Every copy contributes its own transaction, and exactly one final copy
    // emits the producer arrival after all asynchronous copies have been
    // issued.
    for (ttmg::AsyncTMECopyGlobalToLocalOp copy : perCopyCopies) {
      if (!copy->hasAttr(ttmg::kTLEExpectBytesAttr)) {
        copy.emitOpError("per-copy grouped completion requires positive "
                         "expect_bytes on every payload");
        signalPassFailure();
        return;
      }
    }

    struct PerCopyGroup {
      Value barrier;
      SmallVector<ttmg::AsyncTMECopyGlobalToLocalOp> copies;
      SmallVector<ttmg::AsyncTMECopyGlobalToLocalOp> finals;
    };
    SmallVector<PerCopyGroup> perCopyGroups;
    for (ttmg::AsyncTMECopyGlobalToLocalOp copy : perCopyCopies) {
      PerCopyGroup *group = nullptr;
      for (PerCopyGroup &candidate : perCopyGroups) {
        if (candidate.barrier == copy.getBarId()) {
          group = &candidate;
          break;
        }
      }
      if (!group) {
        perCopyGroups.push_back(PerCopyGroup{copy.getBarId(), {}, {}});
        group = &perCopyGroups.back();
      }
      group->copies.push_back(copy);
      if (copy->hasAttr(ttmg::kTLEGroupedCompletionFinalAttr))
        group->finals.push_back(copy);
    }

    for (PerCopyGroup &group : perCopyGroups) {
      if (group.copies.size() < 2) {
        group.copies.front().emitOpError(
            "per-copy grouped completion requires at least two payloads");
        signalPassFailure();
        return;
      }
      if (group.finals.size() != 1) {
        group.copies.front().emitOpError(
            "per-copy grouped completion requires exactly one final payload");
        signalPassFailure();
        return;
      }
      auto final = group.finals.front();
      auto branchContext = getGroupedBranchContext(group.copies);
      // Arriving before a later payload has been issued would permit the
      // barrier to become ready too early.  The ordinary form is restricted
      // to one block.  A narrowly proven predicated branch may defer the
      // arrival until after its `scf.if`; the branch predicate is retained on
      // the deferred op so a not-taken prefetch does not signal a transaction
      // that was never issued.
      Block *finalBlock = final->getBlock();
      for (auto copy : group.copies) {
        bool orderedInBlock = copy->getBlock() == finalBlock &&
                              (copy == final || copy->isBeforeInBlock(final));
        bool orderedInBranch = branchContext &&
                               (copy == final || copy->isBeforeInBlock(final));
        if ((!branchContext && !orderedInBlock) ||
            (branchContext && !orderedInBranch)) {
          final.emitOpError("per-copy grouped completion final payload must "
                            "follow all other payload copies in one block");
          signalPassFailure();
          return;
        }
      }

      FailureOr<int32_t> issueThread = resolveIssueThread(final);
      if (failed(issueThread)) {
        signalPassFailure();
        return;
      }
      auto issueThreadAttr = rewriter.getI32IntegerAttr(*issueThread);
      auto explicitCompletionAttr = rewriter.getUnitAttr();
      for (auto copy : group.copies) {
        auto expectBytes =
            copy->getAttrOfType<IntegerAttr>(ttmg::kTLEExpectBytesAttr);
        if (!expectBytes || !expectBytes.getType().isInteger(32) ||
            expectBytes.getInt() <= 0) {
          copy.emitOpError(
              "per-copy grouped completion requires positive expect_bytes");
          signalPassFailure();
          return;
        }
        Location loc = copy.getLoc();
        copy->setAttr(ttmg::kTMEIssueThreadAttr, issueThreadAttr);
        copy->setAttr(ttmg::kTMEExplicitCompletionAttr,
                      explicitCompletionAttr);
        rewriter.setInsertionPoint(copy);
        Value bytes = arith::ConstantIntOp::create(
            rewriter, loc, expectBytes.getInt(), 32);
        auto addTrans = ttmg::BarrierAddTransOp::create(
            rewriter, loc, copy.getBarId(), bytes, copy.getPred());
        addTrans->setAttr(ttmg::kTMEIssueThreadAttr, issueThreadAttr);
        addTrans->setAttr(ttmg::kTMEExplicitCompletionAttr,
                          explicitCompletionAttr);
        if (copy == final) {
          if (branchContext)
            rewriter.setInsertionPointAfter(branchContext->ifOp);
          else
            rewriter.setInsertionPointAfter(copy);
          Value arrivePred = copy.getPred();
          if (branchContext)
            arrivePred = combineBranchAndCopyPred(
                rewriter, loc, *branchContext, arrivePred);
          auto arrive = ttmg::ArriveBarrierNoRetOp::create(
              rewriter, loc, copy.getBarId(), arrivePred);
          arrive->setAttr(ttmg::kTMEIssueThreadAttr, issueThreadAttr);
          arrive->setAttr(ttmg::kTMEExplicitCompletionAttr,
                          explicitCompletionAttr);
        }
        copy->removeAttr(ttmg::kTLEExpectBytesAttr);
        copy->removeAttr(ttmg::kTLEGroupedPerCopyCompletionAttr);
        copy->removeAttr(ttmg::kTLEGroupedCompletionFinalAttr);
        // The legacy pass below iterates over the original `copies` snapshot.
        // Mark this operation explicitly because the per-copy lowering removes
        // its marker attributes before that second pass runs.
        loweredPerCopyCopies.insert(copy.getOperation());
      }
    }

    for (ttmg::AsyncTMECopyGlobalToLocalOp copy : copies) {
      // Per-copy groups were lowered above and their temporary attributes have
      // been removed.  Do not let this loop emit the legacy add/arrive pair a
      // second time when the vectors overlap.
      if (loweredPerCopyCopies.contains(copy.getOperation()))
        continue;
      auto expectBytes =
          copy->getAttrOfType<IntegerAttr>(ttmg::kTLEExpectBytesAttr);
      bool groupedCompletion =
          copy->hasAttr(ttmg::kTLEGroupedCompletionAttr);
      if (expectBytes && (!expectBytes.getType().isInteger(32) ||
                          expectBytes.getInt() <= 0)) {
        copy.emitOpError(
            "mthreads TLE completion TME copy requires positive expect_bytes");
        signalPassFailure();
        return;
      }
      if (!expectBytes && !groupedCompletion) {
        copy.emitOpError("mthreads TLE completion TME copy requires transaction "
                         "bytes or a grouped completion marker");
        signalPassFailure();
        return;
      }

      FailureOr<int32_t> issueThread = resolveIssueThread(copy);
      if (failed(issueThread)) {
        signalPassFailure();
        return;
      }

      Location loc = copy.getLoc();
      auto issueThreadAttr = rewriter.getI32IntegerAttr(*issueThread);
      auto explicitCompletionAttr = rewriter.getUnitAttr();
      bool groupedFinal =
          copy->hasAttr(ttmg::kTLEGroupedCompletionFinalAttr);
      SmallVector<ttmg::AsyncTMECopyGlobalToLocalOp> legacyGroup;
      std::optional<GroupBranchContext> legacyBranch;
      if (groupedFinal) {
        legacyGroup = collectLegacyGroupedCopies(copy, copies);
        if (legacyGroup.size() > 1)
          legacyBranch = getGroupedBranchContext(legacyGroup);
      }
      copy->setAttr(ttmg::kTMEIssueThreadAttr, issueThreadAttr);
      copy->setAttr(ttmg::kTMEExplicitCompletionAttr, explicitCompletionAttr);
      if (expectBytes) {
        rewriter.setInsertionPoint(copy);
        SmallVector<int32_t> transactionParts;
        if (auto parts = copy->getAttrOfType<DenseI32ArrayAttr>(
                ttmg::kTLEExpectBytesPartsAttr)) {
          int64_t sum = 0;
          for (int32_t part : parts.asArrayRef()) {
            if (part <= 0 || sum > std::numeric_limits<int32_t>::max() - part) {
              copy.emitOpError("expect_bytes_parts must contain positive i32 values");
              signalPassFailure();
              return;
            }
            sum += part;
            transactionParts.push_back(part);
          }
          if (transactionParts.empty() || sum != expectBytes.getInt()) {
            copy.emitOpError("expect_bytes_parts must sum to expect_bytes");
            signalPassFailure();
            return;
          }
        } else {
          transactionParts.push_back(expectBytes.getInt());
        }
        Value bytes = arith::ConstantIntOp::create(
            rewriter, loc, expectBytes.getInt(), 32);
        auto addTrans = ttmg::BarrierAddTransOp::create(
            rewriter, loc, copy.getBarId(), bytes, copy.getPred());
        addTrans->setAttr(ttmg::kTMEIssueThreadAttr, issueThreadAttr);
        addTrans->setAttr(ttmg::kTMEExplicitCompletionAttr,
                          explicitCompletionAttr);
        if (transactionParts.size() > 1)
          addTrans->setAttr(ttmg::kTLEExpectBytesPartsAttr,
                            rewriter.getDenseI32ArrayAttr(transactionParts));
        if (!groupedCompletion || groupedFinal) {
          if (legacyBranch)
            rewriter.setInsertionPointAfter(legacyBranch->ifOp);
          else
            rewriter.setInsertionPointAfter(copy);
          Value arrivePred = copy.getPred();
          if (legacyBranch)
            arrivePred = combineBranchAndCopyPred(
                rewriter, loc, *legacyBranch, arrivePred);
          auto arrive = ttmg::ArriveBarrierNoRetOp::create(
              rewriter, loc, copy.getBarId(), arrivePred);
          arrive->setAttr(ttmg::kTMEIssueThreadAttr, issueThreadAttr);
          arrive->setAttr(ttmg::kTMEExplicitCompletionAttr,
                          explicitCompletionAttr);
        }
      } else if (groupedCompletion && groupedFinal) {
        // Non-leader fields contribute completion bytes through the aggregate
        // transaction registered by the first field.  The final field emits
        // the single producer arrival for this commit.
        rewriter.setInsertionPointAfter(copy);
        auto arrive = ttmg::ArriveBarrierNoRetOp::create(
            rewriter, loc, copy.getBarId(), copy.getPred());
        arrive->setAttr(ttmg::kTMEIssueThreadAttr, issueThreadAttr);
        arrive->setAttr(ttmg::kTMEExplicitCompletionAttr,
                        explicitCompletionAttr);
      }
      copy->removeAttr(ttmg::kTLEExpectBytesAttr);
      copy->removeAttr(ttmg::kTLEExpectBytesPartsAttr);
      copy->removeAttr(ttmg::kTLEGroupedCompletionAttr);
      copy->removeAttr(ttmg::kTLEGroupedCompletionFinalAttr);
    }

  }
};

} // namespace
} // namespace mlir

#endif // __TLE__
