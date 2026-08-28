#ifdef __TLE__

#include "Dialect/MUSA/IR/Dialect.h"
#include "MUSATLE/Transforms/PipePartitionUtils.h"
#include "TritonMUSACommon/TMEUtils.h"
#include "TritonMUSAGPUTransforms/Passes.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/PatternMatch.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallPtrSet.h"

#include <cstdint>
#include <limits>
#include <memory>
#include <optional>

namespace mlir {

#define GEN_PASS_DEF_TRITONMUSAGPUTLELOWERTMETRANSACTIONS
#include "TritonMUSAGPUTransforms/Passes.h.inc"

namespace {

namespace ttg = triton::gpu;
namespace ttmg = triton::musa;

static FailureOr<int32_t>
resolveIssueThread(ttmg::AsyncTMECopyGlobalToLocalOp copy) {
  // Standalone completion TME keeps its pre-existing static-WS contract.
  // Only the marked MUSA TLE static warp-specialize path uses arbitrary
  // default/worker partition starts.
  auto ws = copy->getParentOfType<ttg::WarpSpecializeOp>();
  if (ws && !ws->hasAttr("musa_tle.static_warp_specialize")) {
    Region *copyRegion = copy->getParentRegion();
    if (ws.getDefaultRegion().isAncestor(copyRegion))
      return copy.emitOpError(
          "mthreads TLE completion TME copy must be in the producer partition");
    std::optional<unsigned> worker;
    for (auto [index, region] : llvm::enumerate(ws.getPartitionRegions())) {
      if (region->isAncestor(copyRegion)) {
        worker = index;
        break;
      }
    }
    if (!worker || *worker != 0)
      return copy.emitOpError(
          "mthreads TLE completion TME copy must be in the producer partition");
    ModuleOp module = copy->getParentOfType<ModuleOp>();
    auto numWarps = module->getAttrOfType<IntegerAttr>(ttg::AttrNumWarpsName);
    auto threadsPerWarp =
        module->getAttrOfType<IntegerAttr>(ttg::AttrNumThreadsPerWarp);
    if (!numWarps || !threadsPerWarp || numWarps.getInt() <= 0 ||
        threadsPerWarp.getInt() <= 0)
      return copy.emitOpError(
          "mthreads TLE producer issue thread requires ttg.num-warps and "
          "ttg.threads-per-warp");
    int64_t value = numWarps.getInt() * threadsPerWarp.getInt();
    if (value > std::numeric_limits<int32_t>::max())
      return copy.emitOpError(
          "mthreads TLE producer issue thread exceeds int32 range");
    return static_cast<int32_t>(value);
  }
  FailureOr<int32_t> startThread =
      triton::musa_tle::getPipePartitionStartThread(copy);
  if (failed(startThread))
    return failure();
  return *startThread;
}

struct CompletionGroupMetadata {
  int64_t id = 0;
  int64_t memberIndex = 0;
  int64_t memberCount = 0;
  int64_t totalTransactionBytes = 0;
};

struct CompletionGroupInstance {
  CompletionGroupMetadata metadata;
  Value barrier;
  Value predicate;
  int32_t issueThread = 0;
  int64_t accumulatedBytes = 0;
  SmallVector<bool> seenMembers;
  SmallVector<ttmg::AsyncTMECopyGlobalToLocalOp> members;
};

static FailureOr<std::optional<CompletionGroupMetadata>>
getCompletionGroupMetadata(ttmg::AsyncTMECopyGlobalToLocalOp copy) {
  Attribute raw = copy->getAttr(ttmg::kTLECompletionGroupAttr);
  if (!raw)
    return std::optional<CompletionGroupMetadata>{};

  auto group = dyn_cast<DenseI64ArrayAttr>(raw);
  if (!group || group.size() != 4) {
    copy.emitOpError("MUSA TLE completion group metadata requires group id, "
                     "member index, member count, and total bytes");
    return failure();
  }

  ArrayRef<int64_t> values = group.asArrayRef();
  CompletionGroupMetadata metadata{values[0], values[1], values[2], values[3]};
  if (metadata.id < 0 || metadata.memberIndex < 0 ||
      metadata.memberCount <= 0 ||
      metadata.memberCount > std::numeric_limits<int32_t>::max() ||
      metadata.memberIndex >= metadata.memberCount ||
      metadata.totalTransactionBytes <= 0 ||
      metadata.totalTransactionBytes > std::numeric_limits<int32_t>::max()) {
    copy.emitOpError("MUSA TLE completion group metadata is out of range");
    return failure();
  }
  return std::optional<CompletionGroupMetadata>{metadata};
}

static FailureOr<int32_t>
getTransactionBytes(ttmg::AsyncTMECopyGlobalToLocalOp copy) {
  auto expectBytes =
      copy->getAttrOfType<IntegerAttr>(ttmg::kTLEExpectBytesAttr);
  if (!expectBytes || !expectBytes.getType().isInteger(32) ||
      expectBytes.getInt() <= 0) {
    copy.emitOpError(
        "MUSA TLE completion TME copy requires positive expect_bytes");
    return failure();
  }
  return static_cast<int32_t>(expectBytes.getInt());
}

static bool haveEquivalentPredicates(Value lhs, Value rhs) {
  if (lhs == rhs)
    return true;
  auto lhsConstant = lhs.getDefiningOp<arith::ConstantOp>();
  auto rhsConstant = rhs.getDefiningOp<arith::ConstantOp>();
  return lhsConstant && rhsConstant &&
         lhsConstant.getValue() == rhsConstant.getValue();
}

class LowerTMETransactionsPass
    : public impl::TritonMUSAGPUTLELowerTMETransactionsBase<
          LowerTMETransactionsPass> {
  void runOnOperation() override {
    ModuleOp module = getOperation();
    IRRewriter rewriter(&getContext());

    SmallVector<Block *> blocks;
    llvm::SmallPtrSet<Block *, 8> seenBlocks;
    module.walk([&](ttmg::AsyncTMECopyGlobalToLocalOp copy) {
      if ((copy->hasAttr(ttmg::kTLEExpectBytesAttr) ||
           copy->hasAttr(ttmg::kTLECompletionGroupAttr)) &&
          seenBlocks.insert(copy->getBlock()).second)
        blocks.push_back(copy->getBlock());
    });

    SmallVector<std::unique_ptr<CompletionGroupInstance>> groups;
    for (Block *block : blocks) {
      llvm::DenseMap<int64_t, std::unique_ptr<CompletionGroupInstance>> open;
      for (Operation &operation : *block) {
        auto copy = dyn_cast<ttmg::AsyncTMECopyGlobalToLocalOp>(&operation);
        if (!copy || (!copy->hasAttr(ttmg::kTLEExpectBytesAttr) &&
                      !copy->hasAttr(ttmg::kTLECompletionGroupAttr)))
          continue;

        FailureOr<int32_t> bytes = getTransactionBytes(copy);
        FailureOr<int32_t> issueThread = resolveIssueThread(copy);
        FailureOr<std::optional<CompletionGroupMetadata>> metadata =
            getCompletionGroupMetadata(copy);
        if (failed(bytes) || failed(issueThread) || failed(metadata)) {
          signalPassFailure();
          return;
        }

        if (!*metadata) {
          auto instance = std::make_unique<CompletionGroupInstance>();
          instance->metadata = CompletionGroupMetadata{0, 0, 1, *bytes};
          instance->barrier = copy.getBarId();
          instance->predicate = copy.getPred();
          instance->issueThread = *issueThread;
          instance->accumulatedBytes = *bytes;
          instance->seenMembers.push_back(true);
          instance->members.push_back(copy);
          groups.push_back(std::move(instance));
          continue;
        }

        const CompletionGroupMetadata &member = **metadata;
        std::unique_ptr<CompletionGroupInstance> &instance = open[member.id];
        if (!instance) {
          instance = std::make_unique<CompletionGroupInstance>();
          instance->metadata = member;
          instance->barrier = copy.getBarId();
          instance->predicate = copy.getPred();
          instance->issueThread = *issueThread;
          instance->seenMembers.assign(member.memberCount, false);
        }

        if (instance->metadata.memberCount != member.memberCount ||
            instance->metadata.totalTransactionBytes !=
                member.totalTransactionBytes) {
          copy.emitOpError(
              "MUSA TLE completion group members have inconsistent metadata");
          signalPassFailure();
          return;
        }
        if (instance->barrier != copy.getBarId() ||
            !haveEquivalentPredicates(instance->predicate, copy.getPred()) ||
            instance->issueThread != *issueThread) {
          copy.emitOpError("MUSA TLE completion group members must use the "
                           "same barrier, predicate, and issue thread");
          signalPassFailure();
          return;
        }
        if (instance->seenMembers[member.memberIndex]) {
          copy.emitOpError(
              "MUSA TLE completion group contains a duplicate member index");
          signalPassFailure();
          return;
        }
        instance->seenMembers[member.memberIndex] = true;
        instance->members.push_back(copy);
        instance->accumulatedBytes += *bytes;
        if (instance->accumulatedBytes > std::numeric_limits<int32_t>::max()) {
          copy.emitOpError(
              "MUSA TLE completion group bytes exceed the positive i32 range");
          signalPassFailure();
          return;
        }

        if (instance->members.size() ==
            static_cast<size_t>(member.memberCount)) {
          if (instance->accumulatedBytes != member.totalTransactionBytes) {
            copy.emitOpError(
                "MUSA TLE completion group member bytes do not match the "
                "declared total");
            signalPassFailure();
            return;
          }
          groups.push_back(std::move(instance));
          open.erase(member.id);
        }
      }

      if (!open.empty()) {
        open.begin()->second->members.front().emitOpError(
            "MUSA TLE completion group is incomplete within its block");
        signalPassFailure();
        return;
      }
    }

    for (const std::unique_ptr<CompletionGroupInstance> &ownedGroup : groups) {
      CompletionGroupInstance &group = *ownedGroup;
      ttmg::AsyncTMECopyGlobalToLocalOp first = group.members.front();
      ttmg::AsyncTMECopyGlobalToLocalOp last = group.members.back();
      Location loc = first.getLoc();
      auto issueThreadAttr = rewriter.getI32IntegerAttr(group.issueThread);
      auto explicitCompletionAttr = rewriter.getUnitAttr();

      rewriter.setInsertionPoint(first);
      Value bytes = arith::ConstantIntOp::create(
          rewriter, loc, group.metadata.totalTransactionBytes, 32);
      auto addTrans = ttmg::BarrierAddTransOp::create(
          rewriter, loc, group.barrier, bytes, group.predicate);

      rewriter.setInsertionPointAfter(last);
      auto arrive = ttmg::ArriveBarrierNoRetOp::create(
          rewriter, last.getLoc(), group.barrier, group.predicate);

      for (ttmg::AsyncTMECopyGlobalToLocalOp copy : group.members) {
        copy->setAttr(ttmg::kTMEIssueThreadAttr, issueThreadAttr);
        copy->setAttr(ttmg::kTMEExplicitCompletionAttr, explicitCompletionAttr);
        copy->removeAttr(ttmg::kTLEExpectBytesAttr);
        copy->removeAttr(ttmg::kTLECompletionGroupAttr);
      }
      for (Operation *op : {addTrans.getOperation(), arrive.getOperation()}) {
        op->setAttr(ttmg::kTMEIssueThreadAttr, issueThreadAttr);
        op->setAttr(ttmg::kTMEExplicitCompletionAttr, explicitCompletionAttr);
      }
    }
  }
};

} // namespace
} // namespace mlir

#endif // __TLE__
