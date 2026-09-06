#include "Dialect/MUSA/IR/Dialect.h"
#include "TritonMUSACommon/MMAOperandUtils.h"
#ifdef __TLE__
#include "TritonMUSACommon/TMEUtils.h"
#endif // __TLE__
#include "TritonMUSAGPUTransforms/Passes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/PatternMatch.h"
#include "triton/Conversion/TritonGPUToLLVM/Utility.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/SmallVector.h"

using namespace mlir;
namespace ttg = mlir::triton::gpu;

namespace {

static Value peelSqmmaIssueOperand(Value value) {
  llvm::SmallPtrSet<void *, 8> visited;
  while (value) {
    if (!visited.insert(value.getAsOpaquePointer()).second)
      break;

    if (auto cvt = value.getDefiningOp<ttg::ConvertLayoutOp>()) {
      value = cvt.getSrc();
      continue;
    }
    if (auto load = value.getDefiningOp<ttg::LocalLoadOp>()) {
      value = load.getSrc();
      continue;
    }
    if (auto view = value.getDefiningOp<ttg::MemDescIndexOp>()) {
      value = view.getSrc();
      continue;
    }
    if (auto view = value.getDefiningOp<ttg::MemDescSubsliceOp>()) {
      value = view.getSrc();
      continue;
    }
    if (auto view = value.getDefiningOp<ttg::MemDescReinterpretOp>()) {
      value = view.getSrc();
      continue;
    }
    if (auto view = value.getDefiningOp<ttg::MemDescTransOp>()) {
      value = view.getSrc();
      continue;
    }
    if (auto view = value.getDefiningOp<ttg::MemDescReshapeOp>()) {
      value = view.getSrc();
      continue;
    }
    break;
  }
  return value;
}

static bool isIssueBarrier(ttg::BarrierOp barrier) {
  return barrier && barrier.hasLocal() &&
         barrier.getAddrSpace() != ttg::AddrSpace::Local;
}

#ifdef __TLE__
static bool isInsideStaticWarpSpecialize(Operation *op) {
  for (Operation *parent = op->getParentOp(); parent;
       parent = parent->getParentOp()) {
    if (parent->hasAttr("musa_tle.static_warp_specialize"))
      return true;
  }
  return false;
}
#endif // __TLE__

static bool
shouldInsertIssueBarrierBefore(triton::musa::AsyncTMECopyLocalToGlobalOp op) {
#ifdef __TLE__
  bool isPipeReaderStore =
      op->hasAttr(triton::musa::kTLEPipeReaderTMEStoreAttr);
  bool suppressBarrier =
      isPipeReaderStore && isInsideStaticWarpSpecialize(op.getOperation());
  if (isPipeReaderStore)
    op->removeAttr(triton::musa::kTLEPipeReaderTMEStoreAttr);
  if (suppressBarrier)
    return false;
#endif // __TLE__
  Operation *prev = op->getPrevNode();
  auto prevBarrier = dyn_cast_or_null<ttg::BarrierOp>(prev);
  return !isIssueBarrier(prevBarrier);
}

static bool shouldInsertIssueBarrierBefore(triton::musa::SquadDotOp op) {
#ifdef __TLE__
  bool partitionLocal = isInsideStaticWarpSpecialize(op);
  if (partitionLocal && op->hasAttr("musa_tle.explicit_sqmma"))
    return false;

  Operation *prev = op->getPrevNode();
  auto prevBarrier = dyn_cast_or_null<ttg::BarrierOp>(prev);
  if (isIssueBarrier(prevBarrier) ||
      (partitionLocal && prevBarrier && prevBarrier.hasLocal()))
    return false;

  Value aMemDesc = peelSqmmaIssueOperand(op.getA());
  Value bMemDesc = peelSqmmaIssueOperand(op.getB());
  if (!aMemDesc || !bMemDesc)
    return true;
  if (!isa<ttg::MemDescType>(aMemDesc.getType()) ||
      !isa<ttg::MemDescType>(bMemDesc.getType()))
    return true;

  return triton::musa::needsSqmmaIssueBarrier(aMemDesc, bMemDesc);
#else
  Operation *prev = op->getPrevNode();
  auto prevBarrier = dyn_cast_or_null<ttg::BarrierOp>(prev);
  if (isIssueBarrier(prevBarrier))
    return false;

  Value aMemDesc = peelSqmmaIssueOperand(op.getA());
  Value bMemDesc = peelSqmmaIssueOperand(op.getB());
  if (!aMemDesc || !bMemDesc)
    return true;
  if (!isa<ttg::MemDescType>(aMemDesc.getType()) ||
      !isa<ttg::MemDescType>(bMemDesc.getType()))
    return true;

  return triton::musa::needsSqmmaIssueBarrier(aMemDesc, bMemDesc);
#endif // __TLE__
}

static void insertIssueBarrierBefore(Operation *op, RewriterBase &rewriter) {
  OpBuilder::InsertionGuard guard(rewriter);
  rewriter.setInsertionPoint(op);
#ifdef __TLE__
  if (isa<triton::musa::SquadDotOp>(op) && isInsideStaticWarpSpecialize(op)) {
    // Publish shared operands within the issuing partition. Late static WS
    // lowering turns local barriers into partition arrival/wait pairs; a CTA
    // issue barrier here would also wait for unrelated producer/reader warps.
    ttg::BarrierOp::create(rewriter, op->getLoc(), ttg::AddrSpace::Local);
    return;
  }
#endif // __TLE__
  // MUSA lowers non-local TTG barriers to llvm.musa.barrier0.
  ttg::BarrierOp::create(rewriter, op->getLoc(), ttg::AddrSpace::All);
}

} // namespace

namespace mlir {

#define GEN_PASS_DEF_TRITONMUSAGPUISSUEBARRIERINSERTION
#include "TritonMUSAGPUTransforms/Passes.h.inc"

struct TritonMUSAGPUIssueBarrierInsertionPass
    : impl::TritonMUSAGPUIssueBarrierInsertionBase<
          TritonMUSAGPUIssueBarrierInsertionPass> {
  void runOnOperation() override {
    ModuleOp mod = getOperation();
    IRRewriter rewriter(&getContext());
    SmallVector<Operation *> candidates;

    mod.walk([&](Operation *op) {
      if (isa<triton::musa::AsyncTMECopyLocalToGlobalOp,
              triton::musa::SquadDotOp>(op))
        candidates.push_back(op);
    });

    for (Operation *op : candidates) {
      if (auto store =
              dyn_cast<triton::musa::AsyncTMECopyLocalToGlobalOp>(op)) {
        if (shouldInsertIssueBarrierBefore(store))
          insertIssueBarrierBefore(op, rewriter);
        continue;
      }

      if (auto sqmma = dyn_cast<triton::musa::SquadDotOp>(op)) {
        if (shouldInsertIssueBarrierBefore(sqmma))
          insertIssueBarrierBefore(op, rewriter);
        continue;
      }
    }
  }
};

} // namespace mlir
