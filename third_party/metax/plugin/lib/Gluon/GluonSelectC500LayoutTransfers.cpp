#include "Gluon/Passes.h"
#include "Gluon/Targets/GluonC500Layout.h"

#include "mlir/Interfaces/LoopLikeInterface.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "triton/Tools/GenericSwizzling.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/raw_ostream.h"

#define DEBUG_TYPE "metax-gluon-select-c500-layout-transfers"
#define DBGS() (llvm::dbgs() << "[" DEBUG_TYPE "]: ")
#define LDBG(X) LLVM_DEBUG(DBGS() << X << "\n")

namespace tt = ::mlir::triton;
namespace ttg = ::mlir::triton::gpu;
namespace c500 = ::mlir::triton::gpu::metax::gluon::c500;

namespace mlir {

#define GEN_PASS_DEF_TRITONMETAXGPUGLUONSELECTC500LAYOUTTRANSFERS
#include "Gluon/Passes.h.inc"

namespace {

/// Provenance for the lowering attribute materialized by this pass. A plain
/// `mxg.shared_mem_force_no_vec` may be user-authored or owned by another pass
/// and must never be removed here.
constexpr StringLiteral kC500RepeatedSharedProvenance =
    "mxg.gluon_c500_repeated_shared";

static bool isTerminalGlobalStoreValue(Value value) {
  return !value.use_empty() && llvm::all_of(value.getUses(), [](OpOperand &use) {
           auto store = dyn_cast<tt::StoreOp>(use.getOwner());
           return store && store.getValue() == use.get();
         });
}

static bool isOutsideLoop(Operation *op) {
  return !op->getParentOfType<LoopLikeOpInterface>();
}

struct LayoutTransferRewrite {
  ttg::ConvertLayoutOp op;
  c500::LayoutTransferPlan plan;
  bool removeOwnedSelection = false;
};

static LogicalResult selectC500LayoutTransfers(ModuleOp module) {
  SmallVector<LayoutTransferRewrite> rewrites;
  WalkResult result = module.walk([&](ttg::ConvertLayoutOp op) {
    auto sourceType = dyn_cast<RankedTensorType>(op.getSrc().getType());
    auto resultType = dyn_cast<RankedTensorType>(op.getType());
    if (!sourceType || !resultType)
      return WalkResult::advance();

    c500::LayoutTransferRequirement requirement{
        sourceType, resultType, isTerminalGlobalStoreValue(op.getResult()),
        isOutsideLoop(op)};
    FailureOr<c500::LayoutTransferPlan> plan =
        c500::planLayoutTransfer(requirement, op);
    if (failed(plan))
      return WalkResult::interrupt();
    bool selected = plan->implementation ==
                    c500::LayoutTransferImplementation::RepeatedShared;
    bool owned = op->hasAttr(kC500RepeatedSharedProvenance);
    if (selected || owned)
      rewrites.push_back(
          {op, *plan, /*removeOwnedSelection=*/owned && !selected});
    return WalkResult::advance();
  });
  if (result.wasInterrupted())
    return failure();

  Builder builder(module.getContext());
  for (LayoutTransferRewrite &rewrite : rewrites) {
    if (rewrite.removeOwnedSelection) {
      rewrite.op->removeAttr(kC500RepeatedSharedProvenance);
      rewrite.op->removeAttr(tt::AttrSharedMemForceNoVec);
      LDBG("[layout-transfer-materialize] loc="
           << rewrite.op.getLoc() << ", implementation=generic"
           << ", action=remove-owned-selection");
      continue;
    }

    // Do not claim ownership of an attribute that was already supplied by a
    // user or another pass. Such an op already requests the selected lowering.
    if (!rewrite.op->hasAttr(tt::AttrSharedMemForceNoVec)) {
      rewrite.op->setAttr(tt::AttrSharedMemForceNoVec,
                          builder.getUnitAttr());
      rewrite.op->setAttr(kC500RepeatedSharedProvenance,
                          builder.getUnitAttr());
    }
    LDBG("[layout-transfer-materialize] loc="
         << rewrite.op.getLoc() << ", generic-scratch-elements="
         << rewrite.plan.genericMetrics.scratchElements
         << ", selected-scratch-elements="
         << rewrite.plan.selectedMetrics.scratchElements
         << ", generic-repetitions="
         << rewrite.plan.genericMetrics.repetitions
         << ", selected-repetitions="
         << rewrite.plan.selectedMetrics.repetitions
         << ", implementation=repeated-shared");
  }
  return success();
}

class TritonMETAXGPUGluonSelectC500LayoutTransfersPass
    : public impl::TritonMETAXGPUGluonSelectC500LayoutTransfersBase<
          TritonMETAXGPUGluonSelectC500LayoutTransfersPass> {
public:
  void runOnOperation() override {
    if (failed(selectC500LayoutTransfers(getOperation())))
      signalPassFailure();
  }
};

} // namespace

std::unique_ptr<Pass>
createTritonMETAXGPUGluonSelectC500LayoutTransfersPass() {
  return std::make_unique<
      TritonMETAXGPUGluonSelectC500LayoutTransfersPass>();
}

} // namespace mlir
