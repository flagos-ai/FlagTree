#include "Gluon/GluonLayoutPlaceholders.h"
#include "Gluon/GluonC500AsyncCopyLayout.h"
#include "Gluon/GluonC500LayoutHelpers.h"
#include "Gluon/GluonC500AsyncCopyPlan.h"

#include "triton/Dialect/Gluon/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/Transforms/Utility.h"
#include "llvm/Support/Debug.h"

#define DEBUG_TYPE "metax-gluon-c500-async-copy-layout"
#define DBGS() (llvm::dbgs() << "[" DEBUG_TYPE "]: ")
#define LDBG(X) LLVM_DEBUG(DBGS() << X << "\n")

namespace tt = ::mlir::triton;
namespace ttg = ::mlir::triton::gpu;

namespace mlir::triton::gpu::metax::gluon::c500 {

static Attribute getDefaultAsyncCopySharedEncoding(
    ttg::AsyncCopyGlobalToLocalOp copyOp, ArrayRef<unsigned> order) {
  auto dstTy = cast<ttg::MemDescType>(copyOp->getOperand(1).getType());
  auto ctaLayout =
      ttg::CTAEncodingAttr::getDefault(copyOp.getContext(), dstTy.getRank());
  return ttg::SwizzledSharedEncodingAttr::get(copyOp.getContext(), 1, 1, 1,
                                              order, ctaLayout);
}

FailureOr<AsyncCopyIssuePlan> inferC500AsyncCopyIssuePlan(
    ttg::AsyncCopyGlobalToLocalOp copyOp, ArrayRef<unsigned> srcOrder,
    AsyncCopyAddressContiguity addressContiguity,
    Attribute requiredSharedEncoding) {
  if (srcOrder.size() != 2 || !addressContiguity.isProven() ||
      srcOrder.front() != addressContiguity.contiguousDim) {
    LDBG("[async-copy-issue-reject] loc="
         << copyOp.getLoc()
         << ", reason=unproven rank-2 address contiguity");
    return failure();
  }

  Attribute sharedRequirement = unwrapNoVerifyEncoding(requiredSharedEncoding);
  auto sharedEncoding = dyn_cast_or_null<ttg::SwizzledSharedEncodingAttr>(
      sharedRequirement ? sharedRequirement
                        : getDefaultAsyncCopySharedEncoding(copyOp, srcOrder));
  if (!sharedEncoding) {
    LDBG("[async-copy-issue-reject] loc="
         << copyOp.getLoc() << ", reason=non-swizzled shared requirement, "
         << "encoding=" << sharedRequirement);
    return failure();
  }

  FailureOr<AsyncCopyIssuePlan> plan = selectC500AsyncCopyIssuePlan(
      copyOp, srcOrder, addressContiguity, sharedEncoding);
  if (failed(plan)) {
    LDBG("[async-copy-issue-reject] loc="
         << copyOp.getLoc() << ", source-order=[" << srcOrder[0] << ", "
         << srcOrder[1] << "], shared=" << sharedEncoding
         << ", reason=no legal 16/8/4-byte issue width");
    return failure();
  }
  LDBG("[async-copy-issue] source="
       << plan->sourceEncoding << " shared=" << plan->sharedEncoding
       << " contiguous-dim=" << plan->contiguousDim
       << " max-contiguous-elements=" << plan->maxContiguousElements
       << " copy-bytes=" << plan->copyBytes
       << " instructions-per-thread=" << plan->instructionsPerThread);
  return plan;
}

LogicalResult materializeAsyncCopyFollowerLayouts(tt::FuncOp func) {
  WalkResult result = func.walk([&](ttg::AsyncCopyGlobalToLocalOp copyOp) {
    auto srcTy = dyn_cast<RankedTensorType>(copyOp.getSrc().getType());
    if (!srcTy || !srcTy.getEncoding() || hasAutoEncoding(copyOp.getSrc()))
      return WalkResult::advance();
    Attribute srcEncoding = unwrapNoVerifyEncoding(srcTy.getEncoding());
    OpBuilder builder(copyOp);

    auto convertOperand = [&](Value operand,
                              MutableOperandRange mutableOperand) {
      auto operandTy = dyn_cast<RankedTensorType>(operand.getType());
      if (!operandTy || unwrapNoVerifyEncoding(operandTy.getEncoding()) ==
                            srcEncoding)
        return;
      auto targetTy =
          cast<RankedTensorType>(cloneTypeWithEncoding(operandTy, srcEncoding));
      auto converted = builder.create<ttg::ConvertLayoutOp>(
          copyOp.getLoc(), targetTy, operand);
      mutableOperand.assign(converted.getResult());
    };

    if (copyOp.getMask())
      convertOperand(copyOp.getMask(), copyOp.getMaskMutable());
    if (copyOp.getOther())
      convertOperand(copyOp.getOther(), copyOp.getOtherMutable());
    return WalkResult::advance();
  });
  return result.wasInterrupted() ? failure() : success();
}

LogicalResult verifyAsyncCopyLayouts(tt::FuncOp func) {
  WalkResult result = func.walk([&](ttg::AsyncCopyGlobalToLocalOp copyOp) {
    auto srcTy = dyn_cast<RankedTensorType>(copyOp.getSrc().getType());
    if (!srcTy || !srcTy.getEncoding()) {
      copyOp.emitError() << "async_copy source has no concrete layout";
      return WalkResult::interrupt();
    }
    Attribute srcEncoding = unwrapNoVerifyEncoding(srcTy.getEncoding());

    auto verifyFollower = [&](Value value, StringRef name) -> LogicalResult {
      if (!value)
        return success();
      auto valueTy = dyn_cast<RankedTensorType>(value.getType());
      if (!valueTy)
        return success();
      if (unwrapNoVerifyEncoding(valueTy.getEncoding()) != srcEncoding)
        return copyOp.emitError()
               << "async_copy " << name
               << " layout does not match source layout";
      return success();
    };

    if (failed(verifyFollower(copyOp.getMask(), "mask")) ||
        failed(verifyFollower(copyOp.getOther(), "other")))
      return WalkResult::interrupt();

    if (failed(verifyC500AsyncCopyContiguousSharedWriteLegalizable(copyOp)))
      return WalkResult::interrupt();
    return WalkResult::advance();
  });
  return result.wasInterrupted() ? failure() : success();
}

} // namespace mlir::triton::gpu::metax::gluon::c500
