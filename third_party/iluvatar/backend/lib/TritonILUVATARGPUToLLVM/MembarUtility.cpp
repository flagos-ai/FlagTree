#include "TritonILUVATARGPUToLLVM/MembarUtility.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"

namespace mlir::triton::ILUVATAR {
namespace {
// Returns true if exactly one operand is an SME async copy and the other is a
// LocalLoad synced via AsyncWait, in which case the barrier membar would insert
// between them is redundant (see MembarUtility.h for the full rationale).
bool filterAsyncLocalLoadsDependencies(Operation *op1, Operation *op2) {
  // Only copies with both an SME blocked encoding and an explicit inputStride
  // are safe to filter. Block-pointer rewriting may attach inputStride to a
  // regular copy so the later SME eligibility pass can inspect it; inputStride
  // alone therefore does not prove that the copy uses the G2S engine.
  auto isSmeAsyncLoad = [](Operation *op) {
    auto cp = dyn_cast<triton::gpu::AsyncCopyGlobalToLocalOp>(op);
    return cp && cp.isIluvatarSmeAsyncCopy();
  };
  auto isLocalLoadSyncedViaAsyncWait = [](Operation *op) {
    auto ld = dyn_cast<triton::gpu::LocalLoadOp>(op);
    return ld && ld.getToken();
  };

  // Early return if neither or both operands are an SME async copy.
  if (isSmeAsyncLoad(op1) == isSmeAsyncLoad(op2))
    return false;

  return isLocalLoadSyncedViaAsyncWait(op1) ||
         isLocalLoadSyncedViaAsyncWait(op2);
}
} // namespace

bool membarFilter(Operation *op1, Operation *op2) {
  return filterAsyncLocalLoadsDependencies(op1, op2);
}

} // namespace mlir::triton::ILUVATAR
