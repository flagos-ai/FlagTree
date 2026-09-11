#ifndef TRITON_METAX_GLUON_C500_ASYNC_COPY_PLAN_H
#define TRITON_METAX_GLUON_C500_ASYNC_COPY_PLAN_H

#include "Gluon/GluonC500AsyncCopyLayout.h"

#include "mlir/Support/LLVM.h"
#include "mlir/Support/LogicalResult.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include <cstdint>
#include <optional>

namespace mlir::triton::gpu::metax::gluon::c500 {

struct AsyncCopyTargetCapabilities {
  bool allowPredicatedLaneIssuer = false;
  bool allowPredicatedWarpIssuer = false;
  bool allowPredicatedBlockIssuer = false;
};

inline constexpr AsyncCopyTargetCapabilities kC500AsyncCopyCapabilities{};

struct AsyncCopyIssuePlan {
  triton::gpu::BlockedEncodingAttr sourceEncoding;
  triton::gpu::SwizzledSharedEncodingAttr sharedEncoding;
  unsigned contiguousDim = 0;
  unsigned maxContiguousElements = 0;
  unsigned copyElements = 0;
  unsigned copyBytes = 0;
  unsigned instructionsPerThread = 0;
  unsigned registerRepeats = 0;
  uint32_t freeLaneMask = 0;
  uint32_t freeWarpMask = 0;
  uint32_t freeBlockMask = 0;
  bool requiresSourceSideMaskSwizzle = false;
};

SmallVector<triton::gpu::BlockedEncodingAttr>
getC500AsyncCopyLegalSourceEncodings(
    triton::gpu::AsyncCopyGlobalToLocalOp copyOp, ArrayRef<unsigned> order,
    AsyncCopyAddressContiguity addressContiguity,
    triton::gpu::SwizzledSharedEncodingAttr sharedEncoding);

FailureOr<AsyncCopyIssuePlan> analyzeC500AsyncCopyIssuePlan(
    triton::gpu::AsyncCopyGlobalToLocalOp copyOp,
    triton::gpu::BlockedEncodingAttr sourceEncoding,
    AsyncCopyAddressContiguity addressContiguity,
    triton::gpu::SwizzledSharedEncodingAttr sharedEncoding);

FailureOr<AsyncCopyIssuePlan> selectC500AsyncCopyIssuePlan(
    triton::gpu::AsyncCopyGlobalToLocalOp copyOp, ArrayRef<unsigned> sourceOrder,
    AsyncCopyAddressContiguity addressContiguity,
    triton::gpu::SwizzledSharedEncodingAttr sharedEncoding);

FailureOr<AsyncCopyIssuePlan>
getC500AsyncCopyIssuePlan(triton::gpu::AsyncCopyGlobalToLocalOp copyOp);

/// Construct one async-copy issue plan. A concrete shared encoding is a hard
/// destination contract; otherwise the target default is derived from the
/// proven source order. Source instruction width is selected by the physical
/// issue planner from the C500 ISA widths, not by a workload cost model.
FailureOr<AsyncCopyIssuePlan> inferC500AsyncCopyIssuePlan(
    triton::gpu::AsyncCopyGlobalToLocalOp copyOp,
    ArrayRef<unsigned> sourceOrder,
    AsyncCopyAddressContiguity addressContiguity,
    Attribute requiredSharedEncoding = {});

LogicalResult verifyC500AsyncCopyIssuePlan(
    triton::gpu::AsyncCopyGlobalToLocalOp copyOp);

} // namespace mlir::triton::gpu::metax::gluon::c500

#endif // TRITON_METAX_GLUON_C500_ASYNC_COPY_PLAN_H
