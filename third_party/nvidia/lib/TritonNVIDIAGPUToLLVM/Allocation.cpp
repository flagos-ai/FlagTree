#include <algorithm>
#include <limits>
#include <memory>

#include "Allocation.h"
#include "TargetInfo.h"
#include "TritonNVIDIAGPUToLLVM/Passes.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "triton/Analysis/Allocation.h"
#include "triton/Analysis/Utility.h"
#include "triton/Conversion/TritonGPUToLLVM/AllocateSharedMemoryUtility.h"
#include "triton/Conversion/TritonGPUToLLVM/Utility.h"
#include "triton/Dialect/Triton/IR/Utility.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "triton/Tools/GenericSwizzling.h"
#include "triton/Tools/LayoutUtils.h"
#ifdef __TLE__
#include "tle/dialect/include/IR/Dialect.h"
#endif

using namespace mlir;
using namespace mlir::triton;

namespace mlir {
namespace triton {
#define GEN_PASS_DEF_ALLOCATESHAREDMEMORYNV
#include "TritonNVIDIAGPUToLLVM/Passes.h.inc"
} // namespace triton
} // namespace mlir

namespace {
struct AllocateSharedMemoryNv
    : public mlir::triton::impl::AllocateSharedMemoryNvBase<
          AllocateSharedMemoryNv> {
  using AllocateSharedMemoryNvBase::AllocateSharedMemoryNvBase;

  AllocateSharedMemoryNv(int32_t computeCapability, int32_t ptxVersion)
      : AllocateSharedMemoryNvBase({computeCapability, ptxVersion}) {}

  void runOnOperation() override {
    ModuleOp mod = getOperation();
    mlir::triton::NVIDIA::TargetInfo targetInfo(computeCapability, ptxVersion);
    ModuleAllocation allocation(
        mod, mlir::triton::nvidia_gpu::getNvidiaAllocationAnalysisScratchSizeFn(
                 targetInfo));
    mlir::triton::gpu::attachAllocationSizeAndOffsetAttr(mod, allocation);
  }
};
} // namespace

namespace mlir::triton::nvidia_gpu {

namespace {
#ifdef __TLE__
static bool isTleCtaOrReduceFastpathCandidate(ReduceOp reduceOp) {
  if (reduceOp.getNumOperands() != 1 || reduceOp.getNumResults() != 1)
    return false;
  if (!reduceOp.getResult()[0].getType().isInteger(1))
    return false;
  auto *combine = reduceOp.getSingleCombiner();
  if (!combine || !isa<arith::OrIOp>(combine))
    return false;

  ReduceOpHelper helper(reduceOp);
  if (!helper.isReduceWithinCTA())
    return false;
  if (helper.isWarpSynchronous())
    return false;
  return true;
}
#endif
} // namespace

static unsigned getNumScratchElemsSwizzledCvt(RankedTensorType srcTy,
                                              RankedTensorType dstTy,
                                              TargetInfoBase &targetInfo) {
  auto *ctx = srcTy.getContext();
  auto srcLayout = triton::gpu::toLinearLayout(srcTy);
  auto dstLayout = triton::gpu::toLinearLayout(dstTy);
  srcLayout = actionRemoveBroadcastedRegs(srcLayout).apply(srcLayout);
  dstLayout = actionRemoveBroadcastedRegs(dstLayout).apply(dstLayout);
  auto bitwidth = getBitwidth(srcTy);
  auto [srcTiles, dstTiles] = gpu::getSrcDstTiles(targetInfo, bitwidth);
  auto [smem, _] = triton::gpu::optimalSwizzling(srcLayout, dstLayout, srcTiles,
                                                 dstTiles, bitwidth);
  auto reps = smem.getInDimSize(StringAttr::get(ctx, "reps"));
  return smem.getTotalOutDimSize() / reps;
}

std::function<unsigned(Operation *)>
getNvidiaAllocationAnalysisScratchSizeFn(TargetInfoBase &targetInfo) {
  auto allocation = [&targetInfo](Operation *op) -> unsigned {
#ifdef __TLE__
    if (auto cumsumOp = dyn_cast<mlir::triton::tle::ExclusiveCumsumOp>(op)) {
      auto srcTy = dyn_cast<RankedTensorType>(cumsumOp.getSrc().getType());
      if (!srcTy || srcTy.getRank() != 1)
        return 0;
      int64_t axisExtent = srcTy.getShape()[0];
      if (ShapedType::isDynamic(axisExtent) || axisExtent <= 0)
        return 0;
      unsigned elemBytes =
          static_cast<unsigned>(std::max<int>(1, getBitwidth(srcTy) / 8));
      // Scratch layout for cumsum lowering:
      // [axisExtent data][numWarps warp-prefix slots][1 total slot]
      int64_t numWarps = std::max<int64_t>(1, triton::gpu::lookupNumWarps(op));
      uint64_t totalBytes = (static_cast<uint64_t>(axisExtent) +
                             static_cast<uint64_t>(numWarps) + 1ull) *
                            elemBytes;
      if (totalBytes > std::numeric_limits<unsigned>::max())
        return 0;
      return static_cast<unsigned>(totalBytes);
    }
#endif
    if (auto reduceOp = dyn_cast<ReduceOp>(op)) {
#ifdef __TLE__
      // TLE fastpath lowers CTA-wide i1 OR reduce directly to bar.red.or.pred,
      // so no shared scratch allocation is needed for this op.
      if (isTleCtaOrReduceFastpathCandidate(reduceOp))
        return 0;
#endif
    }
    if (auto cvtOp = dyn_cast<triton::gpu::ConvertLayoutOp>(op)) {
      auto srcTy = cvtOp.getSrc().getType();
      auto dstTy = cvtOp.getType();
      if (!cvtNeedsSharedMemory(srcTy, dstTy))
        return 0;
      // In cuda we always swizzle
      auto elems = getNumScratchElemsSwizzledCvt(srcTy, dstTy, targetInfo);
      return elems * getBitwidth(srcTy) / 8;
    }
    return defaultAllocationAnalysisScratchSizeFn(op);
  };
  return allocation;
}
} // namespace mlir::triton::nvidia_gpu

namespace mlir::triton {
std::unique_ptr<OperationPass<ModuleOp>>
createAllocateSharedMemoryNvPass(int32_t computeCapability,
                                 int32_t ptxVersion) {
  return std::make_unique<AllocateSharedMemoryNv>(computeCapability,
                                                  ptxVersion);
}
} // namespace mlir::triton
