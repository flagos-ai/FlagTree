#include <memory>

#include "Allocation.h"
#include "LoweringUtility.h"
#include "Passes.h"
#include "TargetInfo.h"
#include "triton/Analysis/Allocation.h"
#include "triton/Conversion/TritonGPUToLLVM/AllocateSharedMemoryUtility.h"
#include "triton/Conversion/TritonGPUToLLVM/Utility.h"
#include "triton/Dialect/Triton/IR/Utility.h"
#include "triton/Dialect/TritonInstrument/IR/ConSanConstants.h"
#include "triton/Dialect/TritonNvidiaGPU/IR/Dialect.h"
#include "triton/Tools/GenericSwizzling.h"
#include "triton/Tools/LayoutUtils.h"

using namespace mlir;
using namespace mlir::triton;

namespace {
struct GPNPUAllocateSharedMemoryPass
    : public PassWrapper<GPNPUAllocateSharedMemoryPass,
                         OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(GPNPUAllocateSharedMemoryPass)

  StringRef getArgument() const final { return "gpnpu-allocate-shared-memory"; }
  StringRef getDescription() const final {
    return "assign GPNPU shared-memory scratch sizes and offsets";
  }
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<mlir::triton::TritonDialect,
                    mlir::triton::gpu::TritonGPUDialect,
                    mlir::triton::nvidia_gpu::TritonNvidiaGPUDialect>();
  }

  void runOnOperation() override {
    ModuleOp mod = getOperation();
    int numWarps = mlir::triton::gpu::lookupNumWarps(mod);
    mlir::triton::Intellifusion::TargetInfo targetInfo(/*threadsPerWarp=*/32,
                                                       numWarps);
    ModuleAllocation allocation(
        mod,
        mlir::triton::Intellifusion::getGPNPUAllocationAnalysisScratchSizeFn(
            targetInfo));
    mlir::triton::gpu::attachAllocationSizeAndOffsetAttr(mod, allocation);
  }
};
} // namespace

namespace mlir::triton::Intellifusion {

using namespace mlir::triton::nvidia_gpu;

static unsigned getNumScratchElemsSwizzledCvt(RankedTensorType srcTy,
                                              RankedTensorType dstTy,
                                              TargetInfoBase &targetInfo) {
  auto *ctx = srcTy.getContext();
  auto srcLayout = triton::gpu::toLinearLayout(srcTy);
  auto dstLayout = triton::gpu::toLinearLayout(dstTy);
  srcLayout = actionRemoveBroadcastedRegs(srcLayout).apply(srcLayout);
  dstLayout = actionRemoveBroadcastedRegs(dstLayout).apply(dstLayout);
  auto bitwidth = getBitwidth(srcTy);
  auto [smem, _] =
      getLayoutConversionSwizzling(srcLayout, dstLayout, bitwidth, targetInfo);
  auto reps = smem.getInDimSize(StringAttr::get(ctx, "reps"));
  // The smem has the same cta layout as the srcLayout, so we use that instead
  // We remove the number of elements that are duplicated in the cta layout
  auto nBlocks = product(triton::gpu::getCTASplitNum(srcTy.getEncoding()));
  return smem.getTotalOutDimSize() / (reps * nBlocks);
}

std::function<unsigned(Operation *)>
getGPNPUAllocationAnalysisScratchSizeFn(TargetInfoBase &targetInfo) {
  auto allocation = [&targetInfo](Operation *op) -> unsigned {
    if (auto cvtOp = dyn_cast<triton::gpu::ConvertLayoutOp>(op)) {
      auto srcTy = cvtOp.getSrc().getType();
      auto dstTy = cvtOp.getType();
      if (!cvtNeedsSharedMemory(srcTy, dstTy))
        return 0;
      // IFWA always uses swizzled scratch storage for layout conversions.
      auto elems = getNumScratchElemsSwizzledCvt(srcTy, dstTy, targetInfo);
      return elems * getBitwidth(srcTy) / 8;
    }
    if (auto ws = dyn_cast<triton::gpu::WarpSpecializeOp>(op)) {
      unsigned captureSize = defaultAllocationAnalysisScratchSizeFn(op);
      // ConSan adds captures after allocation; reserve space pre-computed by
      // the common TritonInstrumentPrepareConSanCaptures pass.
      if (auto extra = ws->getAttrOfType<IntegerAttr>(
              mlir::triton::instrument::kConSanExtraCaptureBytesAttr))
        captureSize += extra.getInt();
      return captureSize;
    }
    return defaultAllocationAnalysisScratchSizeFn(op);
  };
  return allocation;
}
std::unique_ptr<Pass> createGPNPUAllocateSharedMemoryPass() {
  return std::make_unique<GPNPUAllocateSharedMemoryPass>();
}

} // namespace mlir::triton::Intellifusion
