/*
 * Copyright (c) 2026 T-Head Semiconductor Co., Ltd. All rights reserved.
 *
 * Permission is hereby granted, free of charge, to any person obtaining
 * a copy of this software and associated documentation files
 * (the "Software"), to deal in the Software without restriction,
 * including without limitation the rights to use, copy, modify, merge,
 * publish, distribute, sublicense, and/or sell copies of the Software,
 * and to permit persons to whom the Software is furnished to do so,
 * subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be
 * included in all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
 * EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
 * MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
 * IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY
 * CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,
 * TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE
 * SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
 */

#include <algorithm>
#include <limits>
#include <memory>

#include "Allocation.h"
#include "TargetInfo.h"
#include "TritonPPUGPUToLLVM/Passes.h"
#include "triton/Analysis/Allocation.h"
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
#define GEN_PASS_DEF_ALLOCATESHAREDMEMORYPPU
#include "TritonPPUGPUToLLVM/Passes.h.inc"
} // namespace triton
} // namespace mlir

namespace {
struct AllocateSharedMemoryPPU
    : public mlir::triton::impl::AllocateSharedMemoryPPUBase<
          AllocateSharedMemoryPPU> {
  using AllocateSharedMemoryPPUBase::AllocateSharedMemoryPPUBase;

  AllocateSharedMemoryPPU(int32_t computeCapability)
      : AllocateSharedMemoryPPUBase({computeCapability}) {}

  void runOnOperation() override {
    ModuleOp mod = getOperation();
    mlir::triton::ppu::TargetInfo targetInfo(computeCapability);
    ModuleAllocation allocation(
        mod, mlir::triton::ppu_gpu::getPPUAllocationAnalysisScratchSizeFn(
                 targetInfo));
    mlir::triton::gpu::attachAllocationSizeAndOffsetAttr(mod, allocation);
  }
};
} // namespace

namespace mlir::triton::ppu_gpu {

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
getPPUAllocationAnalysisScratchSizeFn(TargetInfoBase &targetInfo) {
  auto allocation = [&targetInfo](Operation *op) -> unsigned {
#ifdef __TLE__
    // TLE ops that may fall back to a SMEM-relay lowering need scratch
    // shared memory sized here, or getSharedMemoryBase will assert when the
    // conversion pattern fires. Mirrors nvidia/lib/TritonNVIDIAGPUToLLVM/
    // Allocation.cpp; reduce fastpath is NV-only and intentionally skipped.
    if (auto cumsumOp = dyn_cast<mlir::triton::tle::ExclusiveCumsumOp>(op)) {
      auto srcTy = dyn_cast<RankedTensorType>(cumsumOp.getSrc().getType());
      if (!srcTy || srcTy.getRank() != 1)
        return 0;
      int64_t axisExtent = srcTy.getShape()[0];
      if (ShapedType::isDynamic(axisExtent) || axisExtent <= 0)
        return 0;
      unsigned elemBytes =
          static_cast<unsigned>(std::max<int>(1, getBitwidth(srcTy) / 8));
      // Scratch layout: [axisExtent data][numWarps warp-prefix slots][1 total]
      int64_t numWarps = std::max<int64_t>(1, triton::gpu::lookupNumWarps(op));
      uint64_t totalBytes = (static_cast<uint64_t>(axisExtent) +
                             static_cast<uint64_t>(numWarps) + 1ull) *
                            elemBytes;
      if (totalBytes > std::numeric_limits<unsigned>::max())
        return 0;
      return static_cast<unsigned>(totalBytes);
    }
#endif
    if (auto cvtOp = dyn_cast<triton::gpu::ConvertLayoutOp>(op)) {
      auto srcTy = cvtOp.getSrc().getType();
      auto dstTy = cvtOp.getType();
      if (!cvtNeedsSharedMemory(srcTy, dstTy))
        return 0;
      // In hggc we always swizzle
      auto elems = getNumScratchElemsSwizzledCvt(srcTy, dstTy, targetInfo);
      return elems * getBitwidth(srcTy) / 8;
    }
#ifdef __TLE__
    if (auto extractTileOp = dyn_cast<triton::tle::ExtractTileOp>(op)) {
      auto dstTy = dyn_cast<RankedTensorType>(extractTileOp.getType());
      if (!dstTy)
        return 0;
      return static_cast<unsigned>(dstTy.getNumElements() *
                                   (getBitwidth(dstTy) / 8));
    }
    if (auto insertTileOp = dyn_cast<triton::tle::InsertTileOp>(op)) {
      auto tileTy =
          dyn_cast<RankedTensorType>(insertTileOp.getTile().getType());
      if (!tileTy)
        return 0;
      return static_cast<unsigned>(tileTy.getNumElements() *
                                   (getBitwidth(tileTy) / 8));
    }
#endif
    return defaultAllocationAnalysisScratchSizeFn(op);
  };
  return allocation;
}
} // namespace mlir::triton::ppu_gpu

namespace mlir::triton {
std::unique_ptr<OperationPass<ModuleOp>>
createAllocateSharedMemoryPPUPass(int32_t computeCapability) {
  return std::make_unique<AllocateSharedMemoryPPU>(computeCapability);
}
} // namespace mlir::triton
