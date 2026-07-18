// Copyright 2026 FlagOS Contributors
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.

#include "OptimizeLDSUtility.h"
#include "Analysis/HCUGPUAllocation.h"
#include "triton/Analysis/Allocation.h"
#include "triton/Dialect/Triton/IR/Utility.h"
#include "triton/Dialect/TritonGPU/IR/Attributes.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "llvm/Support/MathExtras.h"

namespace mlir::triton::HCU {

static void stepFactorizationPow2(std::vector<SmallVector<unsigned>> &factors,
                                  SmallVector<unsigned> &curFactor,
                                  int restTwos, int dim) {
  if (dim == curFactor.size()) {
    if (restTwos == 0)
      factors.push_back(curFactor);
    return;
  }
  curFactor[dim] = 1;
  for (int i = 0; i <= restTwos; ++i) {
    stepFactorizationPow2(factors, curFactor, restTwos - i, dim + 1);
    curFactor[dim] *= 2;
  }
}

std::vector<SmallVector<unsigned>> factorizePowerOf2(int n, int rank) {
  assert(llvm::isPowerOf2_32(n));
  int x = log2(n);
  std::vector<SmallVector<unsigned>> factors;
  SmallVector<unsigned> curFactor(rank, 1);
  stepFactorizationPow2(factors, curFactor, x, 0);
  return factors;
}

triton::gpu::DistributedEncodingTrait
createTmpLayout(triton::gpu::DistributedEncodingTrait layout,
                ArrayRef<unsigned> warpsPerCTA) {
  auto ctx = layout.getContext();
  if (auto src = dyn_cast<triton::gpu::AMDMfmaEncodingAttr>(layout))
    return triton::gpu::AMDMfmaEncodingAttr::get(
        ctx, src.getVersion(), warpsPerCTA, src.getInstrShape(),
        src.getIsTransposed(), src.getCTALayout(), src.getTilesPerWarp(),
        src.getElementBitWidth(), src.getMmacLayout());
  if (auto src = dyn_cast<triton::gpu::AMDWmmaEncodingAttr>(layout))
    return triton::gpu::AMDWmmaEncodingAttr::get(
        ctx, src.getVersion(), src.getIsTransposed(), warpsPerCTA,
        src.getTilesPerWarp(), src.getCTALayout(), src.getInstrShape());
  if (auto src = dyn_cast<triton::gpu::BlockedEncodingAttr>(layout))
    return triton::gpu::BlockedEncodingAttr::get(
        ctx, src.getSizePerThread(), src.getThreadsPerWarp(), warpsPerCTA,
        src.getOrder(), src.getCTALayout());
  if (auto src = dyn_cast<triton::gpu::DotOperandEncodingAttr>(layout)) {
    auto parent = cast<triton::gpu::DistributedEncodingTrait>(src.getParent());
    parent = createTmpLayout(parent, warpsPerCTA);
    if (!parent)
      return {};
    return triton::gpu::DotOperandEncodingAttr::get(ctx, src.getOpIdx(), parent,
                                                    src.getKWidth());
  }
  if (auto src = dyn_cast<triton::gpu::SliceEncodingAttr>(layout)) {
    auto warps = to_vector(warpsPerCTA);
    warps.insert(warps.begin() + src.getDim(), 1);
    auto parent = createTmpLayout(src.getParent(), warps);
    if (!parent)
      return {};
    return triton::gpu::SliceEncodingAttr::get(ctx, src.getDim(), parent);
  }
  // TODO: support linear layout if needed.
  if (isa<triton::gpu::LinearEncodingAttr>(layout))
    return {};
  assert(false && "Encountered unsupported layout");
  return {};
}

std::pair<triton::gpu::ConvertLayoutOp, triton::gpu::ConvertLayoutOp>
createNewConvertOps(OpBuilder &builder, triton::gpu::ConvertLayoutOp &cvtOp,
                    Attribute tmpLayout) {
  auto srcType = cvtOp.getSrc().getType();
  auto dstType = cvtOp.getType();

  auto newDstType = RankedTensorType::get(
      dstType.getShape(), dstType.getElementType(), dstType.getEncoding());
  RankedTensorType newSrcType = RankedTensorType::get(
      srcType.getShape(), srcType.getElementType(), tmpLayout);

  auto tmpCvt = triton::gpu::ConvertLayoutOp::create(
      builder, cvtOp.getLoc(), newSrcType, cvtOp.getSrc());
  auto newEpilogueCvt = triton::gpu::ConvertLayoutOp::create(
      builder, cvtOp.getLoc(), newDstType, tmpCvt);
  tmpCvt->setAttrs(cvtOp->getAttrs());
  newEpilogueCvt->setAttrs(cvtOp->getAttrs());

  return std::make_pair(tmpCvt, newEpilogueCvt);
}

Resources
estimateResourcesForReplacement(OpBuilder builder,
                                mlir::triton::gpu::ConvertLayoutOp cvtOp,
                                Attribute tmpLayout) {
  Resources res;
  RankedTensorType srcTy = cvtOp.getSrc().getType();
  RankedTensorType dstTy = cvtOp.getType();
  RankedTensorType intermediateTy = RankedTensorType::get(
      srcTy.getShape(), srcTy.getElementType(), tmpLayout);
  auto *ctx = cvtOp->getContext();

  int tmpCvtLDS = getConvertLayoutScratchInBytes(srcTy, intermediateTy,
                                                 /*usePadding*/ true);
  int tmpCvtLDSNoPad = getConvertLayoutScratchInBytes(srcTy, intermediateTy,
                                                      /*usePadding*/ false);
  int newCvtLDS = getConvertLayoutScratchInBytes(intermediateTy, dstTy,
                                                 /*usePadding*/ true);
  int newCvtLDSNoPad = getConvertLayoutScratchInBytes(intermediateTy, dstTy,
                                                      /*usePadding*/ false);

  res.LDSPad = std::max(tmpCvtLDS, newCvtLDS);
  res.LDSSwizzle = std::max(tmpCvtLDSNoPad, newCvtLDSNoPad);

  return res;
}

} // namespace mlir::triton::HCU
