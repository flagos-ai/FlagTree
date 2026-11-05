#include "LoweringUtility.h"

#include "mlir/Dialect/LLVMIR/NVVMDialect.h"
#include "nvidia/include/Dialect/NVGPU/IR/Dialect.h"
#include "triton/Conversion/TritonGPUToLLVM/TargetInfoBase.h"
#include "triton/Dialect/TritonGPU/Transforms/Utility.h"
#include "triton/Dialect/TritonNvidiaGPU/IR/Dialect.h"
#include "triton/Tools/GenericSwizzling.h"
#include "llvm/Support/MathExtras.h"

namespace mlir::triton::Intellifusion {

std::pair<LinearLayout, std::pair<int32_t, int32_t>>
getLayoutConversionSwizzling(const LinearLayout &src, const LinearLayout &dst,
                             int32_t bitwidth,
                             const TargetInfoBase &targetInfo) {
  auto *ctx = src.getInDimNames().begin()->getContext();
  auto kBlock = StringAttr::get(ctx, "block");
  bool crossCTA = !dst.invertAndCompose(src).isTrivialOver({kBlock});
  auto [srcTiles, dstTiles] =
      gpu::getSrcDstTiles(targetInfo, bitwidth, crossCTA);
  auto [smem, instr] =
      gpu::optimalSwizzling(src, dst, srcTiles, dstTiles, bitwidth);
  if (instr.first == 0 && instr.second == 0)
    return {smem, instr};

  // GPNPU implements warpSync with a CTA barrier. Batch matrix operations up to
  // the ordinary shared transfer's tile size to avoid extra synchronization,
  // without exceeding the larger of the two existing scratch requirements.
  auto vecBitwidth = gpu::getVecBitwidthLdSt(src, dst, bitwidth);
  auto [dstTile, srcTile] = targetInfo.getSharedLdStTiles(vecBitwidth);
  auto generic = gpu::optimalSwizzlingLdSt(
      src, dst, bitwidth, targetInfo.getSharedMemoryBanks(), srcTile, dstTile);
  auto kReps = StringAttr::get(ctx, "reps");
  auto numReps = smem.getInDimSize(kReps);
  auto genericReps = generic.getInDimSize(kReps);
  if (numReps <= genericReps)
    return {smem, instr};

  // Repetition bases are registers in both layouts. Move their low bits into
  // the shared segment so these tiles use distinct addresses within one round.
  // Keep the vector and bank mappings, and therefore the matrix instructions.
  auto bases = smem.getBases();
  auto &reps = bases[kReps];
  auto &segment = bases[StringAttr::get(ctx, "segment")];
  auto batchBits = llvm::Log2_32(numReps / genericReps);
  segment.insert(segment.end(), reps.begin(), reps.begin() + batchBits);
  reps.erase(reps.begin(), reps.begin() + batchBits);
  return {LinearLayout(std::move(bases), smem.getOutDims(),
                       /*requireSurjective=*/true),
          instr};
}

Value createElectPredicate(Location loc, OpBuilder &rewriter) {
  return NVVM::ElectSyncOp::create(rewriter, loc, i1_ty,
                                   /*membermask=*/Value());
}

Value createElectPredicateWarp0(Location loc, OpBuilder &rewriter) {
  TritonLLVMOpBuilder b(loc, rewriter);
  Value warpId = getLaneAndWarpId(rewriter, loc).second;
  Value warp0 = b.icmp_eq(warpId, b.i32_val(0));
  return b.and_(warp0, createElectPredicate(loc, rewriter));
}

Value createTMAMulticastMask(Location loc, ConversionPatternRewriter &rewriter,
                             uint16_t broadcastBits) {
  int numCTAs = gpu::lookupNumCTAs(rewriter);
  auto encoding =
      nvidia_gpu::getTMAMulticastMaskEncoding(numCTAs, broadcastBits);
  TritonLLVMOpBuilder b(loc, rewriter);
  Value ctaId = nvgpu::ClusterCTAIdOp::create(rewriter, loc);
  Value base = b.and_(ctaId, b.i32_val(encoding.fixedBits));
  return b.shl(b.i32_val(encoding.pattern), base);
}

Value createLeadCTAPredicate(Location loc, RewriterBase &rewriter) {
  TritonLLVMOpBuilder b(loc, rewriter);
  Value clusterId = nvgpu::ClusterCTAIdOp::create(rewriter, loc);
  clusterId = b.and_(clusterId, b.i32_val(1));
  return b.icmp_eq(clusterId, b.i32_val(0));
}

} // namespace mlir::triton::Intellifusion
