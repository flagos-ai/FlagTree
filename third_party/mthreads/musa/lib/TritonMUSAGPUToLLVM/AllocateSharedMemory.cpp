#include "TritonMUSAGPUToLLVM/Allocation.h"
#include "TritonMUSAGPUToLLVM/Passes.h"
#include "TritonMUSAGPUToLLVM/TargetInfo.h"
#include "triton/Analysis/Allocation.h"
#include "triton/Conversion/TritonGPUToLLVM/AllocateSharedMemoryUtility.h"
#include "triton/Conversion/TritonGPUToLLVM/Utility.h"
#include "triton/Dialect/Triton/IR/Utility.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "triton/Tools/GenericSwizzling.h"
#include "triton/Tools/LayoutUtils.h"

using namespace mlir;
using namespace mlir::triton;
using namespace mlir::triton::gpu;

namespace mlir::triton {
#define GEN_PASS_DEF_ALLOCATEMUSASHAREDMEMORY
#include "musa/include/TritonMUSAGPUToLLVM/Passes.h.inc"
} // namespace mlir::triton

namespace {

static bool isMusaSqmmaLike(Attribute layout) {
  return isa<MUSASqmmaEncodingAttr>(layout);
}

static bool useMusaReplicatedScratch(Attribute srcLayout, Attribute dstLayout) {
  return (isMusaSqmmaLike(srcLayout) || isMusaSqmmaLike(dstLayout)) &&
         isa<MmaEncodingTrait, BlockedEncodingAttr, SliceEncodingAttr>(
             srcLayout) &&
         isa<MmaEncodingTrait, BlockedEncodingAttr, SliceEncodingAttr>(
             dstLayout);
}

static bool isSqmmaAccumulatorToBlockedLike(Attribute srcLayout,
                                            Attribute dstLayout) {
  return isa<MUSASqmmaEncodingAttr>(srcLayout) &&
         isa<BlockedEncodingAttr, SliceEncodingAttr>(dstLayout);
}

static bool useMusaSqmmaBlockSwizzling(RankedTensorType srcTy,
                                       RankedTensorType dstTy) {
  if (!(isMusaSqmmaLike(srcTy.getEncoding()) ||
        isMusaSqmmaLike(dstTy.getEncoding())))
    return false;

  LinearLayout conversion = minimalCvtLayout(srcTy, dstTy);
  MLIRContext *ctx = srcTy.getContext();
  StringAttr kBlock = str_attr("block");
  StringAttr kWarp = str_attr("warp");
  StringAttr kLane = str_attr("lane");
  auto dims = conversion.getInDimNames();

  if (llvm::is_contained(dims, kBlock))
    return false;
  if (llvm::is_contained(dims, kWarp))
    return true;
  if (llvm::is_contained(dims, kLane))
    return !cvtNeedsWarpShuffle(srcTy, dstTy);
  return false;
}

static bool isPlainBlockedLike(Attribute layout) {
  return isa<BlockedEncodingAttr, SliceEncodingAttr>(layout);
}

static bool useConservativeCarrierScratch(RankedTensorType srcTy,
                                          RankedTensorType dstTy) {
  auto srcElemTy = srcTy.getElementType();
  auto dstElemTy = dstTy.getElementType();

  auto needsByteCarrier = [](Type ty) {
    return ty.isIntOrFloat() && ty.getIntOrFloatBitWidth() < 8;
  };
  bool isPointerCarrier = isa<triton::PointerType>(srcElemTy) &&
                          isa<triton::PointerType>(dstElemTy);
  bool isSubByteCarrier =
      needsByteCarrier(srcElemTy) && needsByteCarrier(dstElemTy);
  if (!isPointerCarrier && !isSubByteCarrier)
    return false;

  if (!isa<BlockedEncodingAttr, SliceEncodingAttr>(srcTy.getEncoding()) ||
      !isa<BlockedEncodingAttr, SliceEncodingAttr>(dstTy.getEncoding()))
    return false;

  LinearLayout conversion = minimalCvtLayout(srcTy, dstTy);
  MLIRContext *ctx = srcTy.getContext();
  StringAttr kBlock = str_attr("block");
  return !llvm::is_contained(conversion.getInDimNames(), kBlock);
}

static bool useMusaGenericBlockSwizzling(RankedTensorType srcTy,
                                         RankedTensorType dstTy) {
  if (isMusaSqmmaLike(srcTy.getEncoding()) ||
      isMusaSqmmaLike(dstTy.getEncoding()))
    return false;
  if (!cvtNeedsSharedMemory(srcTy, dstTy))
    return false;

  LinearLayout conversion = minimalCvtLayout(srcTy, dstTy);
  MLIRContext *ctx = srcTy.getContext();
  StringAttr kBlock = str_attr("block");
  return !llvm::is_contained(conversion.getInDimNames(), kBlock);
}

static LinearLayout
getMusaSwizzledScratchLayout(RankedTensorType srcTy, RankedTensorType dstTy,
                             const TargetInfoBase &targetInfo) {
  auto srcLayout = toLinearLayout(srcTy);
  auto dstLayout = toLinearLayout(dstTy);
  srcLayout = actionRemoveBroadcastedRegs(srcLayout).apply(srcLayout);
  dstLayout = actionRemoveBroadcastedRegs(dstLayout).apply(dstLayout);
  auto bitwidth = getBitwidth(srcTy);
  auto [srcTiles, dstTiles] = getSrcDstTiles(targetInfo, bitwidth);
  auto [smem, _] =
      optimalSwizzling(srcLayout, dstLayout, srcTiles, dstTiles, bitwidth);
  return smem;
}

static bool hasPH1PhysicalSliceRep(const LinearLayout &smem) {
  constexpr int32_t kPH1PhysicalSliceRows = 32;

  auto *ctx = smem.getInDimNames().begin()->getContext();
  auto kReps = StringAttr::get(ctx, "reps");
  auto outDims = smem.getOutDims();
  if (outDims.size() < 2)
    return false;

  auto isPowerOfTwo = [](int32_t value) {
    return value > 0 && (value & (value - 1)) == 0;
  };

  for (const auto &repBasis : smem.getBases().lookup(kReps)) {
    for (auto [dim, component] : llvm::enumerate(repBasis)) {
      if (component == 0)
        continue;
      if (dim + 1 >= outDims.size())
        continue;
      if (!isPowerOfTwo(component) || component < kPH1PhysicalSliceRows)
        continue;

      bool selectsWholeInnerSlice = true;
      for (auto inner = dim + 1; inner < repBasis.size(); ++inner) {
        if (repBasis[inner] != 0) {
          selectsWholeInnerSlice = false;
          break;
        }
      }
      if (selectsWholeInnerSlice)
        return true;
    }
  }
  return false;
}

static bool
needsMusaRepDisjointGenericScratchImpl(RankedTensorType srcTy,
                                       RankedTensorType dstTy,
                                       const TargetInfoBase &targetInfo) {
  if (!(isPlainBlockedLike(srcTy.getEncoding()) &&
        isPlainBlockedLike(dstTy.getEncoding())))
    return false;
  if (getBitwidth(srcTy) != 64)
    return false;
  if (!useMusaGenericBlockSwizzling(srcTy, dstTy))
    return false;
  return hasPH1PhysicalSliceRep(
      getMusaSwizzledScratchLayout(srcTy, dstTy, targetInfo));
}

static unsigned getFullLogicalScratchBytes(RankedTensorType ty) {
  auto elems = product<int64_t>(getShapePerCTA(ty));
  return elems * getBitwidth(ty) / 8;
}

static unsigned getNumScratchElemsSwizzledCvt(RankedTensorType srcTy,
                                              RankedTensorType dstTy,
                                              const TargetInfoBase &targetInfo,
                                              bool separateRepScratch) {
  auto *ctx = srcTy.getContext();
  auto srcLayout = toLinearLayout(srcTy);
  auto dstLayout = toLinearLayout(dstTy);
  srcLayout = actionRemoveBroadcastedRegs(srcLayout).apply(srcLayout);
  dstLayout = actionRemoveBroadcastedRegs(dstLayout).apply(dstLayout);
  auto bitwidth = getBitwidth(srcTy);
  auto [srcTiles, dstTiles] = getSrcDstTiles(targetInfo, bitwidth);
  auto [smem, _] =
      optimalSwizzling(srcLayout, dstLayout, srcTiles, dstTiles, bitwidth);
  if (separateRepScratch)
    return smem.getTotalOutDimSize();
  auto reps = smem.getInDimSize(StringAttr::get(ctx, "reps"));
  return smem.getTotalOutDimSize() / reps;
}

static unsigned getMusaScratchSizeInBytes(Operation *op,
                                          const TargetInfoBase &targetInfo) {
  auto cvtOp = dyn_cast<ConvertLayoutOp>(op);
  if (!cvtOp)
    return defaultAllocationAnalysisScratchSizeFn(op);

  auto srcTy = cvtOp.getSrc().getType();
  auto dstTy = cvtOp.getType();
  if (!cvtNeedsSharedMemory(srcTy, dstTy))
    return 0;

  Attribute srcLayout = srcTy.getEncoding();
  Attribute dstLayout = dstTy.getEncoding();
  if (useMusaReplicatedScratch(srcLayout, dstLayout) &&
      !isSqmmaAccumulatorToBlockedLike(srcLayout, dstLayout))
    return getFullLogicalScratchBytes(srcTy);

  if (useConservativeCarrierScratch(srcTy, dstTy))
    return getFullLogicalScratchBytes(srcTy);

  if (isSqmmaAccumulatorToBlockedLike(srcLayout, dstLayout) ||
      useMusaSqmmaBlockSwizzling(srcTy, dstTy) ||
      useMusaGenericBlockSwizzling(srcTy, dstTy)) {
    bool separateRepScratch =
        mlir::triton::musa_gpu::needsMusaRepDisjointGenericScratch(srcTy, dstTy,
                                                                   targetInfo);
    auto elems = getNumScratchElemsSwizzledCvt(srcTy, dstTy, targetInfo,
                                               separateRepScratch);
    return elems * getBitwidth(srcTy) / 8;
  }

  return defaultAllocationAnalysisScratchSizeFn(op);
}

struct AllocateMUSASharedMemory
    : public mlir::triton::impl::AllocateMUSASharedMemoryBase<
          AllocateMUSASharedMemory> {
  using AllocateMUSASharedMemoryBase::AllocateMUSASharedMemoryBase;

  AllocateMUSASharedMemory(int32_t computeCapability)
      : AllocateMUSASharedMemoryBase({computeCapability}) {}

  void runOnOperation() override {
    ModuleOp mod = getOperation();
    MUSA::TargetInfo targetInfo(computeCapability);
    ModuleAllocation allocation(
        mod, mlir::triton::musa_gpu::getMusaAllocationAnalysisScratchSizeFn(
                 targetInfo));
    mlir::triton::gpu::attachAllocationSizeAndOffsetAttr(mod, allocation);
  }
};

} // namespace

namespace mlir::triton {
namespace musa_gpu {
bool needsMusaRepDisjointGenericScratch(RankedTensorType srcTy,
                                        RankedTensorType dstTy,
                                        const TargetInfoBase &targetInfo) {
  return needsMusaRepDisjointGenericScratchImpl(srcTy, dstTy, targetInfo);
}

std::function<unsigned(Operation *)>
getMusaAllocationAnalysisScratchSizeFn(const TargetInfoBase &targetInfo) {
  return [&targetInfo](Operation *op) {
    return getMusaScratchSizeInBytes(op, targetInfo);
  };
}
} // namespace musa_gpu

std::unique_ptr<OperationPass<ModuleOp>>
createAllocateMUSASharedMemoryPass(int32_t computeCapability) {
  return std::make_unique<AllocateMUSASharedMemory>(computeCapability);
}
} // namespace mlir::triton
