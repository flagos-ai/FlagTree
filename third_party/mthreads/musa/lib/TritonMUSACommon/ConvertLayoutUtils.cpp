#include "TritonMUSACommon/ConvertLayoutUtils.h"

#include "triton/Analysis/Utility.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "llvm/ADT/STLExtras.h"

namespace mlir::triton::musa {
namespace ttg = mlir::triton::gpu;

static bool isSingleCta(ttg::CGAEncodingAttr cgaLayout) {
  return llvm::all_of(cgaLayout.getCTAsPerCGA(),
                      [](unsigned x) { return x == 1; }) &&
         llvm::all_of(cgaLayout.getCTASplitNum(),
                      [](unsigned x) { return x == 1; });
}

static bool isNonSingleCtaMusaSqmmaDotOperand(Attribute layout) {
  auto dotOperand = dyn_cast<ttg::DotOperandEncodingAttr>(layout);
  if (!dotOperand || !isa<ttg::MUSASqmmaEncodingAttr>(dotOperand.getParent()))
    return false;
  return !isSingleCta(dotOperand.getCGALayout());
}

bool isMusaSqmmaLike(Attribute layout) {
  return isa<ttg::MUSASqmmaEncodingAttr>(layout);
}

bool isPlainBlockedLike(Attribute layout) {
  return isa<ttg::BlockedEncodingAttr, ttg::SliceEncodingAttr>(layout);
}

bool useMusaReplicatedScratch(Attribute srcLayout, Attribute dstLayout) {
  return (isMusaSqmmaLike(srcLayout) || isMusaSqmmaLike(dstLayout)) &&
         isa<ttg::MmaEncodingTrait, ttg::BlockedEncodingAttr,
             ttg::SliceEncodingAttr>(srcLayout) &&
         isa<ttg::MmaEncodingTrait, ttg::BlockedEncodingAttr,
             ttg::SliceEncodingAttr>(dstLayout);
}

bool isMusaSqmmaAccumulatorToBlockedLike(Attribute srcLayout,
                                         Attribute dstLayout) {
  return isa<ttg::MUSASqmmaEncodingAttr>(srcLayout) &&
         isa<ttg::BlockedEncodingAttr, ttg::SliceEncodingAttr>(dstLayout);
}

bool useMusaSqmmaBlockSwizzling(RankedTensorType srcTy,
                                RankedTensorType dstTy) {
  if (!(isMusaSqmmaLike(srcTy.getEncoding()) ||
        isMusaSqmmaLike(dstTy.getEncoding())))
    return false;

  LinearLayout conversion = minimalCvtLayout(srcTy, dstTy);
  StringAttr kBlock = StringAttr::get(srcTy.getContext(), "block");
  StringAttr kWarp = StringAttr::get(srcTy.getContext(), "warp");
  StringAttr kLane = StringAttr::get(srcTy.getContext(), "lane");
  auto dims = conversion.getInDimNames();

  if (llvm::is_contained(dims, kBlock))
    return false;
  if (llvm::is_contained(dims, kWarp))
    return true;
  if (llvm::is_contained(dims, kLane))
    return !cvtNeedsWarpShuffle(srcTy, dstTy);
  return false;
}

bool useConservativeCarrierScratch(RankedTensorType srcTy,
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

  if (!isa<ttg::BlockedEncodingAttr, ttg::SliceEncodingAttr>(
          srcTy.getEncoding()) ||
      !isa<ttg::BlockedEncodingAttr, ttg::SliceEncodingAttr>(
          dstTy.getEncoding()))
    return false;

  LinearLayout conversion = minimalCvtLayout(srcTy, dstTy);
  StringAttr kBlock = StringAttr::get(srcTy.getContext(), "block");
  return !llvm::is_contained(conversion.getInDimNames(), kBlock);
}

bool useMusaGenericBlockSwizzling(RankedTensorType srcTy,
                                  RankedTensorType dstTy) {
  if (isMusaSqmmaLike(srcTy.getEncoding()) ||
      isMusaSqmmaLike(dstTy.getEncoding()))
    return false;
  if (isNonSingleCtaMusaSqmmaDotOperand(srcTy.getEncoding()) ||
      isNonSingleCtaMusaSqmmaDotOperand(dstTy.getEncoding()))
    return false;
  if (!cvtNeedsSharedMemory(srcTy, dstTy))
    return false;

  LinearLayout conversion = minimalCvtLayout(srcTy, dstTy);
  StringAttr kBlock = StringAttr::get(srcTy.getContext(), "block");
  return !llvm::is_contained(conversion.getInDimNames(), kBlock);
}

bool musaConvertLayoutHasLoweringPath(RankedTensorType srcTy,
                                      RankedTensorType dstTy) {
  return useMusaReplicatedScratch(srcTy.getEncoding(), dstTy.getEncoding()) ||
         useConservativeCarrierScratch(srcTy, dstTy) ||
         useMusaSqmmaBlockSwizzling(srcTy, dstTy) ||
         useMusaGenericBlockSwizzling(srcTy, dstTy);
}

} // namespace mlir::triton::musa
