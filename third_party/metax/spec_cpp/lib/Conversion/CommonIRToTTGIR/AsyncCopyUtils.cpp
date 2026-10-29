#include "triton/Conversion/CommonIRToTTGIR/AsyncCopyUtils.h"
#include "triton/Dialect/Triton/IR/Utility.h"
#include "triton/Tools/LayoutUtils.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/MathExtras.h"

namespace mlir::triton::metax {

bool isAsyncCopyDestinationSupported(gpu::MemDescType destination) {
  auto shared =
      dyn_cast<gpu::SwizzledSharedEncodingAttr>(destination.getEncoding());
  // Native LLVM lowering drops allocShape when removing the swizzle.
  return shared && (shared.getMaxPhase() == 1 ||
                    destination.getAllocShape() == destination.getShape());
}

gpu::MemDescType getAsyncCopyLinearDestination(gpu::MemDescType destination) {
  auto shared = cast<gpu::SwizzledSharedEncodingAttr>(destination.getEncoding());
  auto linear = gpu::SwizzledSharedEncodingAttr::get(
      destination.getContext(), shared.getVec(), 1, 1, shared.getOrder(),
      shared.getCTALayout());
  return gpu::MemDescType::get(
      destination.getShape(), destination.getElementType(), linear,
      destination.getMemorySpace(), destination.getMutableMemory(),
      destination.getAllocShape());
}

unsigned getAsyncCopyVectorSize(RankedTensorType source,
                               gpu::MemDescType destination,
                               const AxisInfo *sourceInfo) {
  auto layout = gpu::toLinearEncoding(source);
  unsigned dim = layout.getOrder()[0];
  unsigned bits = destination.getElementTypeBitWidth();
  unsigned contiguity = 1;
  if (sourceInfo && sourceInfo->getRank() == source.getRank()) {
    unsigned alignment = std::max<int64_t>(
        sourceInfo->getDivisibility(dim) / std::max(bits / 8, 1u), 1);
    contiguity = std::min<unsigned>(alignment, sourceInfo->getContiguity(dim));
  }
  unsigned vec =
      std::min({contiguity, layout.getContigPerThread()[dim], 128 / bits});
  auto shared = cast<gpu::SwizzledSharedEncodingAttr>(destination.getEncoding());
  if (shared.getMaxPhase() > 1)
    vec = std::min(vec, shared.getVec());
  return vec;
}

gpu::BlockedEncodingAttr chooseAsyncCopyLayout(
    RankedTensorType source, gpu::MemDescType destination,
    const AxisInfo *sourceInfo, unsigned minVec, unsigned maxVec) {
  auto blocked = dyn_cast<gpu::BlockedEncodingAttr>(source.getEncoding());
  auto shared =
      dyn_cast<gpu::SwizzledSharedEncodingAttr>(destination.getEncoding());
  if (!blocked || !shared || !isAsyncCopyDestinationSupported(destination) ||
      source.getShape() != destination.getShape())
    return {};
  auto compatible = [&](RankedTensorType candidate) {
    unsigned vec = getAsyncCopyVectorSize(candidate, destination, sourceInfo);
    return vec >= minVec && vec <= maxVec &&
           isAsyncCopyLayoutCompatible(candidate, destination, vec);
  };
  if (compatible(source))
    return blocked;

  for (unsigned width = 128 / destination.getElementTypeBitWidth(); width;
       width /= 2) {
    SmallVector<unsigned> sizePerThread(source.getRank(), 1);
    sizePerThread[shared.getOrder()[0]] = width;
    auto encoding = gpu::BlockedEncodingAttr::get(
        source.getContext(), source.getShape(), sizePerThread, shared.getOrder(),
        product(blocked.getWarpsPerCTA()), product(blocked.getThreadsPerWarp()),
        blocked.getCTALayout());
    auto candidate = source.cloneWithEncoding(encoding);
    if (compatible(candidate))
      return encoding;
  }
  return {};
}

bool isAsyncCopyLayoutCompatible(RankedTensorType source,
                                gpu::MemDescType destination, unsigned vec) {
  if (!isa<gpu::BlockedEncodingAttr>(source.getEncoding()) ||
      !isAsyncCopyDestinationSupported(destination) ||
      source.getShape() != destination.getShape() || !llvm::isPowerOf2_32(vec))
    return false;

  auto *ctx = source.getContext();
  auto reg = StringAttr::get(ctx, "register");
  auto lane = StringAttr::get(ctx, "lane");
  auto block = StringAttr::get(ctx, "block");
  auto offset = StringAttr::get(ctx, "offset");
  auto srcLayout = gpu::toLinearLayout(source);
  auto dstLayout = gpu::toLinearLayout(getAsyncCopyLinearDestination(destination));
  auto cvt = srcLayout.invertAndCompose(dstLayout);
  if (!cvt.isTrivialOver({block}))
    return false;

  // The first vec registers must be consecutive in shared, in the same order
  // as the vector global read. Remaining register bits select other copies.
  unsigned regBits = llvm::Log2_32(vec);
  if (cvt.getInDimSizeLog2(reg) < regBits)
    return false;
  for (unsigned bit = 0; bit < regBits; ++bit)
    if (cvt.getBasis(reg, bit, offset) != (1u << bit))
      return false;

  for (unsigned bit = 0; bit < cvt.getInDimSizeLog2(lane); ++bit) {
    // Broadcast lanes are suppressed by the lowering's thread predicate.
    if (llvm::all_of(srcLayout.getBasis(lane, bit),
                     [](int32_t basis) { return basis == 0; }))
      continue;
    if (cvt.getBasis(lane, bit, offset) != (vec << bit))
      return false;
  }
  return true;
}

} // namespace mlir::triton::metax
