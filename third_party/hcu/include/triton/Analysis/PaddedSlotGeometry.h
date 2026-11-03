#ifndef TRITON_ANALYSIS_PADDED_SLOT_GEOMETRY_H
#define TRITON_ANALYSIS_PADDED_SLOT_GEOMETRY_H

#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "triton/Tools/LayoutUtils.h"
#include "llvm/Support/MathExtras.h"
#include "llvm/ADT/STLExtras.h"
#include <cstdint>
#include <limits>
#include <optional>

namespace mlir::triton::gpu {

// Exact bounded geometry for ONE full MemDescIndex plane. This is not an
// allocation/slots approximation. Matches MemDescIndexOpConversion and
// emitPadding, including the absence of padding after the final element.
struct PaddedSlotGeometry {
  uint64_t slots, strideBytes, footprintBytes, rootBytes;
};

inline bool paddedSlotAdd(uint64_t a, uint64_t b, uint64_t &out) {
  if (a > std::numeric_limits<uint64_t>::max() - b)
    return false;
  out = a + b;
  return true;
}
inline bool paddedSlotMul(uint64_t a, uint64_t b, uint64_t &out) {
  if (b && a > std::numeric_limits<uint64_t>::max() / b)
    return false;
  out = a * b;
  return true;
}

inline std::optional<PaddedSlotGeometry>
getPaddedSlotGeometry(MemDescType rootTy, MemDescType planeTy) {
  auto enc = dyn_cast<PaddedSharedEncodingAttr>(planeTy.getEncoding());
  if (!enc || rootTy.getEncoding() != planeTy.getEncoding() ||
      planeTy.getRank() < 1 || rootTy.getRank() != planeTy.getRank() + 1 ||
      rootTy.getShape() != rootTy.getAllocShape() ||
      planeTy.getShape() != planeTy.getAllocShape() ||
      rootTy.getShape().drop_front() != planeTy.getShape() ||
      rootTy.getElementType() != planeTy.getElementType() ||
      rootTy.getMemorySpace() != planeTy.getMemorySpace() ||
      rootTy.getMutableMemory() != planeTy.getMutableMemory() ||
      ArrayRef<int64_t>(getAllocationShapePerCTA(rootTy)) != rootTy.getShape() ||
      ArrayRef<int64_t>(getAllocationShapePerCTA(planeTy)) != planeTy.getShape())
    return std::nullopt;
  Type elem = planeTy.getElementType();
  if (!elem.isIntOrFloat())
    return std::nullopt;
  unsigned bits = elem.getIntOrFloatBitWidth();
  if (!bits || bits % 8)
    return std::nullopt; // No fp4 packing or reinterpretation proof.
  uint64_t bytes = bits / 8, elements = 1;
  for (int64_t dim : planeTy.getShape()) {
    if (dim <= 0 || !paddedSlotMul(elements, uint64_t(dim), elements))
      return std::nullopt;
  }
  if (rootTy.getShape()[0] <= 0)
    return std::nullopt;
  uint64_t slots = rootTy.getShape()[0], rootElements;
  if (!paddedSlotMul(slots, elements, rootElements) ||
      rootElements > uint64_t(std::numeric_limits<int32_t>::max()))
    return std::nullopt; // Lowering uses signed i32 shifts/multiply.

  // The unpadded plane must be the complete [0,elements) offset domain,
  // bijectively permuted into its logical shape, with no block replication.
  const auto &ll = enc.getLinearComponent();
  auto offset = StringAttr::get(planeTy.getContext(), "offset");
  auto block = StringAttr::get(planeTy.getContext(), "block");
  bool hasOffset = false, hasBlock = false;
  for (StringAttr name : ll.getInDimNames()) {
    if (name == offset)
      hasOffset = true;
    else if (name == block)
      hasBlock = true;
    else
      return std::nullopt;
  }
  if (!hasOffset || !hasBlock || !ll.isInvertible() ||
      ll.getNumOutDims() != planeTy.getRank() ||
      uint64_t(ll.getInDimSize(offset)) != elements ||
      ll.getInDimSize(block) != 1)
    return std::nullopt;
  unsigned dim = 0;
  for (int32_t size : ll.getOutDimSizes())
    if (int64_t(size) != planeTy.getShape()[dim++])
      return std::nullopt;

  if (enc.getIntervals().size() != enc.getPaddings().size())
    return std::nullopt;
  // I|elements proves pad(s*elements+x)=pad(s*elements)+pad(x).
  // Otherwise a shifted plane needs a different local padding map; reject.
  for (auto [interval, padding] :
       llvm::zip_equal(enc.getIntervals(), enc.getPaddings()))
    if (!llvm::isPowerOf2_32(interval) ||
        !llvm::isPowerOf2_32(padding) || elements % interval)
      return std::nullopt;
  auto physical = [&](uint64_t n) -> std::optional<uint64_t> {
    uint64_t result = n;
    for (auto [interval, padding] :
         llvm::zip_equal(enc.getIntervals(), enc.getPaddings())) {
      uint64_t extra;
      if (!paddedSlotMul(n / interval, padding, extra) ||
          !paddedSlotAdd(result, extra, result))
        return std::nullopt;
    }
    return result;
  };
  auto stride = physical(elements);
  auto last = physical(elements - 1);
  auto rootLast = physical(rootElements - 1);
  uint64_t footprint, rootSize, strideBytes, footprintBytes, rootBytes;
  if (!stride || !last || !rootLast ||
      !paddedSlotAdd(*last, 1, footprint) ||
      !paddedSlotAdd(*rootLast, 1, rootSize) || footprint > *stride ||
      !paddedSlotMul(*stride, bytes, strideBytes) ||
      !paddedSlotMul(footprint, bytes, footprintBytes) ||
      !paddedSlotMul(rootSize, bytes, rootBytes) ||
      rootBytes > uint64_t(std::numeric_limits<int32_t>::max()))
    return std::nullopt;
  return PaddedSlotGeometry{slots, strideBytes, footprintBytes, rootBytes};
}

// MemDescTrans lowering keeps the base pointer, only permuting offsets. Accept
// exactly a verified whole-plane permutation with the matching linear map.
inline bool isFullPaddedSlotTranspose(MemDescTransOp trans) {
  auto src = trans.getSrc().getType(), dst = trans.getType();
  auto a = dyn_cast<PaddedSharedEncodingAttr>(src.getEncoding());
  auto b = dyn_cast<PaddedSharedEncodingAttr>(dst.getEncoding());
  auto order = trans.getOrder();
  if (!a || !b || src.getRank() != dst.getRank() ||
      order.size() != size_t(src.getRank()) ||
      src.getShape() != src.getAllocShape() ||
      dst.getShape() != dst.getAllocShape() ||
      src.getElementType() != dst.getElementType() ||
      src.getMemorySpace() != dst.getMemorySpace() ||
      src.getMutableMemory() != dst.getMutableMemory() ||
      a.getIntervals() != b.getIntervals() || a.getPaddings() != b.getPaddings())
    return false;
  SmallVector<bool> seen(src.getRank(), false);
  SmallVector<int> permutation;
  for (auto [i, axis] : llvm::enumerate(order)) {
    if (axis < 0 || axis >= src.getRank() || seen[axis] ||
        dst.getShape()[i] != src.getShape()[axis])
      return false;
    seen[axis] = true;
    permutation.push_back(axis);
  }
  return transposeLinearLayout(a.getLinearComponent(), permutation) ==
         b.getLinearComponent();
}

} // namespace mlir::triton::gpu
#endif
