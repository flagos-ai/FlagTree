#ifndef TRITON_HCU_MATRIX_COPY_GEOMETRY_H
#define TRITON_HCU_MATRIX_COPY_GEOMETRY_H

#include "triton/Tools/LinearLayout.h"
#include "llvm/Support/MathExtras.h"
#include <cstdint>
#include <limits>
#include <optional>

namespace mlir::triton::HCU {
struct MatrixCopyGeometry {
  unsigned row, column;
  int64_t rows, columns, pitch;
  uint64_t elements, tileBytes;
};

inline std::optional<MatrixCopyGeometry>
getMatrixCopyGeometry(ArrayRef<int64_t> extents, ArrayRef<int64_t> strides,
                      ArrayRef<int64_t> tile, unsigned waves) {
  if (extents.size() != 2 || strides.size() != 2 || tile.size() != 2 ||
      !waves || waves > 8 || !llvm::isPowerOf2_32(waves))
    return std::nullopt;
  unsigned column = strides[1] == 1 ? 1 : 0;
  unsigned row = 1 - column;
  int64_t rows = extents[row], columns = extents[column], pitch = strides[row];
  constexpr uint64_t maxElements = uint64_t{1} << 20;
  if (strides[column] != 1 || rows <= 0 || columns <= 0 ||
      pitch < columns || pitch > std::numeric_limits<int32_t>::max() ||
      pitch % 8 || columns % 8 || tile[0] <= 0 || tile[1] <= 0 ||
      uint64_t(tile[0]) > maxElements / uint64_t(tile[1]))
    return std::nullopt;
  uint64_t elements = uint64_t(tile[0]) * tile[1];
  if (elements < 512 * waves || elements % (512 * waves))
    return std::nullopt;
  uint64_t tileBytes =
      2 * (uint64_t(tile[row] - 1) * pitch + tile[column]);
  return MatrixCopyGeometry{row, column, rows, columns, pitch, elements,
                            tileBytes};
}

inline bool hasAddressableMatrixCopyTile(const MatrixCopyGeometry &geometry) {
  return geometry.pitch < 0x2000 || geometry.tileBytes < (uint64_t{1} << 31);
}

// A LinearLayout is XOR-linear. A group of eight element offsets is a
// consecutive aligned vector exactly when the low three bits enumerate the
// contiguous axis and the remaining bases contribute no low column bits.
inline bool isMatrixCopyVectorLayout(const LinearLayout &layout,
                                     StringAttr offset, unsigned row,
                                     unsigned column) {
  if (!layout.hasInDim(offset) || layout.getInDimSizeLog2(offset) < 3)
    return false;
  for (unsigned bit = 0, end = layout.getInDimSizeLog2(offset); bit < end; ++bit) {
    auto basis = layout.getBasis(offset, bit);
    if (bit < 3) {
      if (basis[row] != 0 || basis[column] != (1 << bit))
        return false;
    } else if (basis[column] & 7) {
      return false;
    }
  }
  return true;
}
} // namespace mlir::triton::HCU

#endif
