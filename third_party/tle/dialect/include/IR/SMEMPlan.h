/*
 * Copyright 2025- FlagOS Contributors
 * SPDX-License-Identifier: MIT
 */
#ifndef TRITON_TLE_IR_SMEM_PLAN_H_
#define TRITON_TLE_IR_SMEM_PLAN_H_

#include "mlir/IR/Builders.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/IR/LinearLayoutConversions.h"
#include "llvm/Support/MathExtras.h"
#include <optional>

namespace mlir::triton::tle {

inline constexpr llvm::StringLiteral kSMEMPlanAttr("tle.smem_plan");
inline constexpr llvm::StringLiteral
    kWGMMAOperandBPlanAttr("tle.wgmma_operand_b_plan");
inline constexpr llvm::StringLiteral kCopyPlanAttr("tle.copy_plan");

inline bool isLegalSMEMTile(ArrayRef<int64_t> shape, ArrayRef<int64_t> tile,
                            gpu::NVMMASharedEncodingAttr encoding) {
  if (shape.size() != 2 || tile.size() != 2 || !encoding ||
      encoding.getFp4Padded() ||
      !llvm::is_contained({8u, 16u, 32u}, encoding.getElementBitWidth()) ||
      !llvm::is_contained({0u, 32u, 64u, 128u},
                          encoding.getSwizzlingByteWidth()))
    return false;
  for (unsigned d = 0; d < 2; ++d)
    if (shape[d] <= 0 || tile[d] <= 0 || !llvm::isPowerOf2_64(tile[d]) ||
        shape[d] % tile[d])
      return false;
  unsigned contiguous = encoding.getTransposed() ? 0u : 1u;
  int64_t span = std::max<unsigned>(16, encoding.getSwizzlingByteWidth()) * 8 /
                 encoding.getElementBitWidth();
  // A complete upstream core also makes each tile and stage base preserve
  // the swizzle phase. The descriptor matcher may then reason in cores.
  return tile[contiguous] >= span && tile[1 - contiguous] >= 8;
}

/// A compact array of ordinary upstream shared-memory tiles. Only the tile
/// interior is XOR-linear; the outer grid uses integer strides. Keeping that
/// distinction makes a five-panel allocation independent of its N128 carrier.
class SMEMLayoutPlan {
public:
  SMEMLayoutPlan(ArrayRef<int64_t> shape, ArrayRef<int64_t> tile,
                 gpu::NVMMASharedEncodingAttr encoding)
      : rows(shape[0]), cols(shape[1]), tileRows(tile[0]), tileCols(tile[1]),
        encoding(encoding),
        plainTileInv(gpu::nvmmaSharedToLinearLayout(tile, encoding,
                                                    /*disableSwizzle=*/true)
                         .pseudoinvert()) {}

  int64_t rows, cols, tileRows, tileCols;
  gpu::NVMMASharedEncodingAttr encoding;
  LinearLayout plainTileInv;

  int64_t elementBytes() const { return encoding.getElementBitWidth() / 8; }
  unsigned contiguousAxis() const { return encoding.getTransposed() ? 0u : 1u; }
  int64_t tileBytes() const { return tileRows * tileCols * elementBytes(); }
  int64_t stageBytes() const { return rows * cols * elementBytes(); }
  int64_t tileIndex(int64_t row, int64_t col) const {
    return !llvm::isPowerOf2_64(rows) ? col * (rows / tileRows) + row
                                      : row * (cols / tileCols) + col;
  }
  // Byte offset before swizzling, for compile-time WGMMA/TMA analysis.
  int64_t offsetBeforeSwizzle(int64_t row, int64_t col) const {
    auto dims = llvm::to_vector(plainTileInv.getInDimNames());
    auto local =
        plainTileInv
            .apply({{dims[0], row % tileRows}, {dims[1], col % tileCols}})[0]
            .second;
    return tileIndex(row / tileRows, col / tileCols) * tileBytes() +
           local * elementBytes();
  }
  DictionaryAttr getAttr() const {
    Builder b(encoding.getContext());
    return b.getDictionaryAttr(
        {b.getNamedAttr("version", b.getI32IntegerAttr(1)),
         b.getNamedAttr("shape", b.getDenseI64ArrayAttr({rows, cols})),
         b.getNamedAttr("tile", b.getDenseI64ArrayAttr({tileRows, tileCols})),
         b.getNamedAttr("encoding", encoding)});
  }
  static std::optional<SMEMLayoutPlan> fromAttr(DictionaryAttr attr) {
    if (!attr)
      return std::nullopt;
    auto version = attr.getAs<IntegerAttr>("version");
    auto shape = attr.getAs<DenseI64ArrayAttr>("shape");
    auto tile = attr.getAs<DenseI64ArrayAttr>("tile");
    auto encoding = attr.getAs<gpu::NVMMASharedEncodingAttr>("encoding");
    if (!version || version.getInt() != 1 || !shape || shape.size() != 2 ||
        !tile || tile.size() != 2 || !encoding || encoding.getFp4Padded())
      return std::nullopt;
    if (!isLegalSMEMTile(shape.asArrayRef(), tile.asArrayRef(), encoding))
      return std::nullopt;
    return SMEMLayoutPlan(shape.asArrayRef(), tile.asArrayRef(), encoding);
  }
};

struct WGMMAOperandPlan {
  SMEMLayoutPlan storage;
  bool viewTransposed;
  unsigned instructionN;
  int64_t lboBytes, sboBytes;

  int64_t logicalK() const {
    return viewTransposed ? storage.cols : storage.rows;
  }
  int64_t logicalN() const {
    return viewTransposed ? storage.rows : storage.cols;
  }
  // In B's (K, N) coordinates, axis 0 is K. This describes the hardware
  // major order, not the logical transpose requested by a consumer.
  bool isKMajor() const {
    return storage.contiguousAxis() == (viewTransposed ? 1u : 0u);
  }
  DictionaryAttr getAttr() const {
    Builder b(storage.encoding.getContext());
    return b.getDictionaryAttr(
        {b.getNamedAttr("storage", storage.getAttr()),
         b.getNamedAttr("view_transposed", b.getBoolAttr(viewTransposed)),
         b.getNamedAttr("instruction_n", b.getI32IntegerAttr(instructionN)),
         b.getNamedAttr("lbo_bytes", b.getI64IntegerAttr(lboBytes)),
         b.getNamedAttr("sbo_bytes", b.getI64IntegerAttr(sboBytes))});
  }
  static std::optional<WGMMAOperandPlan> fromAttr(DictionaryAttr attr) {
    if (!attr)
      return std::nullopt;
    auto storage =
        SMEMLayoutPlan::fromAttr(attr.getAs<DictionaryAttr>("storage"));
    auto trans = attr.getAs<BoolAttr>("view_transposed");
    auto n = attr.getAs<IntegerAttr>("instruction_n");
    auto lbo = attr.getAs<IntegerAttr>("lbo_bytes");
    auto sbo = attr.getAs<IntegerAttr>("sbo_bytes");
    if (!storage || !trans || !n || !lbo || !sbo || n.getInt() <= 0 ||
        n.getInt() > 256 || n.getInt() % 8 || lbo.getInt() < 0 ||
        sbo.getInt() < 0 || lbo.getInt() % 16 || sbo.getInt() % 16 ||
        lbo.getInt() >= (1 << 18) || sbo.getInt() >= (1 << 18))
      return std::nullopt;
    return WGMMAOperandPlan{*storage, trans.getValue(),
                            static_cast<unsigned>(n.getInt()), lbo.getInt(),
                            sbo.getInt()};
  }
};

inline bool isLegalWGMMAInstructionN(unsigned n, bool integer) {
  if (!n || n % 8 || n > (integer ? 224u : 256u))
    return false;
  return !integer || n <= 32 || n % 16 == 0;
}

/// Match the upstream core matrix against a compact integer-strided array.
/// LBO/SBO are derived, then checked over every instruction footprint,
/// including panel crossings. This is compile-time work and creates no
/// temporary IR.
inline std::optional<WGMMAOperandPlan>
planWGMMAOperand(const SMEMLayoutPlan &storage, bool viewTransposed,
                 unsigned instructionK, unsigned maxN, bool integer = false) {
  bool bIsKMajor = storage.contiguousAxis() == (viewTransposed ? 1u : 0u);
  int64_t logicalK = viewTransposed ? storage.cols : storage.rows;
  int64_t logicalN = viewTransposed ? storage.rows : storage.cols;
  if (!instructionK || !maxN || maxN % 8 || logicalK % instructionK ||
      logicalN % maxN ||
      !isLegalSMEMTile({storage.rows, storage.cols},
                       {storage.tileRows, storage.tileCols}, storage.encoding))
    return std::nullopt;
  auto coreInv =
      gpu::getCoreMatrixLinearLayout(storage.encoding, true).pseudoinvert();
  auto coreDims = llvm::to_vector(coreInv.getInDimNames());
  int64_t span = coreInv.getInDimSize(coreDims[1]);
  int64_t coreK = bIsKMajor ? span : 8;
  int64_t coreN = bIsKMajor ? 8 : span;
  auto address = [&](int64_t k, int64_t n) {
    return viewTransposed ? storage.offsetBeforeSwizzle(n, k)
                          : storage.offsetBeforeSwizzle(k, n);
  };
  // NVMMAShared uses row-major storage for swizzle=0. A wide row-major
  // tile does not necessarily contain WGMMA's interleaved core, even when
  // all core origins fit LBO/SBO. Prove the core interior as well.
  for (int64_t k = 0; k < coreK; ++k)
    for (int64_t n = 0; n < coreN; ++n) {
      int64_t coreOffset = coreInv
                               .apply({{coreDims[0], bIsKMajor ? n : k},
                                       {coreDims[1], bIsKMajor ? k : n}})[0]
                               .second;
      if (address(k, n) != coreOffset * storage.elementBytes())
        return std::nullopt;
    }
  int64_t strideK = address(coreK, 0);
  int64_t strideN = address(0, coreN);
  int64_t lbo = bIsKMajor ? strideK : strideN;
  int64_t sbo = bIsKMajor ? strideN : strideK;
  if (!storage.encoding.getSwizzlingByteWidth() && !bIsKMajor)
    std::swap(lbo, sbo);
  // Preserve the canonical K-major unused LBO used by the established N80
  // path. Nontrivial K-major leading strides remain available when needed.
  if (bIsKMajor && storage.encoding.getSwizzlingByteWidth() &&
      instructionK <= coreK)
    lbo = 16;
  if (lbo < 0 || sbo < 0 || lbo % 16 || sbo % 16 || lbo >= (1 << 18) ||
      sbo >= (1 << 18))
    return std::nullopt;

  for (unsigned n = maxN; n >= 8; n -= 8) {
    if (maxN % n || !isLegalWGMMAInstructionN(n, integer))
      continue;
    bool matches = true;
    for (int64_t k0 = 0; matches && k0 < logicalK; k0 += instructionK) {
      for (int64_t n0 = 0; matches && n0 < logicalN; n0 += n) {
        int64_t base = address(k0 - k0 % coreK, n0 - n0 % coreN);
        // Tile interiors and WGMMA use the very same upstream core layout.
        // Checking all intersected core origins proves equality throughout
        // the footprint, including instructions beginning inside a core.
        int64_t firstK = k0 / coreK, firstN = n0 / coreN;
        for (int64_t ck = firstK; matches && ck * coreK < k0 + instructionK;
             ++ck) {
          for (int64_t cn = firstN; cn * coreN < n0 + n; ++cn) {
            int64_t expected =
                base + (ck - firstK) * strideK + (cn - firstN) * strideN;
            if (address(ck * coreK, cn * coreN) != expected) {
              matches = false;
              break;
            }
          }
        }
      }
    }
    if (matches)
      return WGMMAOperandPlan{storage, viewTransposed, n, lbo, sbo};
  }
  return std::nullopt;
}

} // namespace mlir::triton::tle
#endif
