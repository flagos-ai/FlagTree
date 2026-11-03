#ifndef TRITON_HCU_DIALECT_IR_WHOLE_PACKET_LAYOUT_H_
#define TRITON_HCU_DIALECT_IR_WHOLE_PACKET_LAYOUT_H_

#include <array>
#include <cstdint>
#include <initializer_list>
#include <optional>

namespace mlir::triton::hcugpu::wholepacket {

// This is an arithmetic hardware packet map, NOT a LinearLayout. Only the
// logical source chart below is linear over GF(2). Constants describe the
// measured gfx936 16-bit/16-byte packet ABI, not an MM shape selector.
enum class Role { StructuredRows, RawBytes };
constexpr unsigned Elements = 4096, Bytes = Elements * 2;
constexpr unsigned Lanes = 64, VectorElements = 8, PacketWaves = 8;
constexpr unsigned FrameBytes = 2 * Bytes;


// v258 separates the reserved A/B half from DMA addressing and storage axes.
// Alt2Kind is a versioned two-bit payload here, not a legacy ALT instruction:
// bit0 = structured rows (otherwise raw bytes), bit1 = swapped storage axes.
// Old v256/v257 semantics and mangles remain unchanged.
struct PacketSemantics {
  bool prefix;
  Role transfer;
  bool transpose;

  static std::optional<PacketSemantics> get(unsigned version, unsigned half,
                                            unsigned payload) {
    if (half > 1) return std::nullopt;
    if (version == 256 || version == 257) {
      if (payload) return std::nullopt;
      return PacketSemantics{version == 257,
          half == 0 ? Role::StructuredRows : Role::RawBytes, false};
    }
    if (version != 258 || payload > 3) return std::nullopt;
    return PacketSemantics{true,
        (payload & 1) ? Role::StructuredRows : Role::RawBytes,
        bool(payload & 2)};
  }
  unsigned storageRows(unsigned rows, unsigned columns) const {
    return transpose ? columns : rows;
  }
  unsigned storageColumns(unsigned rows, unsigned columns) const {
    return transpose ? rows : columns;
  }
};

inline unsigned physicalHalf(Role role, unsigned ordinal) {
  unsigned frame = ordinal / Elements;
  unsigned wave = (ordinal / 512) % PacketWaves;
  unsigned within = ordinal % 512;
  unsigned rotation = role == Role::StructuredRows ? 8 * (wave & 3)
                                                  : 32 * (wave & 1);
  return frame * Elements + wave * 512 + ((within + rotation) & 511);
}

inline unsigned sourceOrdinal(Role role, unsigned physical) {
  unsigned frame = physical / Elements;
  unsigned wave = (physical / 512) % PacketWaves;
  unsigned rotation = role == Role::StructuredRows ? 8 * (wave & 3)
                                                  : 32 * (wave & 1);
  return frame * Elements + wave * 512 + ((physical - rotation) & 511);
}

// Complete finite chart: ordinal = ((copy * warps + wave) * 64 + lane) * 8
// + element; logical = row * columns + column. No shape/name/gfx gate here.
// Unknown, non-bijective, sub-packet and non-vector-contiguous charts fail.
struct SourceLayout {
  unsigned rows = 0, columns = 0, warps = 0;
  std::array<unsigned, 12> basis{};
  std::array<unsigned, 12> inverseBasis{};

  static unsigned apply(const std::array<unsigned, 12> &b, unsigned value) {
    unsigned result = 0;
    for (unsigned i = 0; i < b.size(); ++i)
      if (value & (1u << i))
        result ^= b[i];
    return result;
  }
  unsigned logical(unsigned ordinal) const { return apply(basis, ordinal); }
  unsigned ordinal(unsigned coordinate) const {
    return apply(inverseBasis, coordinate);
  }
  unsigned physical(Role role, unsigned coordinate) const {
    return physicalHalf(role, ordinal(coordinate));
  }

  static std::optional<SourceLayout>
  get(unsigned rows, unsigned columns, unsigned warps,
      const std::array<unsigned, 12> &basis) {
    auto powerOfTwo = [](unsigned x) { return x && !(x & (x - 1)); };
    if (!powerOfTwo(rows) || !powerOfTwo(columns) ||
        uint64_t(rows) * columns != Elements || !powerOfTwo(warps) ||
        warps > PacketWaves)
      return std::nullopt;
    SourceLayout result{rows, columns, warps, basis, {}};
    std::array<unsigned, Elements> inverse{};
    inverse.fill(Elements);
    for (unsigned u = 0; u < Elements; ++u) {
      unsigned c = result.logical(u);
      if (c >= Elements || inverse[c] != Elements)
        return std::nullopt;
      inverse[c] = u;
      unsigned base = result.logical(u & ~7u);
      if (base % columns % VectorElements ||
          c / columns != base / columns || c != base + (u & 7))
        return std::nullopt;
    }
    for (unsigned i = 0; i < 12; ++i)
      result.inverseBasis[i] = inverse[1u << i];
    for (unsigned c = 0; c < Elements; ++c)
      if (result.ordinal(c) != inverse[c])
        return std::nullopt;
    return result;
  }

  // A complete active prefix of the same physical packet. The reserved role
  // region stays Elements words; every CTA wave executes full 16-byte vectors.
  // This separate entry leaves the original full-packet contract unchanged.
  static std::optional<SourceLayout>
  getActivePrefix(unsigned rows, unsigned columns, unsigned warps,
                  const std::array<unsigned, 12> &basis) {
    auto powerOfTwo = [](unsigned x) { return x && !(x & (x - 1)); };
    uint64_t count = uint64_t(rows) * columns;
    if (!powerOfTwo(rows) || !powerOfTwo(columns) || !powerOfTwo(warps) ||
        warps > PacketWaves || count > Elements ||
        count < uint64_t(warps) * Lanes * VectorElements ||
        count % (warps * Lanes * VectorElements))
      return std::nullopt;
    if (count == Elements)
      return get(rows, columns, warps, basis);
    unsigned active = unsigned(count), bits = 0;
    for (unsigned n = active; n > 1; n >>= 1) ++bits;
    for (unsigned i = bits; i < basis.size(); ++i)
      if (basis[i]) return std::nullopt;
    SourceLayout result{rows, columns, warps, basis, {}};
    std::array<unsigned, Elements> inverse{};
    inverse.fill(Elements);
    for (unsigned u = 0; u < active; ++u) {
      unsigned c = result.logical(u);
      if (c >= active || inverse[c] != Elements) return std::nullopt;
      inverse[c] = u;
      unsigned base = result.logical(u & ~7u);
      if (base % columns % VectorElements || c / columns != base / columns ||
          c != base + (u & 7)) return std::nullopt;
      for (Role role : {Role::StructuredRows, Role::RawBytes}) {
        unsigned physical = physicalHalf(role, u);
        if (physical >= active || sourceOrdinal(role, physical) != u)
          return std::nullopt;
      }
    }
    for (unsigned i = 0; i < bits; ++i)
      result.inverseBasis[i] = inverse[1u << i];
    for (unsigned c = 0; c < active; ++c)
      if (result.ordinal(c) != inverse[c]) return std::nullopt;
    return result;
  }

  unsigned activeElements() const { return rows * columns; }
  unsigned copiesPerWave() const {
    return warps ? activeElements() / (warps * Lanes * VectorElements) : 0;
  }
};

// One allocation owns paired B/A packets. All views are complete packets;
// arbitrary subviews are deliberately unrepresentable here. LDS alias/size
// analysis can use these byte intervals without linearizing hardware F.
struct WholeFrame {
  unsigned slots;
  static std::optional<WholeFrame> get(unsigned slots, unsigned capacityBytes) {
    if (!slots || uint64_t(slots) * FrameBytes > capacityBytes ||
        uint64_t(slots) * FrameBytes > 65536)
      return std::nullopt;
    return WholeFrame{slots};
  }
  std::optional<unsigned> origin(Role role, unsigned slot) const {
    if (slot >= slots)
      return std::nullopt;
    return slot * FrameBytes + (role == Role::StructuredRows ? Bytes : 0);
  }
  std::optional<unsigned> packedM0(Role role, unsigned slot,
                                   unsigned packetWave) const {
    auto base = origin(role, slot);
    if (!base || packetWave >= PacketWaves)
      return std::nullopt;
    unsigned low = *base + packetWave * 1024;
    unsigned high = role == Role::StructuredRows ? (packetWave & 3) << 16
                                               : (packetWave & 1) << 18;
    return low | high;
  }
};

// Explicit units avoid applying byte NUM_RECORDS arithmetic to structured
// row records. Dynamic proof must be supplied by the caller before selection.
struct StructuredDomain {
  unsigned strideElements, remainingRows;
  static std::optional<StructuredDomain> get(unsigned strideElements,
                                             unsigned remainingRows) {
    if (strideElements < 256 || strideElements > 4096 ||
        (strideElements & (strideElements - 1)))
      return std::nullopt;
    return StructuredDomain{strideElements, remainingRows};
  }
  uint16_t mode() const { return 0x4000 | (2 * strideElements); }
};
struct RawDomain {
  uint32_t remainingBytes;
  static std::optional<RawDomain> get(uint64_t remainingBytes) {
    if (remainingBytes >= (uint64_t(1) << 31))
      return std::nullopt;
    return RawDomain{uint32_t(remainingBytes)};
  }
};

} // namespace mlir::triton::hcugpu::wholepacket
#endif
