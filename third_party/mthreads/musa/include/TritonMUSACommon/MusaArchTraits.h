#ifndef TRITONMUSA_COMMON_MUSA_ARCH_TRAITS_H
#define TRITONMUSA_COMMON_MUSA_ARCH_TRAITS_H

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/StringRef.h"

#include <array>
#include <optional>
#include <utility>

namespace mlir::triton::musa {

enum class MusaArch { QY2, PH1 };

inline std::optional<MusaArch>
getMusaArchFromCapability(int computeCapability) {
  switch (computeCapability) {
  case 22:
    return MusaArch::QY2;
  case 31:
    return MusaArch::PH1;
  default:
    return std::nullopt;
  }
}

inline std::optional<MusaArch>
getMusaArchFromWmmaVersion(unsigned versionMajor, unsigned versionMinor) {
  if (versionMajor == 2 && versionMinor == 2)
    return MusaArch::QY2;
  if (versionMajor == 3 && versionMinor == 1)
    return MusaArch::PH1;
  return std::nullopt;
}

struct MusaWmmaInstrShapeGroup {
  unsigned operandBitWidth;
  llvm::ArrayRef<std::array<unsigned, 3>> shapes;
};

struct MusaWmmaArchTraits {
  MusaArch arch;
  unsigned versionMajor;
  unsigned versionMinor;
  unsigned waveSize;
  bool supportsNativeFp8Wmma;
  bool supportsContiguousRank2FastPath;
  llvm::ArrayRef<MusaWmmaInstrShapeGroup> instrShapeGroups;
};

namespace detail {

inline constexpr std::array<unsigned, 3> kWmmaShapesQY2Any[] = {{32, 32, 16}};

inline constexpr std::array<unsigned, 3> kWmmaShapesPH1x32[] = {
    {16, 8, 4}, {16, 8, 8}, {16, 16, 16}};
inline constexpr std::array<unsigned, 3> kWmmaShapesPH1x16[] = {
    {8, 16, 16}, {16, 8, 8}, {16, 8, 16}, {16, 16, 16}, {16, 16, 32}};
inline constexpr std::array<unsigned, 3> kWmmaShapesPH1x8[] = {
    {8, 16, 16}, {16, 8, 16}, {16, 16, 16}, {16, 16, 32}, {16, 16, 64}};

inline constexpr MusaWmmaInstrShapeGroup kWmmaShapeGroupsQY2[] = {
    {8, kWmmaShapesQY2Any}, {16, kWmmaShapesQY2Any}, {32, kWmmaShapesQY2Any}};
inline constexpr MusaWmmaInstrShapeGroup kWmmaShapeGroupsPH1[] = {
    {32, kWmmaShapesPH1x32}, {16, kWmmaShapesPH1x16}, {8, kWmmaShapesPH1x8}};

inline const MusaWmmaArchTraits kMusaWmmaArchTraits[] = {
    {MusaArch::QY2, 2, 2, 128, false, false, kWmmaShapeGroupsQY2},
    {MusaArch::PH1, 3, 1, 32, true, true, kWmmaShapeGroupsPH1},
};

} // namespace detail

inline const MusaWmmaArchTraits &getMusaWmmaArchTraits(MusaArch arch) {
  return detail::kMusaWmmaArchTraits[static_cast<unsigned>(arch)];
}

inline llvm::ArrayRef<std::array<unsigned, 3>>
lookupWmmaCandidateInstrShapes(const MusaWmmaArchTraits &traits,
                               unsigned operandBitWidth) {
  for (const auto &group : traits.instrShapeGroups)
    if (group.operandBitWidth == operandBitWidth)
      return group.shapes;
  return {};
}

inline std::optional<MusaArch>
getMusaArchFromSqmmaVersion(unsigned versionMajor, unsigned versionMinor) {
  if (versionMajor == 3 && versionMinor == 1)
    return MusaArch::PH1;
  return std::nullopt;
}

struct MusaSqmmaCandidateKs {
  unsigned operandBitWidth;
  llvm::ArrayRef<unsigned> kValues;
};

struct MusaSqmmaArchTraits {
  MusaArch arch;
  unsigned versionMajor;
  unsigned versionMinor;
  unsigned instMAlignment;
  llvm::ArrayRef<unsigned> candidateM;
  llvm::ArrayRef<unsigned> candidateN;
  llvm::ArrayRef<MusaSqmmaCandidateKs> candidateKs;
  llvm::ArrayRef<std::pair<unsigned, unsigned>> tf32InstrMN;
};

namespace detail {

inline constexpr unsigned kSqmmaCandidateMNPH1[] = {128, 64, 32, 16};

inline constexpr unsigned kSqmmaKsPH1x16[] = {64, 32, 16};
inline constexpr unsigned kSqmmaKsPH1x32[] = {32, 16, 8};
inline constexpr unsigned kSqmmaKsPH1x8[] = {128, 64, 32};

inline constexpr MusaSqmmaCandidateKs kSqmmaCandidateKsPH1[] = {
    {16, kSqmmaKsPH1x16}, {32, kSqmmaKsPH1x32}, {8, kSqmmaKsPH1x8}};

inline constexpr std::pair<unsigned, unsigned> kSqmmaInstrMN[] = {
    {32, 32}, {32, 64},  {32, 128}, {16, 64},  {64, 16},   {64, 32},
    {64, 64}, {64, 128}, {128, 32}, {128, 64}, {128, 128},
};
inline constexpr std::pair<unsigned, unsigned> kSqmmaTf32InstrMNPH1[] = {
    {16, 64}, {32, 32}, {32, 64},  {64, 16},
    {64, 32}, {64, 64}, {128, 64}, {128, 128},
};

inline const MusaSqmmaArchTraits kMusaSqmmaArchTraitsPH1 = {
    MusaArch::PH1,
    3,
    1,
    4,
    kSqmmaCandidateMNPH1,
    kSqmmaCandidateMNPH1,
    kSqmmaCandidateKsPH1,
    kSqmmaTf32InstrMNPH1};

} // namespace detail

inline const MusaSqmmaArchTraits *getMusaSqmmaArchTraits(MusaArch arch) {
  switch (arch) {
  case MusaArch::PH1:
    return &detail::kMusaSqmmaArchTraitsPH1;
  default:
    return nullptr;
  }
}

inline llvm::ArrayRef<unsigned>
lookupSqmmaCandidateKs(const MusaSqmmaArchTraits &traits,
                       unsigned operandBitWidth) {
  for (const auto &group : traits.candidateKs)
    if (group.operandBitWidth == operandBitWidth)
      return group.kValues;
  return {};
}

enum class MusaTMELoadOperandLayout {
  Persistence,
};

struct MusaTMEArchTraits {
  MusaArch arch;
  llvm::ArrayRef<llvm::StringRef> loadIntrinsicNames;
  MusaTMELoadOperandLayout loadOperandLayout;
};

namespace detail {

inline constexpr llvm::StringRef kTMELoadIntrinsicNamesPH1[] = {
    "llvm.musa.tme.ld.tile.1d", "llvm.musa.tme.ld.tile.2d",
    "llvm.musa.tme.ld.tile.3d", "llvm.musa.tme.ld.tile.4d",
    "llvm.musa.tme.ld.tile.5d"};

inline const MusaTMEArchTraits kMusaTMEArchTraitsPH1 = {
    MusaArch::PH1, kTMELoadIntrinsicNamesPH1,
    MusaTMELoadOperandLayout::Persistence};

} // namespace detail

inline const MusaTMEArchTraits *getMusaTMEArchTraits(MusaArch arch) {
  switch (arch) {
  case MusaArch::PH1:
    return &detail::kMusaTMEArchTraitsPH1;
  default:
    return nullptr;
  }
}

inline bool supportsMusaTME(int computeCapability) {
  auto arch = getMusaArchFromCapability(computeCapability);
  return arch && getMusaTMEArchTraits(*arch) != nullptr;
}

} // namespace mlir::triton::musa

#endif // TRITONMUSA_COMMON_MUSA_ARCH_TRAITS_H
