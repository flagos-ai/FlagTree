#ifndef TRITON_THIRD_PARTY_HCU_INCLUDE_TRITONHCUGPUTOLLVM_TARGETUTILS_H_
#define TRITON_THIRD_PARTY_HCU_INCLUDE_TRITONHCUGPUTOLLVM_TARGETUTILS_H_

#include "llvm/ADT/StringRef.h"

namespace mlir::triton::HCU {

// A list of ISA families we care about.
enum class ISAFamily {
  Unknown,
  CDNA1,
  CDNA2,
  CDNA3,
  CDNA4,
  RDNA1,
  RDNA2,
  RDNA3,
  RDNA4,
  GFX1250,
};

// Deduces the corresponding ISA family for the given target gfx |arch|.
ISAFamily deduceISAFamily(llvm::StringRef arch);

// Retursn true if given architecture support V_DOT instruction.
bool supportsVDot(llvm::StringRef arch);

bool isCDNA(ISAFamily isaFamily);

bool isRDNA(ISAFamily isaFamily);

// Here is a partial definition of DppCtrl enums. For the complete definition,
// please check:
// https://github.com/llvm/llvm-project/blob/8c75290/llvm/lib/Target/HCUGPU/SIDefines.h#L939
enum class DppCtrl : uint32_t {
  QUAD_PERM_FIRST = 0,
  ROW_SHL0 = 0x100,
  ROW_SHR0 = 0x110,
  BCAST15 = 0x142,
  BCAST31 = 0x143
};

// HCU ISA features
enum class HCUISAFeature : uint64_t{
  NONE          = 0,
  MMAC_LAYOUT   = 1 << 0,
  MAMC_FP8      = 1 << 1,
  MMAC_FP6FP4   = 1 << 2,
  MMAC_SCALE    = 1 << 3,
  MLS           = 1 << 4,
  CVT_FP8F32    = 1 << 5,
  CVT_FP8F16    = 1 << 6,
};

inline constexpr bool operator&(HCUISAFeature lhs, HCUISAFeature rhs) {
  return static_cast<uint64_t>(lhs) & static_cast<uint64_t>(rhs);
}
inline constexpr HCUISAFeature operator|(HCUISAFeature lhs, HCUISAFeature rhs) {
  return static_cast<HCUISAFeature>(static_cast<uint64_t>(lhs) | static_cast<uint64_t>(rhs));
}
HCUISAFeature deduceHCUISAFeature(llvm::StringRef arch);
bool supportsHCUISAFeature(llvm::StringRef arch, HCUISAFeature feature);

} // namespace mlir::triton::HCU

#endif // TRITON_THIRD_PARTY_HCU_INCLUDE_TRITONHCUGPUTOLLVM_TARGETUTILS_H_
