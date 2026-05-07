#include "TritonHCUGPUToLLVM/TargetUtils.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/TargetParser/TargetParser.h"

namespace mlir::triton::HCU {

ISAFamily deduceISAFamily(llvm::StringRef arch) {
  llvm::AMDGPU::GPUKind kind = llvm::AMDGPU::parseArchAMDGCN(arch);

  // See https://llvm.org/docs/HCUGPUUsage.html#processors for how to categorize
  // the following target gfx architectures.

  if (kind == llvm::AMDGPU::GK_GFX1250)
    return ISAFamily::GFX1250;

  // CDNA ISA cases
  switch (kind) {
  case llvm::AMDGPU::GK_GFX950:
    return ISAFamily::CDNA4;
  case llvm::AMDGPU::GK_GFX942:
    return ISAFamily::CDNA3;
  case llvm::AMDGPU::GK_GFX90A:
    return ISAFamily::CDNA2;
  case llvm::AMDGPU::GK_GFX908:
    return ISAFamily::CDNA1;
  default:
    break;
  }

  // HCU ISA cases
  switch (kind) {
  case llvm::AMDGPU::GK_GFX928: /* ZD */
  case llvm::AMDGPU::GK_GFX936: /* BMZ */
  case llvm::AMDGPU::GK_GFX938: /* NMZ */
  case llvm::AMDGPU::GK_GFX92A: /* YY */
    return ISAFamily::CDNA3;
  case llvm::AMDGPU::GK_GFX946: /* SB */
    return ISAFamily::CDNA4;
  default:
    break;
  }

  // RDNA ISA cases
  if (kind >= llvm::AMDGPU::GK_GFX1200 && kind <= llvm::AMDGPU::GK_GFX1201)
    return ISAFamily::RDNA4;
  if (kind >= llvm::AMDGPU::GK_GFX1100 && kind <= llvm::AMDGPU::GK_GFX1153)
    return ISAFamily::RDNA3;
  if (kind >= llvm::AMDGPU::GK_GFX1030 && kind <= llvm::AMDGPU::GK_GFX1036)
    return ISAFamily::RDNA2;
  if (kind >= llvm::AMDGPU::GK_GFX1010 && kind <= llvm::AMDGPU::GK_GFX1013)
    return ISAFamily::RDNA1;

  return ISAFamily::Unknown;
}

bool supportsVDot(llvm::StringRef arch) {
  switch (deduceISAFamily(arch)) {
  case HCU::ISAFamily::CDNA1:
  case HCU::ISAFamily::CDNA2:
  case HCU::ISAFamily::CDNA3:
  case HCU::ISAFamily::CDNA4:
  case HCU::ISAFamily::RDNA2:
  case HCU::ISAFamily::RDNA3:
  case HCU::ISAFamily::RDNA4:
    return true;
  default:
    break;
  }
  return false;
}

bool isCDNA(ISAFamily isaFamily) {
  switch (isaFamily) {
  case ISAFamily::CDNA1:
  case ISAFamily::CDNA2:
  case ISAFamily::CDNA3:
  case ISAFamily::CDNA4:
    return true;
  default:
    break;
  }

  return false;
}

bool isRDNA(ISAFamily isaFamily) {
  switch (isaFamily) {
  case ISAFamily::RDNA1:
  case ISAFamily::RDNA2:
  case ISAFamily::RDNA3:
  case ISAFamily::RDNA4:
    return true;
  default:
    break;
  }

  return false;
}

// HCU ISA features
HCUISAFeature deduceHCUISAFeature(llvm::StringRef arch) {
  HCUISAFeature commonFeatures1 = HCUISAFeature::MMAC_LAYOUT |
                                  HCUISAFeature::MAMC_FP8 | HCUISAFeature::MLS |
                                  HCUISAFeature::CVT_FP8F32;
  static const llvm::DenseMap<llvm::AMDGPU::GPUKind, HCUISAFeature>
      hcuIsaFeatures = {
          {llvm::AMDGPU::GK_GFX928, HCUISAFeature::NONE},
          {llvm::AMDGPU::GK_GFX936, HCUISAFeature::NONE},
          {llvm::AMDGPU::GK_GFX938, commonFeatures1},
          {llvm::AMDGPU::GK_GFX92A,
           commonFeatures1 | HCUISAFeature::MMAC_FP6FP4},
          {llvm::AMDGPU::GK_GFX946, commonFeatures1 |
                                        HCUISAFeature::MMAC_FP6FP4 |
                                        HCUISAFeature::MMAC_SCALE},
      };

  llvm::AMDGPU::GPUKind kind = llvm::AMDGPU::parseArchAMDGCN(arch);
  auto it = hcuIsaFeatures.find(kind);
  if (it == hcuIsaFeatures.end())
    return HCUISAFeature::NONE;
  return it->second;
}

bool supportsHCUISAFeature(llvm::StringRef arch, HCUISAFeature feature) {
  HCUISAFeature hcuIsaFeatures = deduceHCUISAFeature(arch);
  uint64_t featureBits = static_cast<uint64_t>(feature);
  return (uint64_t(hcuIsaFeatures) & featureBits) == featureBits;
}

} // namespace mlir::triton::HCU
