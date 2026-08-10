#include "TritonILUVATARGPUToLLVM/TargetUtils.h"

namespace mlir::triton::ILUVATAR {

ISAFamily deduceISAFamily(llvm::StringRef arch) {
  if (arch.starts_with("ivcore11"))
    return ISAFamily::IVCORE11;
  if (arch.starts_with("ivcore30"))
    return ISAFamily::IVCORE30;
  return ISAFamily::Unknown;
}

} // namespace mlir::triton::ILUVATAR
