#ifndef TRITON_THIRD_PARTY_ILUVATAR_INCLUDE_TRITONILUVATARGPUTOLLVM_TARGETUTILS_H_
#define TRITON_THIRD_PARTY_ILUVATAR_INCLUDE_TRITONILUVATARGPUTOLLVM_TARGETUTILS_H_

#include "llvm/ADT/StringRef.h"

namespace mlir::triton::ILUVATAR {

// A list of ISA families we care about.
enum class ISAFamily {
  Unknown,
  IVCORE11,
  IVCORE30,
};

// Deduces the corresponding ISA family for the given target |arch|.
ISAFamily deduceISAFamily(llvm::StringRef arch);

} // namespace mlir::triton::ILUVATAR

#endif // TRITON_THIRD_PARTY_ILUVATAR_INCLUDE_TRITONILUVATARGPUTOLLVM_TARGETUTILS_H_
