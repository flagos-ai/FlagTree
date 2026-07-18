#pragma once

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/StringRef.h"
#include <cstdint>
#include <optional>

namespace mlir {
namespace rpu {
namespace exec {

struct ExecutableKernelKindContract {
  llvm::StringLiteral kind;
  std::optional<unsigned> f16PointerArgCount;
};

llvm::ArrayRef<ExecutableKernelKindContract>
supportedExecutableKernelKindContracts();

const ExecutableKernelKindContract *
lookupExecutableKernelKindContract(llvm::StringRef kind);

bool isSupportedExecutableKernelKind(llvm::StringRef kind);

std::optional<unsigned> expectedExecutableKernelArgCount(llvm::StringRef kind);

bool isSupportedExecutableConvKxKKernelSize(int64_t kernelSize);

} // namespace exec
} // namespace rpu
} // namespace mlir
