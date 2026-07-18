#include "RPU/IR/ExecutableKind.h"

#include <array>

namespace mlir {
namespace rpu {
namespace exec {

static constexpr std::array<ExecutableKernelKindContract, 14>
    kSupportedExecutableKindContracts = {{
        {"add", 3},
        {"gemm", 3},
        {"softmax", 2},
        {"convkxk", 3},
        {"resnet_block", 4},
        {"resnet50_bottleneck", 5},
        {"sqrt", 2},
        {"reduce_sum_all", 2},
        {"relu", 2},
        {"maximum", 3},
        {"reduce_sum_axis0", 2},
        {"reduce_sum_axis1", 2},
        {"broadcast_add", 3},
        {"generic", std::nullopt},
    }};

llvm::ArrayRef<ExecutableKernelKindContract>
supportedExecutableKernelKindContracts() {
  return kSupportedExecutableKindContracts;
}

const ExecutableKernelKindContract *
lookupExecutableKernelKindContract(llvm::StringRef kind) {
  for (const ExecutableKernelKindContract &contract :
       kSupportedExecutableKindContracts) {
    if (kind == contract.kind)
      return &contract;
  }
  return nullptr;
}

bool isSupportedExecutableKernelKind(llvm::StringRef kind) {
  return lookupExecutableKernelKindContract(kind) != nullptr;
}

std::optional<unsigned> expectedExecutableKernelArgCount(llvm::StringRef kind) {
  if (const ExecutableKernelKindContract *contract =
          lookupExecutableKernelKindContract(kind))
    return contract->f16PointerArgCount;
  return std::nullopt;
}

bool isSupportedExecutableConvKxKKernelSize(int64_t kernelSize) {
  return kernelSize == 3 || kernelSize == 5 || kernelSize == 7 ||
         kernelSize == 9;
}

} // namespace exec
} // namespace rpu
} // namespace mlir
