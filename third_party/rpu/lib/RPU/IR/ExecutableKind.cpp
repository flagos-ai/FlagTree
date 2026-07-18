// Copyright 2026 FlagOS Contributors
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.

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
