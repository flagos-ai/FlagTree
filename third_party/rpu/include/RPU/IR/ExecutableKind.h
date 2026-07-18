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
