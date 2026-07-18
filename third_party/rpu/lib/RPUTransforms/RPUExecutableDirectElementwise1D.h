// Copyright 2026 FlagOS Contributors
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.

#pragma once

#include "RPU/IR/Dialect.h"
#include "RPUTTIRPatternMatcher.h"
#include "mlir/Support/LogicalResult.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include <cstdint>

namespace mlir {
namespace rpu {

struct Elementwise1DExecutableRequest {
  TraceAnchor anchor;
  int64_t n = 0;
  int64_t logicalN = 0;
  bool masked = false;
  unsigned outputArgIndex = 0;
  SmallVector<unsigned, 4> inputArgIndices;
  SmallVector<exec::ExecutableCompactVectorBinaryBuildOp, 4> ops;
};

constexpr StringRef kElementwise1DFailureReason =
    "did not match supported compact 1D elementwise op sequence";

FailureOr<Elementwise1DExecutableRequest>
recognizeElementwise1DExecutableRequest(triton::FuncOp func);

} // namespace rpu
} // namespace mlir
