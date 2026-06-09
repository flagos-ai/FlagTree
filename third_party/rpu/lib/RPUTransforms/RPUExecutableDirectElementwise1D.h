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
