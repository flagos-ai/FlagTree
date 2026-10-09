// MIT License
// Copyright (c) 2026 The FlagOS Contributors

#ifndef TRITON_METAX_COMMONIR_BSM_LAYOUT_H
#define TRITON_METAX_COMMONIR_BSM_LAYOUT_H

#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include <optional>
#include <string>

namespace mlir::triton::metax {

struct BsmLayoutChain {
  gpu::LocalLoadOp load;
  gpu::BsmPermOp perm;
  SmallVector<gpu::ConvertLayoutOp> conversions;
};

// A missing chain means an ordinary operand. Malformed split-load chains
// diagnose a failure instead of falling through to ordinary dot lowering.
FailureOr<std::optional<BsmLayoutChain>> getBsmLayoutChain(Value operand);

// Shared by candidate enumeration and layout application. Returns a reason when
// the native split LDS / BSM lowering cannot implement this B encoding.
std::optional<std::string> checkBsmLayout(BsmLayoutChain chain,
                                          Attribute encoding);

} // namespace mlir::triton::metax

#endif
