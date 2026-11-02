// MIT License
// Copyright (c) 2026 The FlagOS Contributors

#ifndef TRITON_METAX_COMMONIR_BSM_LAYOUT_H
#define TRITON_METAX_COMMONIR_BSM_LAYOUT_H

#include "mlir/IR/BuiltinOps.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include <optional>
#include <string>

namespace mlir::triton::tile {
struct BufferLayoutPlan;
struct TensorLayoutRequirements;
} // namespace mlir::triton::tile

namespace mlir::triton::metax {

struct DotLayoutChoice;

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
// An optional planned shared type permits validation before rewriting buffers.
std::optional<std::string> checkBsmLayout(BsmLayoutChain chain,
                                         Attribute encoding,
                                         gpu::MemDescType shared = {});

LogicalResult
collectBsmLayoutRequirements(ModuleOp module, ArrayRef<DotLayoutChoice> choices,
                            tile::TensorLayoutRequirements &requirements,
                            const tile::BufferLayoutPlan *buffers = nullptr);

} // namespace mlir::triton::metax

#endif
