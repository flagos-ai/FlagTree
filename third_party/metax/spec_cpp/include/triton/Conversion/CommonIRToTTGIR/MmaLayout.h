// MIT License
// Copyright (c) 2026 The FlagOS Contributors

#ifndef TRITON_METAX_MMA_LAYOUT_H
#define TRITON_METAX_MMA_LAYOUT_H

#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include <optional>
#include <string>

namespace mlir::triton::metax {

struct DotLayoutPlan {
  Attribute mma;
  Attribute operandA;
  Attribute operandB;
};

struct DotLayoutChoice {
  triton::DotOp dot;
  DotLayoutPlan plan;
};

// Shared M/N register-group bound for selection and validation.
unsigned getMmaRegisterGroupLimit(int64_t size, unsigned atomSize,
                                  unsigned warps);

// Call after checking the target and geometry preconditions.
gpu::MACAMmaEncodingAttr buildMmaEncoding(MLIRContext *ctx,
                                          int computeCapability,
                                          ArrayRef<unsigned> warpsPerCTA,
                                          ArrayRef<unsigned> elementsMNK,
                                          unsigned colMajor);
DotLayoutPlan makeDotLayoutPlan(triton::DotOp dot,
                                gpu::MACAMmaEncodingAttr mma);

std::optional<StringRef> checkMacaDotPreconditions(triton::DotOp dot,
                                                   int computeCapability,
                                                   int numWarps);
// Check a fixed layout without diagnostics or IR mutations.
std::optional<StringRef>
checkMacaDotLayout(triton::DotOp dot, Attribute mma, Attribute operandA,
                   Attribute operandB, int computeCapability, int numWarps);

inline constexpr unsigned kBsmSlots = 32;

struct BsmLayoutChain {
  gpu::LocalLoadOp load;
  gpu::BsmPermOp perm;
  SmallVector<gpu::ConvertLayoutOp> conversions;
};

// Return an empty optional for non-BSM operands, or failure for invalid chains.
FailureOr<std::optional<BsmLayoutChain>> getBsmLayoutChain(Value operand);

// Check native split LDS / BSM mapping, optionally against planned shared
// storage.
std::optional<std::string> checkBsmLayout(BsmLayoutChain chain,
                                          Attribute encoding,
                                          gpu::MemDescType shared = {});

} // namespace mlir::triton::metax

#endif
