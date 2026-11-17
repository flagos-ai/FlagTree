// MIT License
// Copyright (c) 2026 The FlagOS Contributors

#ifndef TRITON_METAX_MMA_LAYOUT_H
#define TRITON_METAX_MMA_LAYOUT_H

#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include <optional>

namespace mlir::triton::metax {

struct DotLayoutPlan {
  Attribute mma;
  Attribute operandA;
  Attribute operandB;
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

std::optional<StringRef> checkMacaDotPreconditions(triton::DotOp dot,
                                                   int computeCapability,
                                                   int numWarps);
// Check a fixed layout without diagnostics or IR mutations.
std::optional<StringRef>
checkMacaDotLayout(triton::DotOp dot, Attribute mma, Attribute operandA,
                   Attribute operandB, int computeCapability, int numWarps);

} // namespace mlir::triton::metax

#endif
