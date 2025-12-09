#ifndef TRITON_DIALECT_FLAGTREE_TRANSFORMS_PASSES_H
#define TRITON_DIALECT_FLAGTREE_TRANSFORMS_PASSES_H

#include <memory>

#include "mlir/Pass/Pass.h"

namespace mlir::triton::flagtree {
#define GEN_PASS_DECL
#define GEN_PASS_REGISTRATION
#include "triton/Dialect/FlagTree/Transforms/Passes.h.inc"
} // namespace mlir::triton::flagtree

#endif
