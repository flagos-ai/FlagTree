// MIT License
//
// Copyright (c) 2025 The FlagOS Contributors

#ifndef TRITON_METAX_COMMONIR_DOT_LAYOUT_PLANNER_H
#define TRITON_METAX_COMMONIR_DOT_LAYOUT_PLANNER_H

#include "mlir/IR/Attributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Support/LogicalResult.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "llvm/ADT/SmallVector.h"

#include <string>

namespace mlir::triton::metax {

struct DotLayoutPlan {
  Attribute mma;
  Attribute operandA;
  Attribute operandB;
};

// Enumerate MACA MMA layout plans passing the per-dot geometry checks.
// The first entry is the conservative fallback. Multi-dot coordination (picking
// a mutually compatible plan across chained dots) and full-kernel lowering
// validation are separate steps.
FailureOr<SmallVector<DotLayoutPlan>>
generateDotLayoutPlans(triton::DotOp dot, int computeCapability, int numWarps);

struct DotLayoutDomain {
  triton::DotOp dot;
  std::string functionName;
  unsigned dotIndex;
  SmallVector<DotLayoutPlan> plans;
};

// Read-only module query. Dots are visited in preorder, numbered separately
// within each function. These coordinates identify the current IR snapshot,
// not operations after inlining, unrolling, or other structural rewrites.
FailureOr<SmallVector<DotLayoutDomain>>
enumerateDotLayoutPlans(ModuleOp module, int computeCapability, int numWarps);

// Layout identities are independent of MLIR contexts, source locations, SSA
// names and enumeration order. They are not whole-kernel compilation keys.
std::string getDotLayoutPlanId(const DotLayoutPlan &plan);
// Matches Gluon's haveSameC500AccumulatorProfile: MMA version, warp grid,
// elementsMNK, colMajor and CTA layout, excluding operand transfer details.
std::string getDotLayoutProfileId(const DotLayoutPlan &plan);

} // namespace mlir::triton::metax

#endif // TRITON_METAX_COMMONIR_DOT_LAYOUT_PLANNER_H
