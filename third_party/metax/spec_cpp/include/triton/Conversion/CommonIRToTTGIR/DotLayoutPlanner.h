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
#include "llvm/ADT/StringMap.h"

#include <string>

namespace mlir::triton::metax {

struct DotLayoutPlan {
  Attribute mma;
  Attribute operandA;
  Attribute operandB;
};

// Enumerate locally valid layouts, with the default first.
// Full-kernel compilation may still reject a candidate.
FailureOr<SmallVector<DotLayoutPlan>>
generateDotLayoutPlans(triton::DotOp dot, int computeCapability, int numWarps);

struct DotLayoutDomain {
  triton::DotOp dot;
  // <function>/dot/<index>. Each function numbers all dots in preorder from
  // zero, including fixed dots. Re-enumerate if the name or dot order changes.
  std::string dotId;
  SmallVector<DotLayoutPlan> plans;
};

// Skip dots with explicit dot-operand layouts on both A and B and an explicit
// MMA layout on the accumulator or result.
FailureOr<SmallVector<DotLayoutDomain>>
enumerateDotLayoutPlans(ModuleOp module, int computeCapability, int numWarps);

// Insert one selected plan per returned dotId into unencoded TTIR.
LogicalResult
insertDotLayoutHints(ModuleOp module,
                     const llvm::StringMap<DotLayoutPlan> &selection);

// Stable layout ID, independent of context and SSA names.
std::string getDotLayoutPlanId(const DotLayoutPlan &plan);
// Group plans by accumulator layout, ignoring operand transfer details.
std::string getDotLayoutProfileId(const DotLayoutPlan &plan);

} // namespace mlir::triton::metax

#endif // TRITON_METAX_COMMONIR_DOT_LAYOUT_PLANNER_H
