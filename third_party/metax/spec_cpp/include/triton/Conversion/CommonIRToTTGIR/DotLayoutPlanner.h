// MIT License
//
// Copyright (c) 2025 The FlagOS Contributors

#ifndef TRITON_METAX_COMMONIR_DOT_LAYOUT_PLANNER_H
#define TRITON_METAX_COMMONIR_DOT_LAYOUT_PLANNER_H

#include "mlir/IR/Attributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Support/LogicalResult.h"
#include "triton/Conversion/CommonIRToTTGIR/MmaLayout.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringMap.h"

#include <string>

namespace mlir::triton::metax {

// Enumerate locally valid layouts, with the default first.
// Full-kernel compilation may still reject a candidate.
FailureOr<SmallVector<DotLayoutPlan>>
generateDotLayoutPlans(triton::DotOp dot, int computeCapability, int numWarps);

struct DotLayoutDomain {
  triton::DotOp dot;
  std::string functionName;
  unsigned dotIndex;
  SmallVector<DotLayoutPlan> plans;
};

// Validate choices and insert A/B/result set_layout ops into unencoded TTIR.
LogicalResult
insertDotLayoutHints(ModuleOp module, int capability, int numWarps,
                     const llvm::StringMap<DotLayoutPlan> &selection);

// Number dots in preorder within each function. IDs are valid only until
// the IR structure changes.
FailureOr<SmallVector<DotLayoutDomain>>
enumerateDotLayoutPlans(ModuleOp module, int computeCapability, int numWarps);

// Stable layout ID, independent of context and SSA names.
std::string getDotLayoutPlanId(const DotLayoutPlan &plan);
// Group plans by accumulator layout, ignoring operand transfer details.
std::string getDotLayoutProfileId(const DotLayoutPlan &plan);

} // namespace mlir::triton::metax

#endif // TRITON_METAX_COMMONIR_DOT_LAYOUT_PLANNER_H
