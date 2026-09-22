#ifndef TRITON_METAX_COMMON_IR_TO_TTGIR_PASSES_H
#define TRITON_METAX_COMMON_IR_TO_TTGIR_PASSES_H

#include "mlir/Pass/Pass.h"
#include "triton/Conversion/CommonIRToTTGIR/DotLayoutPlanner.h"
#include "llvm/ADT/StringMap.h"

namespace mlir::triton::metax {

#define GEN_PASS_DECL
#include "triton/Conversion/CommonIRToTTGIR/Passes.h.inc"

std::unique_ptr<Pass>
createCommonIRInjectDotPlan(CommonIRInjectDotPlanOptions options,
                            llvm::StringMap<DotLayoutPlan> selectedPlans);

#define GEN_PASS_REGISTRATION
#include "triton/Conversion/CommonIRToTTGIR/Passes.h.inc"

} // namespace mlir::triton::metax

#endif
