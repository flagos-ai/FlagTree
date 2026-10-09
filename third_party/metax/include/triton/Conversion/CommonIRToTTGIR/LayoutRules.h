#ifndef TRITON_METAX_COMMONIR_LAYOUT_RULES_H
#define TRITON_METAX_COMMONIR_LAYOUT_RULES_H

#include "mlir-ext/Conversion/CommonIRToTritonGPU/LayoutRules.h"
#include "triton/Conversion/CommonIRToTTGIR/DotLayoutPlanner.h"

namespace mlir::triton::metax {

// All collectors are read-only. Requirements are committed once, after every
// rule has accepted the candidate.
LogicalResult
collectBsmLayoutRequirements(ModuleOp module, ArrayRef<DotLayoutChoice> choices,
                             tile::TensorLayoutRequirements &requirements);
void collectLocalLoadLayoutRequirements(
    ArrayRef<DotLayoutChoice> choices,
    tile::TensorLayoutRequirements &requirements,
    const tile::BufferLayoutPlan &buffers);
FailureOr<tile::BufferLayoutPlan>
planSharedLayouts(ModuleOp module, ArrayRef<DotLayoutChoice> choices,
                  tile::TensorLayoutRequirements &requirements);

} // namespace mlir::triton::metax
#endif
