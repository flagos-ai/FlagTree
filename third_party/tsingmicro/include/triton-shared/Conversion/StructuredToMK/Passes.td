#ifndef TRITON_STRUCTURED_TO_MK_CONVERSION_PASSES
#define TRITON_STRUCTURED_TO_MK_CONVERSION_PASSES

include "mlir/Pass/PassBase.td"

def StructuredToMK : Pass<"structured-to-mk", "mlir::ModuleOp"> {
  let summary = "Convert triton structured pointer ops to mk";
}

#endif
