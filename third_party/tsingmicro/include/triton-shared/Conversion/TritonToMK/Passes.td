#ifndef TRITON_TO_MK_CONVERSION_PASSES
#define TRITON_TO_MK_CONVERSION_PASSES

include "mlir/Pass/PassBase.td"

def TritonToMK : Pass<"triton-to-mk", "mlir::ModuleOp"> {
  let summary = "Convert triton triton pointer ops to mk";
}

#endif
