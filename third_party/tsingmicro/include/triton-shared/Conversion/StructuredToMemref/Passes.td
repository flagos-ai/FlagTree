#ifndef TRITON_ARITH_TO_LINALG_CONVERSION_PASSES
#define TRITON_ARITH_TO_LINALG_CONVERSION_PASSES

include "mlir/Pass/PassBase.td"

def StructuredToMemref : Pass<"structured-to-memref", "mlir::ModuleOp"> {
  let summary = "Convert triton structured pointer ops to memref";
}

#endif
