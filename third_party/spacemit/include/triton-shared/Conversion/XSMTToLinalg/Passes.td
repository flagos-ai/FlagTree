#ifndef XSMT_TO_LINALG_CONVERSION_PASSES
#define XSMT_TO_LINALG_CONVERSION_PASSES

include "mlir/Pass/PassBase.td"

def XSMTToLinalg : Pass<"xsmt-to-linalg", "mlir::ModuleOp"> {
  let summary = "Convert xsmt operations into linalg";
}

#endif
