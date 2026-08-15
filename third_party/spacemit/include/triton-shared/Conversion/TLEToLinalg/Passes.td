#ifndef TLE_TO_LINALG_CONVERSION_PASSES
#define TLE_TO_LINALG_CONVERSION_PASSES

include "mlir/Pass/PassBase.td"

def TLEToLinalg : Pass<"tle-to-linalg", "mlir::ModuleOp"> {
  let summary = "Convert TLE (extract_tile/insert_tile) operations into tensor.extract_slice/insert_slice";
}

#endif
