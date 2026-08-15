#ifndef TLE_DIALECT
#define TLE_DIALECT

include "mlir/IR/OpBase.td"

def TLE_Dialect : Dialect {
  let name = "xtle";
  let cppNamespace = "::mlir::tle";

  let description = [{
    Triton Language Extension (TLE) dialect for spine-triton.
    Provides tile-level tensor operations: extract_tile, insert_tile.
  }];

  let dependentDialects = [
    "mlir::triton::TritonDialect",
  ];
}

#endif
