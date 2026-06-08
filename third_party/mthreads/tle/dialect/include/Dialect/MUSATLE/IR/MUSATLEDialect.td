#ifndef MUSA_TLE_DIALECT
#define MUSA_TLE_DIALECT

include "mlir/IR/OpBase.td"

#ifdef __TLE__
def MUSATLE_Dialect : Dialect {
  let name = "musa_tle";
  let cppNamespace = "::mlir::triton::musa_tle";
  let description = [{
    MUSA backend-local Triton Language Extension dialect.
  }];

  let dependentDialects = [
    "triton::TritonDialect",
    "triton::gpu::TritonGPUDialect",
  ];
  let usePropertiesForAttributes = 1;
}
#endif // __TLE__

#endif // MUSA_TLE_DIALECT
