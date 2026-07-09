#ifndef ILUVATAR_TLE_DIALECT
#define ILUVATAR_TLE_DIALECT

include "mlir/IR/OpBase.td"

#ifdef __ILUVATAR_TLE__
def IluvatarTle_Dialect : Dialect {
  let name = "iluvatar_tle";
  let cppNamespace = "::mlir::triton::iluvatar_tle";
  let description = [{
    Iluvatar backend-local Triton Language Extension dialect.
  }];

  let dependentDialects = [
    "triton::TritonDialect",
    "triton::gpu::TritonGPUDialect",
  ];
  let usePropertiesForAttributes = 1;
}
#endif // __ILUVATAR_TLE__

#endif // ILUVATAR_TLE_DIALECT
