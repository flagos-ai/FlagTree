#ifndef TLE_RAW_DIALECT
#define TLE_RAW_DIALECT

include "mlir/IR/OpBase.td"

def Tle_Dialect : Dialect {
  let name = "tle";
  let cppNamespace = "::mlir::triton::tle";
  let description = [{
    Triton Language Extension dialect.
  }];

  let dependentDialects = [
    "mlir::LLVM::LLVMDialect",
    "triton::TritonDialect",
    "triton::gpu::TritonGPUDialect",
  ];
  let usePropertiesForAttributes = 1;
}

#endif
