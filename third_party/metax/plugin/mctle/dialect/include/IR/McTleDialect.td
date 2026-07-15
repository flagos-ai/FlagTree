#ifndef MCTLE_DIALECT
#define MCTLE_DIALECT

include "mlir/IR/OpBase.td"

def McTle_Dialect : Dialect {
  let name = "mctle";
  let cppNamespace = "::mlir::triton::mctle";
  let description = [{
    Triton Language Extension dialect with MACA.
  }];

  let dependentDialects = [
    "mlir::LLVM::LLVMDialect",
    "triton::TritonDialect",
    "triton::gpu::TritonGPUDialect",
  ];
  let usePropertiesForAttributes = 1;
}

#endif
