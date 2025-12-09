#ifndef FLAGTREE_DIALECT
#define FLAGTREE_DIALECT

include "mlir/IR/OpBase.td"

def FlagTree_Dialect : Dialect {
  let name = "flagtree";
  let cppNamespace = "::mlir::triton::flagtree";
  let description = [{
    Flagtree dialect.
  }];

  let dependentDialects = [
    "mlir::LLVM::LLVMDialect",
    "triton::TritonDialect",
    "triton::gpu::TritonGPUDialect",
  ];
  let usePropertiesForAttributes = 1;
}

#endif
