#ifndef MTGPU_DIALECT
#define MTGPU_DIALECT

include "mlir/IR/OpBase.td"

def MTGPUDialect : Dialect {
  let name = "mtgpu";
  let cppNamespace = "::mlir::triton::mtgpu";
  let useDefaultTypePrinterParser = 1;

  let description = [{
    MUSA backend-private low-level hardware and ABI dialect.
  }];

  let dependentDialects = [
    "triton::gpu::TritonGPUDialect",
    "mlir::LLVM::LLVMDialect"
  ];
}

#endif
