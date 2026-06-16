#ifndef TRITON_ILUVATARGPU_DIALECT
#define TRITON_ILUVATARGPU_DIALECT

include "mlir/IR/OpBase.td"

def TritonILUVATARGPU_Dialect : Dialect {
  let name = "iluvatarg";
  let cppNamespace = "::mlir::triton::iluvatargpu";

  let description = [{
    TritonILUVATARGPU Dialect hosts ILUVATAR specific ops at TritonGPU abstraction level.
  }];

  let dependentDialects = ["triton::TritonDialect"];

  // let useDefaultAttributePrinterParser = 1;
  let usePropertiesForAttributes = 1;
}

#endif
