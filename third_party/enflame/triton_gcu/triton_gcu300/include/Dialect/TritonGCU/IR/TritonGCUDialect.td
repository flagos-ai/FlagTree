#ifndef TRITONGCU_DIALECT
#define TRITONGCU_DIALECT

include "mlir/IR/OpBase.td"

def TritonGCU_Dialect : Dialect {
  let name = "triton_gcu";

  let cppNamespace = "::mlir::triton::gcu";

  let summary = "The TritonGCU IR in MLIR";

  let description = [{
    TritonGCU Dialect.
  }];

  let dependentDialects = [
    "::mlir::arith::ArithDialect",
  ];

  let extraClassDeclaration = [{
    void registerTypes();
  }];

  let hasConstantMaterializer = 1;
  let useDefaultTypePrinterParser = 1;
  // let useDefaultAttributePrinterParser = 1;
  let usePropertiesForAttributes = 1;
}

#endif  // TRITONGCU_DIALECT
