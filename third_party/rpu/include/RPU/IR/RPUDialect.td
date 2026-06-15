#ifndef RPU_EXEC_DIALECT
#define RPU_EXEC_DIALECT

include "mlir/IR/OpBase.td"

def RPU_Dialect : Dialect {
  let name = "rpu";
  let cppNamespace = "::mlir::rpu::exec";

  let summary = "Internal executable RPU dialect";

  let description = [{
    The `rpu` dialect is an internal executable IR used between RPUPlan and
    RPU DSL source emission. It is not a user-facing Triton API and is not the
    RPU C++ DSL API.
  }];

  let dependentDialects = [
    "mlir::triton::TritonDialect"
  ];

  let useDefaultTypePrinterParser = 1;
}

#endif
