#ifndef MUSA_DIALECT
#define MUSA_DIALECT

include "mlir/IR/OpBase.td"

def MUSA_Dialect : Dialect {
  let name = "ttmg";
  let cppNamespace = "::mlir::triton::musa";

  let description = [{
    Triton MUSA GPU middle dialect.
  }];

  let dependentDialects = [
    "mlir::LLVM::LLVMDialect"
  ];
}

#endif
