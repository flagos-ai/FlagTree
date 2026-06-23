// Copyright 2026- Xcoresigma Technology Co., Ltd

#ifndef TRITON_TLE_DSA_DIALECT
#define TRITON_TLE_DSA_DIALECT

include "mlir/IR/OpBase.td"

def Tle_Dialect : Dialect {
  let name = "tle";
  let cppNamespace = "::mlir::triton::tle";
  let description = [{
    Triton Language Extension Dialect.
  }];

  let dependentDialects = [
    "mlir::LLVM::LLVMDialect",
    "triton::TritonDialect",
  ];

  let usePropertiesForAttributes = 1;
}

#endif
