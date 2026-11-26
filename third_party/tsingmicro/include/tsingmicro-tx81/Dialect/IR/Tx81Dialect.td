//===----------------------- Tx81Dialect.td -------------------------------===//
//
// Copyright (C) 2020-2025 Terapines Technology (Wuhan) Co., Ltd
// All rights reserved.
//
//===----------------------------------------------------------------------===//

#ifndef TSINGMICRO_TX81_DIALECT
#define TSINGMICRO_TX81_DIALECT

include "mlir/IR/OpBase.td"

def Tx81Dialect : Dialect {
  let name = "tx";

  let cppNamespace = "::mlir::tx";

  let summary = "The TsingMicro Tx81 IR in MLIR";

  let description = [{
    TsingMicro Tx81 Dialect.

    Dependent Dialects:
      * MK
      * Memref
      * Bufferization
  }];

  let dependentDialects = [
  ];

  let extraClassDeclaration = [{
    void registerTypes();
  }];

  // let hasConstantMaterializer = 1;
  // let useDefaultTypePrinterParser = 1;
  let usePropertiesForAttributes = 1;
}

include "tsingmicro-tx81/Dialect/IR/Tx81Types.td"

#endif // TSINGMICRO_TX81_DIALECT
