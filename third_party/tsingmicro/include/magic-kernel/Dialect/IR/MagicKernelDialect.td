//===------------------- MagicKernelDialect.td ----------------------------===//
//
// Copyright (C) 2020-2025 Terapines Technology (Wuhan) Co., Ltd
// All rights reserved.
//
//===----------------------------------------------------------------------===//

#ifndef MAGIC_KERNEL_DIALECT
#define MAGIC_KERNEL_DIALECT

include "mlir/IR/OpBase.td"

def MagicKernelDialect : Dialect {
  let name = "mk";

  let cppNamespace = "::mlir::mk";

  let summary = "The Magic Kernel IR in MLIR";

  let description = [{
    Magic Kernel Dialect.

    Dependent Dialects:
      * Memref
        * copy, alloc
      * Bufferization
        * to_tensor
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

include "magic-kernel/Dialect/IR/MagicKernelTypes.td"

#endif // MAGIC_KERNEL_DIALECT
