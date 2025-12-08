/**
 * Copyright 2024-2026 Enflame. All Rights Reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *  http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
#ifndef MEMREF_EXT_DIALECT
#define MEMREF_EXT_DIALECT

include "mlir/IR/OpBase.td"

def MemrefExt_Dialect : Dialect {
  let name = "memref_ext";

  let cppNamespace = "::mlir::memref_ext";

  let summary = "The MemrefExt IR in MLIR";

  let description = [{
    MemrefExt Dialect.
  }];

  let dependentDialects = [
    "arith::ArithDialect",
    "memref::MemRefDialect",
  ];

  // let extraClassDeclaration = [{
  //   void registerTypes();
  // }];

  let hasConstantMaterializer = 1;
  // let useDefaultTypePrinterParser = 1;
  // let useDefaultAttributePrinterParser = 1;
  // let usePropertiesForAttributes = 1;
}

#endif // MEMREF_EXT_DIALECT
