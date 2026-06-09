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
#ifndef GCU_DIALECT_MEMREF_EXT_DIALECT_H
#define GCU_DIALECT_MEMREF_EXT_DIALECT_H

#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/IR/SymbolTable.h"

#include "Dialect/MemrefExt/IR/MemrefExtDialect.h.inc"
#define GET_OP_CLASSES
#include "Dialect/MemrefExt/IR/MemrefExtOps.h.inc"
#define GET_ATTRDEF_CLASSES

namespace mlir {
namespace memref_ext {} // namespace memref_ext
} // namespace mlir

#endif // GCU_DIALECT_MEMREF_EXT_DIALECT_H
