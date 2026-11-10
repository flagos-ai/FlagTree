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
#include "mlir/IR/DialectImplementation.h"
#include "mlir/IR/OpImplementation.h"

// clang-format off
#include "Dialect/GCUWS/IR/Dialect.h"
#include "Dialect/GCUWS/IR/Dialect.cpp.inc"
#include "llvm/ADT/TypeSwitch.h" // required by `Types.cpp.inc`
// clang-format on

using namespace mlir;
using namespace mlir::triton::gcuws;

#define GET_ATTRDEF_CLASSES
#include "Dialect/GCUWS/IR/GCUWSAttrDefs.cpp.inc"

#define GET_TYPEDEF_CLASSES
#include "Dialect/GCUWS/IR/Types.cpp.inc"

void mlir::triton::gcuws::GCUWSDialect::initialize() {
  addAttributes<
#define GET_ATTRDEF_LIST
#include "Dialect/GCUWS/IR/GCUWSAttrDefs.cpp.inc" // NOLINT(build/include)
      >();

  addTypes<
#define GET_TYPEDEF_LIST
#include "Dialect/GCUWS/IR/Types.cpp.inc" // NOLINT(build/include)
      >();

  addOperations<
#define GET_OP_LIST
#include "Dialect/GCUWS/IR/Ops.cpp.inc"
      >();
}
