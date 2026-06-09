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
#include "Dialect/TritonGCU/IR/TritonGCUTypes.h"
#include "Dialect/TritonGCU/IR/TritonGCUDialect.h"

#include "mlir/IR/Builders.h"              // required by `Types.cpp.inc`
#include "mlir/IR/DialectImplementation.h" // required by `Types.cpp.inc`
#include "llvm/ADT/TypeSwitch.h"           // required by `Types.cpp.inc`

using namespace mlir;
using namespace mlir::triton;
using namespace mlir::triton::gcu;

#define GET_TYPEDEF_CLASSES
#include "Dialect/TritonGCU/IR/TritonGCUTypes.cpp.inc"

//===----------------------------------------------------------------------===//
// Triton GCU Dialect
//===----------------------------------------------------------------------===//
void TritonGCUDialect::registerTypes() {
  addTypes<
#define GET_TYPEDEF_LIST
#include "Dialect/TritonGCU/IR/TritonGCUTypes.cpp.inc" // NOLINT: This file generated situationally via different environment variables
      >();
}
