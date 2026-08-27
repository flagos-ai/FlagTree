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
#include "Dialect/MathExt/IR/MathExtTypes.h"
#include "Dialect/MathExt/IR/MathExt.h"

#include "mlir/IR/DialectImplementation.h" // required by `Types.cpp.inc`
#include "llvm/ADT/TypeSwitch.h"           // required by `Types.cpp.inc`

using namespace mlir;
using namespace mlir::math_ext;

#define GET_TYPEDEF_CLASSES
#include "Dialect/MathExt/IR/MathExtTypes.cpp.inc"

//===----------------------------------------------------------------------===//
// MathExt Dialect
//===----------------------------------------------------------------------===//
// void MathExtDialect::registerTypes() {
//   addTypes<
// #define GET_TYPEDEF_LIST
// #include "Dialect/MathExt/IR/MathExtTypes.cpp.inc"  // NOLINT: This file
// generated situationally via different environment variables
//       >();
// }
