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

#include "Dialect/MathExt/IR/MathExt.h"
#include "Dialect/MathExt/IR/MathExtDialect.cpp.inc"
#include "Dialect/MathExt/IR/MathExtTypes.h"

#include "mlir/IR/DialectImplementation.h"
#include "mlir/Transforms/InliningUtils.h"
#include "llvm/ADT/StringSwitch.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/raw_ostream.h"

using namespace mlir;
using namespace mlir::math_ext;

void MathExtDialect::initialize() {
  // registerTypes();
  addOperations<
#define GET_OP_LIST
#include "Dialect/MathExt/IR/MathExtOps.cpp.inc" // NOLINT: This file generated situationally via different environment variables
      >();
}

Operation *MathExtDialect::materializeConstant(OpBuilder &builder,
                                               Attribute value, Type type,
                                               Location loc) {
  return arith::ConstantOp::materialize(builder, value, type, loc);
}
