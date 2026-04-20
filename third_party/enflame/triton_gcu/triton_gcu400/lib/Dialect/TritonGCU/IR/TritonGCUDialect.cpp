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

#include "Dialect/TritonGCU/IR/TritonGCUDialect.h"
#include "Dialect/TritonGCU/IR/TritonGCUDialect.cpp.inc"
#include "Dialect/TritonGCU/IR/TritonGCUTypes.h"

#include "mlir/IR/DialectImplementation.h"
#include "mlir/Transforms/InliningUtils.h"
#include "llvm/ADT/StringSwitch.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/raw_ostream.h"

#define GET_ATTRDEF_CLASSES
#include "Dialect/TritonGCU/IR/TritonGCUOpsAttributes.cpp.inc"
// #define GET_OP_CLASSES
// #include "Dialect/TritonGCU/IR/TritonGCUOps.cpp.inc"
#include "Dialect/TritonGCU/IR/TritonGCUOpsEnums.cpp.inc"

using namespace ::mlir;
using namespace ::mlir::triton;
using namespace ::mlir::triton::gcu;

void TritonGCUDialect::initialize() {
  registerTypes();
  addOperations<
#define GET_OP_LIST
#include "Dialect/TritonGCU/IR/TritonGCUOps.cpp.inc" // NOLINT: This file generated situationally via different environment variables
      >();

  addAttributes<
#define GET_ATTRDEF_LIST
#include "Dialect/TritonGCU/IR/TritonGCUOpsAttributes.cpp.inc" // NOLINT: This file generated situationally via different environment variables
      >();
}

Operation *TritonGCUDialect::materializeConstant(OpBuilder &builder,
                                                 Attribute value, Type type,
                                                 Location loc) {
  return arith::ConstantOp::materialize(builder, value, type, loc);
}
