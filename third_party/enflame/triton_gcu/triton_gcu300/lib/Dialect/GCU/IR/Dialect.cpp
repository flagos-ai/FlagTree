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
#include "Dialect/GCU/IR/Dialect.h"
#include "Dialect/GCU/IR/Dialect.cpp.inc"
#include "Dialect/GCU/IR/Types.h"

#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"
#include "mlir/IR/DialectImplementation.h"
#include "mlir/Transforms/InliningUtils.h"
#include "llvm/ADT/StringSwitch.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/raw_ostream.h"

#define GET_ATTRDEF_CLASSES
#include "Dialect/GCU/IR/OpsAttributes.cpp.inc"
#define GET_OP_CLASSES
#include "Dialect/GCU/IR/Ops.cpp.inc"
#include "Dialect/GCU/IR/OpsEnums.cpp.inc"

using namespace mlir;
using namespace mlir::gcu;

void GCUDialect::initialize() {
  registerTypes();
  addOperations<
#define GET_OP_LIST
#include "Dialect/GCU/IR/Ops.cpp.inc" // NOLINT: This file generated situationally via different environment variables
      >();
  addAttributes<
#define GET_ATTRDEF_LIST
#include "Dialect/GCU/IR/OpsAttributes.cpp.inc" // NOLINT: This file generated situationally via different environment variables
      >();

  // We can also add interface here.
  // addInterfaces<GCUInlinerInterface>();
}

Operation *GCUDialect::materializeConstant(OpBuilder &builder, Attribute value,
                                           Type type, Location loc) {
  return arith::ConstantOp::materialize(builder, value, type, loc);
}

//===----------------------------------------------------------------------===//
// GCU target attribute.
//===----------------------------------------------------------------------===//
LogicalResult
GCUTargetAttr::verify(function_ref<InFlightDiagnostic()> emitError,
                      int optLevel, StringRef triple, StringRef chip,
                      StringRef arch, StringRef features, StringRef abiVersion,
                      DictionaryAttr flags, ArrayAttr files) {
  if (optLevel < 0 || optLevel > 3) {
    emitError() << "The optimization level must be a number between 0 and 3.";
    return failure();
  }
  if (triple.empty()) {
    emitError() << "The target triple cannot be empty.";
    return failure();
  }
  if (chip.empty()) {
    emitError() << "The target chip cannot be empty.";
    return failure();
  }
  if (arch.empty()) {
    emitError() << "The target arch cannot be empty.";
    return failure();
  }
  if (abiVersion != "1") {
    emitError() << "Invalid ABI version, it must be `1`.";
    return failure();
  }
  if (files && !llvm::all_of(files, [](::mlir::Attribute attr) {
        return attr && mlir::isa<StringAttr>(attr);
      })) {
    emitError() << "All the elements in the `link` array must be strings.";
    return failure();
  }
  return success();
}
