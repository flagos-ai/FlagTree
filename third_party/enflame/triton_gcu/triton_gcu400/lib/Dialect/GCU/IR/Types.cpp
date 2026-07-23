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
#include "Dialect/GCU/IR/Types.h"
#include "Dialect/GCU/IR/Dialect.h"

#include "mlir/IR/DialectImplementation.h" // required by `Types.cpp.inc`
#include "llvm/ADT/TypeSwitch.h"           // required by `Types.cpp.inc`

using namespace mlir;
using namespace mlir::gcu;

#define GET_TYPEDEF_CLASSES
#include "Dialect/GCU/IR/Types.cpp.inc"

//===----------------------------------------------------------------------===//
// GCU Dialect
//===----------------------------------------------------------------------===//
void GCUDialect::registerTypes() {
  addTypes<
#define GET_TYPEDEF_LIST
#include "Dialect/GCU/IR/Types.cpp.inc" // NOLINT: This file generated situationally via different environment variables
      >();
}

/*
Type DTEType::parse(AsmParser &odsParser) {
  Builder odsBuilder(odsParser.getContext());
  llvm::SMLoc odsLoc = odsParser.getCurrentLocation();
  (void) odsLoc;
  FailureOr<gcu::AddressSpaceAttr> addressSpace;
  // Parse literal '<'
  if (odsParser.parseLess()) return {};

  // Parse variable 'addressSpace'
  addressSpace = FieldParser<gcu::AddressSpaceAttr>::parse(odsParser);
  if (failed(addressSpace)) {
    odsParser.emitError(odsParser.getCurrentLocation(),
       "failed to parse GCU_DTEType parameter 'addressSpace' which is to be a
`::mlir::gcu::AddressSpaceAttr`"); return {};
  }
  // Parse literal '>'
  if (odsParser.parseGreater()) return {};
  assert(succeeded(addressSpace));
  return DTEType::get(odsParser.getContext(), *addressSpace);
}
*/
