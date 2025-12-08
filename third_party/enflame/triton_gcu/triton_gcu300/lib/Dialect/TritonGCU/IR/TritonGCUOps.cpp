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
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/OperationSupport.h"
#include "mlir/Support/LogicalResult.h"

#include "Dialect/TritonGCU/IR/TritonGCUDialect.h"

#define GET_OP_CLASSES
#include "Dialect/TritonGCU/IR/TritonGCUOps.cpp.inc"

namespace mlir {
namespace triton {
namespace gcu {

LogicalResult LoadOp::verify() {
  if (getOffsets().size() != getShape().size() ||
      getOffsets().size() != getStrides().size() ||
      getOffsets().size() != static_cast<unsigned>(getType().getRank()))
    return emitOpError() << "shape/strides/offsets mismatch with result rank";
  if (getPtr().getType().getElementType() != getType().getElementType())
    return emitOpError() << "pointer element type mismatch";
  if (getDefaultValue() &&
      getDefaultValue().getType() != getType().getElementType())
    return emitOpError() << "default element type mismatch";
  if (getOrderHint().size() > getShape().size())
    return emitOpError() << "order_hint rank mismatch with result rank";
  return success();
}

LogicalResult StoreOp::verify() {
  if (getOffsets().size() != getShape().size() ||
      getOffsets().size() != getStrides().size() ||
      getOffsets().size() !=
          static_cast<unsigned>(getValue().getType().getRank()))
    return emitOpError() << "shape/strides/offsets mismatch with value rank";
  if (getPtr().getType().getElementType() !=
      getValue().getType().getElementType())
    return emitOpError() << "pointer element type mismatch";
  if (getOrderHint().size() > getShape().size())
    return emitOpError() << "order_hint rank mismatch with result rank";
  return success();
}

} // namespace gcu
} // namespace triton
} // namespace mlir
