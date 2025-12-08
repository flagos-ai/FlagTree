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

#include "Dialect/MemrefExt/IR/MemrefExt.h"

#define GET_OP_CLASSES
#include "Dialect/MemrefExt/IR/MemrefExtOps.cpp.inc"

namespace mlir {
namespace memref_ext {

LogicalResult MemsetStartOp::verify() {
  MemRefType dst = getDst().getType();
  if (getValue().getType() == dst.getElementType() && dst.getRank() <= 5)
    return success();
  if (dst.getRank() > 5)
    return emitOpError() << "rank should <=5 ";
  return emitOpError() << "dst and src types should has same rank, "
                          "element type and be identity memref";
}

LogicalResult SliceStartOp::verify() {
  MemRefType dst = getDst().getType();
  MemRefType src = getSrc().getType();

  auto defaultValue = getDefaultValue();
  if (defaultValue.getType().isInteger(64)) {
    auto constOp = defaultValue.getDefiningOp<arith::ConstantOp>();
    if (constOp) {
      auto value = cast<IntegerAttr>(constOp.getValue()).getInt();
      if (value != 0 && value != -1) {
        return emitOpError() << "for i64 element type the default value"
                                " can only be 0 or -1";
      }
    }
  }

  if (dst.getElementType() == src.getElementType() &&
      dst.getRank() == src.getRank() &&
      static_cast<unsigned>(dst.getRank()) == getOffsets().size() &&
      dst.getRank() <= 5)
    return success();

  if (dst.getRank() > 5)
    return emitOpError() << "rank should <=5 ";
  return emitOpError() << "dst and src types should has same rank, "
                          "element type and be identity memref";
}

} // namespace memref_ext
} // namespace mlir
