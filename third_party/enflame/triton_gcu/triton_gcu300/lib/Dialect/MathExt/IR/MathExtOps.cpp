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

#include "mlir/Support/LogicalResult.h"

#include "Dialect/MathExt/IR/MathExt.h"

#define GET_OP_CLASSES
#include "Dialect/MathExt/IR/MathExtOps.cpp.inc"

namespace mlir {
namespace math_ext {

LogicalResult UmulhiOp::verify() {
  if (getLhs().getType() != getRhs().getType() ||
      getLhs().getType() != getResult().getType())
    return emitOpError() << "Lhs, Rhs and Result must be of the same type";
  return success();
}

LogicalResult HistogramOp::verify() {
  if (getOperand().getType().getRank() != 1)
    return emitOpError() << "histogram only supports 1D input";
  if (getResult().getType().getRank() != 1 ||
      getResult().getType().getShape()[0] == 0)
    return emitOpError()
           << "histogram only supports 1D output with a ​size greater than 0";
  return success();
}

} // namespace math_ext
} // namespace mlir
