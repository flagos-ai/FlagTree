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

#ifndef KURAMA_TRITON_TO_GCU_REDUCESCANCOMMON_H
#define KURAMA_TRITON_TO_GCU_REDUCESCANCOMMON_H

#include <mlir/IR/Attributes.h>
#include <optional>
#include <utility>

#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "triton/Dialect/Triton/IR/Dialect.h"

namespace mlir {
namespace triton {
namespace gcu {

struct Fold3DResult {
  std::array<int64_t, 3> dims;
  int64_t axis;
};

Fold3DResult foldTo3D(ArrayRef<unsigned> elemsPerThread, unsigned axis);

class CombineOpDesc {
public:
  explicit CombineOpDesc(triton::ReduceOp op)
      : CombineOpDesc(op.getCombineOp(), op.getElementTypes()) {}
  explicit CombineOpDesc(triton::ScanOp op)
      : CombineOpDesc(op.getCombineOp(), op.getElementTypes()) {}
  const std::optional<vector::CombiningKind> &getCombiningKind() const {
    return combiningKind;
  }
  bool hasFastReduceLanesImpl() const { return combiningKind.has_value(); }
  SmallVector<Value> applyScalarCombine(OpBuilder &builder, Location loc,
                                        ValueRange operands) const;
  SmallVector<Value> applyVectorizedCombine(OpBuilder &builder, Location loc,
                                            ValueRange operands,
                                            unsigned vectorLength) const;
  SmallVector<Type> getElementTypes() const { return operandElementTypes; }
  unsigned getNumOperands() const { return operandElementTypes.size(); }
  FailureOr<SmallVector<TypedAttr>>
  inferIdentityAttrs(OpBuilder &builder) const;

private:
  SmallVector<Type> operandElementTypes;
  Region &combineOp;
  std::optional<vector::CombiningKind> combiningKind;

private:
  CombineOpDesc(Region &combineOp, SmallVector<Type> elementTypes)
      : operandElementTypes(std::move(elementTypes)), combineOp(combineOp),
        combiningKind(matchCombiningKind(combineOp)) {}

  static std::optional<vector::CombiningKind>
  matchCombiningKind(Region &combineOp);
};

SmallVector<Value> reduceVectorLanes(OpBuilder &builder, Location loc,
                                     const CombineOpDesc &combineOpDesc,
                                     ValueRange vecValues);

} // namespace gcu
} // namespace triton
} // namespace mlir
#endif
