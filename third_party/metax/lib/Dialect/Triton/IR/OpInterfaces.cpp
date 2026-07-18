// Copyright 2026 FlagOS Contributors
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.

#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/Support/LogicalResult.h"

#include "triton/Dialect/Triton/IR/OpInterfaces.h"
#include "triton/Dialect/Triton/IR/Types.h"

namespace mlir {
namespace triton {
namespace impl {

LogicalResult verifyTransposeOpInterface(Operation *op) {
  TransposeOpInterface transposeOp = cast<TransposeOpInterface>(op);
  auto rank = cast<ShapedType>(transposeOp.getSrc().getType()).getRank();
  auto order = transposeOp.getOrder();
  if (static_cast<size_t>(rank) != order.size()) {
    return op->emitError(
        "order must have the same size as the rank of the operand and result");
  }

  SmallVector<int32_t, 8> sortedOrder(order);
  llvm::sort(sortedOrder);
  for (int32_t i = 0; i < sortedOrder.size(); i++) {
    if (sortedOrder[i] != i) {
      return op->emitError("order must be a permutation of [0, ..., rank - 1]");
    }
  }

  return success();
}

// A DotOpInterface operation should have at least three operands.
// The first two operands should share a common dimension, and the result
// should have the dimensions of the two operands that are not shared.
// A DotOpInterface operation can be either 2d or 3d.
// In the 3d case, the first dimension of operands is the batch dimension.
LogicalResult verifyDotOpInterface(Operation *op) {
  DotOpInterface dotOp = cast<mlir::triton::DotOpInterface>(op);

  if (dotOp->getNumOperands() < 3)
    return dotOp->emitOpError("expected at least 3 operands");
  auto aTy = cast<ShapedType>(dotOp->getOperand(0).getType());
  auto bTy = cast<ShapedType>(dotOp->getOperand(1).getType());
  auto cTy = cast<ShapedType>(dotOp->getOperand(2).getType());
  auto aShape = aTy.getShape();
  auto bShape = bTy.getShape();
  auto cShape = cTy.getShape();
  // Check if all 3d or all 2d
  if (aShape.size() != 2 && aShape.size() != 3)
    return dotOp->emitOpError("expected operands to be 2d or 3d");
  if (aShape.size() != bShape.size() || aShape.size() != cShape.size())
    return dotOp->emitOpError("expected all operands to have the same rank");

  // Check for valid A, B input shapes for dot
  if (!dotOp.verifyDims())
    return dotOp->emitOpError(
        "expected the last dimension of the first operand "
        "to be equal to the second-to-last dimension of "
        "the second operand");

  // Check the batch dimension
  if (aShape.size() == 3 && (aShape[0] != cShape[0] || bShape[0] != cShape[0]))
    return dotOp->emitOpError("expected the first dimension of the first "
                              "operand to be equal to the first dimension of "
                              "the result");
  // Check the output shape
  if (!dotOp.verifyOutputDims())
    return dotOp->emitOpError(
        "expected the output shape to be the concatenation of the last "
        "dimension of the first operand and the last dimension of the "
        "second ");
  return success();
}

} // namespace impl
} // namespace triton
} // namespace mlir
