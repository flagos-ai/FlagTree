//===----------------------------------------------------------------------===//
//
// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.
//
//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
// This file implements cumulative sum (CumSum) using the TilingInterface. Only
// supports tensors of rank 1 & 2 and axis == rank - 1 (i.e: we can split the
// computation of each row and compute them independently). The semantics of
// tiling for other axes are more complex and require working with
// non-contiguous memory.
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Affine/IR/AffineOps.h"

#include "triton-shared/Dialect/TritonTilingExt/IR/TritonTilingExtDialect.h"

#include "llvm/Support/Debug.h"

#define DEBUG_TYPE "ttx-cumsum"

using namespace mlir;
using namespace mlir::ttx;

void ttx::CumSumOp::build(OpBuilder &odsBuilder, OperationState &odsState,
                          Value input, IntegerAttr axis, Value output,
                          ArrayRef<NamedAttribute> attributes) {
  SmallVector<Value> inputs{input};
  SmallVector<Value> outputs{output};
  odsState.addOperands(inputs);
  odsState.addOperands(outputs);
  odsState.addAttribute(
      "operand_segment_sizes",
      odsBuilder.getDenseI32ArrayAttr({static_cast<int32_t>(inputs.size()),
                                       static_cast<int32_t>(outputs.size())}));

  odsState.addAttribute(getAxisAttrStrName(), axis);
  odsState.addAttributes(attributes);
  odsState.addTypes(SmallVector<Type>{output.getType()});
}

mlir::LogicalResult ttx::CumSumOp::verify() {
  auto inputType = getInput().getType();
  if (!isa<RankedTensorType>(inputType) && !isa<MemRefType>(inputType)) {
    return emitOpError(
        "CumSum op expects input to be either tensor or memref.");
  }

  auto outputType = getOutput().getType();
  if (!isa<RankedTensorType>(outputType) && !isa<MemRefType>(outputType)) {
    return emitOpError(
        "CumSum op expects output to be either tensor or memref.");
  }

  if (dyn_cast<ShapedType>(inputType).getShape() !=
      dyn_cast<ShapedType>(outputType).getShape()) {
    return emitOpError("Input and output types must be the same.");
  }

  int64_t rank = getRank();
  if (rank != 1 && rank != 2) {
    return emitOpError("CumSum op only takes tensors of rank 1 & 2.");
  }

  int64_t axis = getAxis();
  if (axis != rank - 1) {
    return emitOpError("CumSum computation only supports axis == rank - 1");
  }

  return success();
}

AffineMap ttx::CumSumOp::getInputIndexingMap(MLIRContext *context,
                                             unsigned int index,
                                             ArrayRef<OpFoldResult> sizes) {
  assert(index == 0);
  return AffineMap::getMultiDimIdentityMap(getRank(), context);
}

AffineMap ttx::CumSumOp::getOutputIndexingMap(MLIRContext *context,
                                              unsigned int index,
                                              ArrayRef<OpFoldResult> sizes) {
  assert(index == 0);
  return AffineMap::getMultiDimIdentityMap(getRank(), context);
}

SmallVector<utils::IteratorType> ttx::CumSumOp::getLoopIteratorTypes() {
  SmallVector<utils::IteratorType> iterators;
  iterators.append(getRank() - 1, utils::IteratorType::parallel);
  iterators.push_back(utils::IteratorType::reduction);
  return iterators;
}

SmallVector<Range> ttx::CumSumOp::getIterationDomain(OpBuilder &b) {
  OpBuilder::InsertionGuard g(b);
  b.setInsertionPoint(*this);
  auto loc = getLoc();
  auto zero = b.getIndexAttr(0);
  auto one = b.getIndexAttr(1);
  SmallVector<Range> iterationDomain;

  // Return the bounds for all dimensions. The caller is responsible for not
  // tiling the inner most dimension, otherwise the semantic of the resulting op
  // is incorrect.
  for (auto i = 0; i < getRank(); i++) {
    OpFoldResult upperbound = linalg::createFoldedDimOp(b, loc, getInput(), i);
    iterationDomain.push_back(Range{zero, upperbound, one});
  }
  return iterationDomain;
}
