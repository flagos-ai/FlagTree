//===----------------------------------------------------------------------===//
//
// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.
//
//===----------------------------------------------------------------------===//


#include "TritonToLinalg/FunctionConverter.h"

namespace FunctionConverter {
using namespace mlir;
using namespace triton;

LogicalResult GetProgramIDConverter::matchAndRewrite(
    triton::GetProgramIdOp op, OpAdaptor adaptor,
    ConversionPatternRewriter &rewriter) const {
  auto axis = (uint32_t)op.getAxis();
  assert(axis < GetProgramIDConverter::LAUNCH_GRID_RANK &&
         "Invalid axis for GetProgramIdOp");
  auto func = op->getParentOfType<FunctionOpInterface>();
  auto numArgs = func.getNumArguments();
  auto id = func.getArgument(numArgs - GetProgramIDConverter::LAUNCH_GRID_RANK +
                             axis);
  rewriter.replaceOp(op, id);
  return success();
}

LogicalResult GetNumProgramsConverter::matchAndRewrite(
    triton::GetNumProgramsOp op, OpAdaptor adaptor,
    ConversionPatternRewriter &rewriter) const {
  auto axis = (uint32_t)op.getAxis();
  assert(axis < GetNumProgramsConverter::LAUNCH_GRID_RANK &&
         "Invalid axis for GetNumProgramsOp");
  auto func = op->getParentOfType<FunctionOpInterface>();
  auto numArgs = func.getNumArguments();
  auto id = func.getArgument(
      numArgs - GetNumProgramsConverter::LAUNCH_GRID_RANK * 2 + axis);
  rewriter.replaceOp(op, id);
  return success();
}
} // namespace FunctionConverter