/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
 * THE SOFTWARE.
 */

#include "ascend/include/TritonToLinalg/FunctionConverter.h"
#include "ascend/include/Utils/DebugUtils.h"

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

  Location pidLoc = op.getLoc();
  insertDebugNop(pidLoc, rewriter);
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

  Location numProgsLoc = op.getLoc();
  insertDebugNop(numProgsLoc, rewriter);
  rewriter.replaceOp(op, id);
  return success();
}
} // namespace FunctionConverter
