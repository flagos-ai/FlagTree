/*
 * Copyright 2025-     FlagOS Contributors
 *
 * Permission is hereby granted, free of charge, to any person obtaining
 * a copy of this software and associated documentation files
 * (the "Software"), to deal in the Software without restriction,
 * including without limitation the rights to use, copy, modify, merge,
 * publish, distribute, sublicense, and/or sell copies of the Software,
 * and to permit persons to whom the Software is furnished to do so,
 * subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be
 * included in all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
 * EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
 * MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
 * IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY
 * CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,
 * TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE
 * SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
 */

#include "tle/dialect/include/Conversion/TleToLLVM/RemotePointerUtils.h"

#include "mlir/IR/BuiltinTypes.h"
#include "triton/Dialect/Triton/IR/Utility.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/Transforms/Utility.h"

#include <algorithm>

namespace mlir::triton::tle {

unsigned inferTlePointerLayoutVectorHint(Value ptr) {
  auto tensorTy = dyn_cast<RankedTensorType>(ptr.getType());
  if (!tensorTy || tensorTy.getRank() == 0)
    return 1;
  if (!tensorTy.getEncoding())
    return 1;

  auto dist =
      dyn_cast<triton::gpu::DistributedEncodingTrait>(tensorTy.getEncoding());
  if (!dist)
    return 1;
  auto linear = triton::gpu::toLinearEncoding(tensorTy);
  auto order = linear.getOrder();
  auto contigPerThread = linear.getContigPerThread();
  if (order.empty() || contigPerThread.empty())
    return 1;

  unsigned pointeeBitWidth = triton::getPointeeBitWidth(tensorTy);
  if (pointeeBitWidth == 0)
    return 1;
  unsigned maxByType = std::max<unsigned>(1, 128 / pointeeBitWidth);
  unsigned elemsPerThread = std::max<unsigned>(
      1, static_cast<unsigned>(
             triton::gpu::getTotalElemsPerThread(ptr.getType())));
  // Vector operands consume consecutive registers. Contiguity on a slower
  // axis cannot enlarge a vector on the fastest register axis (notably for
  // dot operand layouts, whose two axes can have very different widths).
  unsigned axis = order.front();
  if (axis >= contigPerThread.size())
    return 1;
  return std::max<unsigned>(
      1, std::min({maxByType, contigPerThread[axis], elemsPerThread}));
}

} // namespace mlir::triton::tle
