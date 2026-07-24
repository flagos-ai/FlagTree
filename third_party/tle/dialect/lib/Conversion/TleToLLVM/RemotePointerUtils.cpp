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
  auto order = triton::gpu::getOrder(dist, tensorTy.getShape());
  auto contigPerThread = triton::gpu::getContigPerThread(tensorTy);
  if (order.empty() || contigPerThread.empty())
    return 1;

  unsigned pointeeBitWidth = triton::getPointeeBitWidth(tensorTy);
  if (pointeeBitWidth == 0)
    return 1;
  unsigned maxByType = std::max<unsigned>(1, 128 / pointeeBitWidth);
  unsigned elemsPerThread = std::max<unsigned>(
      1, static_cast<unsigned>(
             triton::gpu::getTotalElemsPerThread(ptr.getType())));
  unsigned best = 1;
  for (unsigned axis : order) {
    if (axis >= contigPerThread.size())
      continue;
    unsigned candidate = std::max<unsigned>(
        1,
        std::min<unsigned>(std::min<unsigned>(maxByType, contigPerThread[axis]),
                           elemsPerThread));
    best = std::max(best, candidate);
  }
  return best;
}

} // namespace mlir::triton::tle
