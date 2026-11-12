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

#ifndef TLE_UTILITY_H
#define TLE_UTILITY_H

#include "mlir/Analysis/SliceAnalysis.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/Value.h"
#include "tle/dialect/include/IR/Dialect.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SetVector.h"

namespace mlir {
bool isSingleForLoop(scf::ForOp forOp);
bool isFromIterArg(Value operand, scf::ForOp forOp);

// Keep ordinary MMAv3 dots and TLE WGMMA on the same NVIDIA warp heuristic.
inline SmallVector<unsigned, 2>
getNvidiaMmaV3WarpsPerCTA(Value result, ArrayRef<int64_t> shape,
                          unsigned numWarps) {
  SetVector<Operation *> slice;
  getForwardSlice(result, &slice);
  if (llvm::any_of(slice, [](Operation *op) {
        return isa<triton::DotOpInterface, triton::tle::WGMMAOp>(op);
      }))
    return {numWarps, 1};

  SmallVector<unsigned, 2> warps{4, 1};
  while (warps[0] * warps[1] < numWarps) {
    if (shape[0] > 16 * warps[0])
      warps[0] *= 2;
    else
      warps[1] *= 2;
  }
  return warps;
}
} // namespace mlir

#endif
