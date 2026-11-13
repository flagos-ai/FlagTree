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

#ifndef TRITON_TLE_ANALYSIS_GATHER_LAYOUT_H
#define TRITON_TLE_ANALYSIS_GATHER_LAYOUT_H

#include "mlir/IR/Matchers.h"
#include "triton/Analysis/Utility.h"

namespace mlir::triton::tle {

/// A uniform, valid gather index selects one source coordinate per output.
/// With matching warp-local row ownership, constant folding selects the source
/// register and one lane shuffle suffices. Repacking the whole source to a
/// dynamic-index layout cannot reduce that communication and may add Shared
/// round-trips at a fixed-layout producer such as WGMMA.
inline bool hasUniformWarpLocalGather(triton::GatherOp op) {
  Attribute value;
  if (!matchPattern(op.getIndices(), m_Constant(&value)))
    return false;
  auto indices = dyn_cast<DenseIntElementsAttr>(value);
  if (!indices || !indices.isSplat())
    return false;
  auto index = indices.getSplatValue<APInt>();
  if (index.isNegative() ||
      index.uge(op.getSrc().getType().getDimSize(op.getAxis())))
    return false;
  return GatherLoweringHelper(op).isWarpLocal();
}

} // namespace mlir::triton::tle

#endif // TRITON_TLE_ANALYSIS_GATHER_LAYOUT_H
