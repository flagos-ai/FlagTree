/*
 * Copyright 2018-2020 Philippe Tillet
 * Copyright 2020-2022 OpenAI
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

#ifndef TRITON_GENERIC_SWIZZLING_H
#define TRITON_GENERIC_SWIZZLING_H

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SmallVector.h"
#include <cstdint>
#include <utility>

namespace mlir::triton {
class LinearLayout;
class TargetInfoBase;
} // namespace mlir::triton

namespace mlir::triton::gpu {
// Store the lane indices that are used in the contiguous part
// of an operation and in the address part.
// The laneAddr part just represents the indices used in one wavefront
// For now we just represent tiles with full vectorisation, meaning
// ld.shared.b32.v4/st.shared.b32.v4
// ldmatrix.v4 / stmatrix.v4
// ldmatrix.trans.v4 / stmatrix.trans.v4
struct LocalMemOpTile {
  // If laneContig.size() < log2(128/bitwidth), we assume that
  // the first log2(128/bitwidth) - laneContig.size() bases are registers
  llvm::SmallVector<int32_t> laneContig;
  // If laneAddr.size() < 3, we assume that the first
  // 3 - laneAddr.size() bases are registers
  llvm::SmallVector<int32_t> laneAddr;
};

// Given a set of possible instructions given by
// targetInfo.laneIdTiles(bitwidth) returns the optimal swizzling given these
// instructions and a pair of indices into the ldStTiles that's needed to lower
// this swizzling
std::pair<LinearLayout, std::pair<int32_t, int32_t>>
optimalSwizzling(const LinearLayout &src, const LinearLayout &dst,
                 llvm::ArrayRef<LocalMemOpTile> srcTiles,
                 llvm::ArrayRef<LocalMemOpTile> dstTiles, int32_t bitwidth);

LinearLayout optimalSwizzlingLdSt(const LinearLayout &src,
                                  const LinearLayout &dst, int32_t bitwidth);

std::pair<int, int> bankConflictsLdSt(const LinearLayout &src,
                                      const LinearLayout &dst,
                                      const LinearLayout &smem,
                                      int32_t bitwidth);

int bankConflictsMemDesc(const LinearLayout &reg, const LinearLayout &smem,
                         int32_t bitwidth);

std::pair<int, int> bankConflicts(llvm::ArrayRef<int32_t> tileSrc,
                                  llvm::ArrayRef<int32_t> tileDst,
                                  const LinearLayout &smem);
} // namespace mlir::triton::gpu

#endif // TRITON_GENERIC_SWIZZLING_H
