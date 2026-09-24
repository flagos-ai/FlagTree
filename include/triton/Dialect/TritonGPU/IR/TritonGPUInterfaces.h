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

#ifndef TRITON_GPU_DIALECT_INTERFACES_H
#define TRITON_GPU_DIALECT_INTERFACES_H

#include "mlir/IR/OpDefinition.h"
#include "triton/Dialect/TritonGPU/IR/CTAEncodingAttr.h"
#include <optional>

// TLE is also built with older backend IR overlays. They keep their existing
// extraction behavior until they provide this interface and its layout proof.
#define TRITON_GPU_WARP_SLICE_INTERFACE

namespace mlir::triton::gpu {
// A per-thread register view held by one consecutive, aligned range of warps.
// registers[i] is the root-source register containing destination register i;
// lane and block IDs are unchanged, and the destination warp ID is relative
// to startWarp. No communication between threads is needed to create the
// view. For a nested view, localRegisters keeps the mapping relative to the
// immediate source value so LLVM lowering can consume the already-packed
// parent result without reinterpreting it as the root register tuple.
struct WarpSlice {
  unsigned startWarp;
  unsigned numWarps;
  SmallVector<unsigned> registers;
  SmallVector<unsigned> localRegisters;
};

// Infer a register-only view of the rectangular tile at offsets in src.
// Shapes must be powers of two and offsets must be aligned to dst's shape.
// Returns nullopt if the layouts require communication, do not select a unique
// consecutive warp range, or cannot be represented by one per-thread register
// mapping. In particular, this does not select an arbitrary owner when the
// source replicates the tile across multiple possible warp ranges.
std::optional<WarpSlice> inferWarpSlice(RankedTensorType src,
                                        RankedTensorType dst,
                                        ArrayRef<int64_t> offsets);

// Compose a child view mapping with the mapping of its immediate source.
// `child.registers` are indexed in the parent's result register tuple, while
// the returned `registers` are expressed in the original root source
// registers and `localRegisters` retains the child mapping. The physical warp
// ranges must be nested; otherwise the two views cannot be represented as one
// register-local slice.
std::optional<WarpSlice> composeWarpSlices(const WarpSlice &parent,
                                           const WarpSlice &child);

} // namespace mlir::triton::gpu

// clang-format off
#include "triton/Dialect/TritonGPU/IR/LinearLayoutConversions.h"
#include "triton/Dialect/TritonGPU/IR/AttrInterfaces.h.inc"
#include "triton/Dialect/TritonGPU/IR/OpInterfaces.h.inc"
// clang-format on

#endif // TRITON_GPU_DIALECT_INTERFACES_H
