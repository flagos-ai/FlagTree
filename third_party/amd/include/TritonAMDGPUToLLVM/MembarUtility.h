// Copyright 2026 FlagOS Contributors
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.

#ifndef TRITON_THIRD_PARTY_AMD_INCLUDE_TRITONAMDGPUTOLLVM_MEMBARUTILITY_H_
#define TRITON_THIRD_PARTY_AMD_INCLUDE_TRITONAMDGPUTOLLVM_MEMBARUTILITY_H_

#include "mlir/IR/Operation.h"

namespace mlir::triton::AMD {

// Filter function used in the AMDGPU backend to filter unnecessary barriers
// during Membar Analysis. Filters applied by this function:
// 1) Do not create barriers between AsyncCopyGlobalToLocal and LocalLoad if the
// LocalLoad is synced by AsyncWait. This prevents a redundant barrier between
// LocalLoad and prefetches because membar cannot see that subviews from the
// same shared allocation do not alias when pipelining loads. See
// amdgpu_membar.mlir for examples. This filter can produce wrong IR/assembly if
// we pipeline with a single buffer in lds because it filters out a required
// gpu.barrier between the LocalLoad and the prefetches. However the pipeliner
// will always use at least 2 buffers so this IR cannot be produced. Example
// membar input IR to produce incorrect results:
//   %tile_a = ttg.memdesc_index
//   %1 = AsyncCopyGlobalToLocal %ptr %tile_a
//   scf.for
//     %2 = AsyncWait %1
//      # Membar will add a required gpu.barrier here
//     %3 = LocalLoad %tile_a
//      # Requires gpu.barrier but filter will prevent it
//     %4 = AsyncCopyGlobalToLocal %ptr_2 %tile_a
//     scf.yield
bool membarFilter(Operation *op1, Operation *op2);
} // namespace mlir::triton::AMD

#endif
