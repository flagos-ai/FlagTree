// Copyright 2025-     FlagOS Contributors
//
// Permission is hereby granted, free of charge, to any person obtaining
// a copy of this software and associated documentation files
// (the "Software"), to deal in the Software without restriction,
// including without limitation the rights to use, copy, modify, merge,
// publish, distribute, sublicense, and/or sell copies of the Software,
// and to permit persons to whom the Software is furnished to do so,
// subject to the following conditions:
//
// The above copyright notice and this permission notice shall be
// included in all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
// EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
// MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
// IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY
// CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,
// TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE
// SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

// RUN: env FLAGTREE_SAME_WARP_LAYOUT_SHUFFLE=1 triton-opt %s -split-input-file --allocate-shared-memory-nv --convert-triton-gpu-to-llvm -reconcile-unrealized-casts 2>/dev/null | FileCheck %s --check-prefixes=COMMON,ON
// RUN: env FLAGTREE_SAME_WARP_LAYOUT_SHUFFLE=0 triton-opt %s -split-input-file --allocate-shared-memory-nv --convert-triton-gpu-to-llvm -reconcile-unrealized-casts 2>/dev/null | FileCheck %s --check-prefixes=COMMON,OFF

// The warp-shuffle lowering of layout conversions that read broadcast copies
// in other warps (__FLAGTREE_SAME_WARP_LAYOUT_SHUFFLE__, NVIDIA build, from
// https://github.com/triton-lang/triton/pull/11646) is switched off with
// FLAGTREE_SAME_WARP_LAYOUT_SHUFFLE=0: such a conversion then goes through
// shared memory, as before the backport. A conversion within the lanes of a
// warp is shuffled either way.

#broadcast_src = #ttg.linear<{register=[[0, 1], [0, 2]], lane=[[1, 0], [2, 0], [4, 0], [8, 0], [16, 0]], warp=[[32, 0], [64, 0], [0, 0]], block=[]}>
#broadcast_dst = #ttg.linear<{register=[[0, 1], [0, 2], [8, 0]], lane=[[0, 0], [0, 0], [1, 0], [2, 0], [4, 0]], warp=[[32, 0], [64, 0], [16, 0]], block=[]}>
// ON: ttg.shared = 0 : i32
// OFF: ttg.shared = 1024 : i32
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 8 : i32} {
  // The source repeats its rows in warp ^ 4: the destination lanes read them
  // from the copy in their own warp.
  // COMMON-LABEL: @convert_layout_broadcast_warp
  tt.func @convert_layout_broadcast_warp(%arg0: tensor<128x4xf16, #broadcast_src>) {
    // ON-NOT: nvvm.barrier
    // ON-COUNT-4: nvvm.shfl.sync
    // ON-NOT: nvvm.shfl.sync
    // ON-NOT: nvvm.barrier
    // OFF-NOT: nvvm.shfl.sync
    // OFF: nvvm.barrier
    // COMMON: llvm.return
    %0 = ttg.convert_layout %arg0 : tensor<128x4xf16, #broadcast_src> -> tensor<128x4xf16, #broadcast_dst>
    tt.return
  }
}

// -----

#blocked0 = #ttg.blocked<{sizePerThread = [32, 1], threadsPerWarp = [1, 32], warpsPerCTA = [1, 1], order = [0, 1]}>
#blocked1 = #ttg.blocked<{sizePerThread = [16, 2], threadsPerWarp = [2, 16], warpsPerCTA = [1, 1], order = [0, 1]}>
// COMMON: ttg.shared = 0 : i32
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 1 : i32} {
  // COMMON-LABEL: @convert_layout_within_warp
  tt.func @convert_layout_within_warp(%arg0: tensor<32x32xi32, #blocked0>) {
    // COMMON-NOT: nvvm.barrier
    // COMMON-COUNT-16: nvvm.shfl.sync
    // COMMON-NOT: nvvm.barrier
    // COMMON: llvm.return
    %0 = ttg.convert_layout %arg0 : tensor<32x32xi32, #blocked0> -> tensor<32x32xi32, #blocked1>
    tt.return
  }
}
