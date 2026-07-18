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

// RUN: triton-opt %s | FileCheck %s


#shared = #ttg.swizzled_shared<{vec = 8, perPhase = 2, maxPhase = 8, order = [1, 0]}>
#padded = #ttg.padded_shared<[32:+4] {order = [1, 0], shape = [256, 128]}>
#smem = #ttg.shared_memory

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 8 : i32, ttg.target = "hip:gfx942", "ttg.threads-per-warp" = 64 : i32} {
  // CHECK-LABEL: memdesc_subslice_spliting
  tt.func public @memdesc_subslice_spliting() {
    %c0_i32 = arith.constant 0 : i32
    %0 = ttg.local_alloc : () -> !ttg.memdesc<1x256x128xf16, #shared, #smem, mutable>
    %1 = ttg.memdesc_index %0[%c0_i32] : !ttg.memdesc<1x256x128xf16, #shared, #smem, mutable> -> !ttg.memdesc<256x128xf16, #shared, #smem, mutable>
    %2 = ttg.memdesc_subslice %1 [0, 0]  : !ttg.memdesc<256x128xf16, #shared, #smem, mutable> -> !ttg.memdesc<128x32xf16, #shared, #smem, mutable, 256x128>
    %3 = ttg.memdesc_subslice %1 [0, 32]  : !ttg.memdesc<256x128xf16, #shared, #smem, mutable> -> !ttg.memdesc<128x32xf16, #shared, #smem, mutable, 256x128>
    %4 = ttg.memdesc_subslice %1 [0, 64]  : !ttg.memdesc<256x128xf16, #shared, #smem, mutable> -> !ttg.memdesc<128x32xf16, #shared, #smem, mutable, 256x128>
    %5 = ttg.memdesc_subslice %1 [0, 96]  : !ttg.memdesc<256x128xf16, #shared, #smem, mutable> -> !ttg.memdesc<128x32xf16, #shared, #smem, mutable, 256x128>
    %6 = ttg.memdesc_subslice %1 [128, 0]  : !ttg.memdesc<256x128xf16, #shared, #smem, mutable> -> !ttg.memdesc<128x32xf16, #shared, #smem, mutable, 256x128>
    %7 = ttg.memdesc_subslice %1 [128, 32]  : !ttg.memdesc<256x128xf16, #shared, #smem, mutable> -> !ttg.memdesc<128x32xf16, #shared, #smem, mutable, 256x128>
    %8 = ttg.memdesc_subslice %1 [128, 64]  : !ttg.memdesc<256x128xf16, #shared, #smem, mutable> -> !ttg.memdesc<128x32xf16, #shared, #smem, mutable, 256x128>
    %9 = ttg.memdesc_subslice %1 [128, 96]  : !ttg.memdesc<256x128xf16, #shared, #smem, mutable> -> !ttg.memdesc<128x32xf16, #shared, #smem, mutable, 256x128>

    %padded = ttg.local_alloc : () -> !ttg.memdesc<1x256x128xf16, #padded, #smem, mutable>
    %padded_indexed_explicit_alloc_shape = ttg.memdesc_index %padded[%c0_i32] : !ttg.memdesc<1x256x128xf16, #padded, #smem, mutable> -> !ttg.memdesc<256x128xf16, #padded, #smem, mutable>
    %10 = ttg.memdesc_subslice %padded_indexed_explicit_alloc_shape [128, 96]  : !ttg.memdesc<256x128xf16, #padded, #smem, mutable> -> !ttg.memdesc<128x32xf16, #padded, #smem, mutable, 256x128>
    %padded_indexed_implicit_alloc_shape = ttg.memdesc_index %padded[%c0_i32] : !ttg.memdesc<1x256x128xf16, #padded, #smem, mutable> -> !ttg.memdesc<256x128xf16, #padded, #smem, mutable>
    %11 = ttg.memdesc_subslice %padded_indexed_implicit_alloc_shape [128, 96]  : !ttg.memdesc<256x128xf16, #padded, #smem, mutable> -> !ttg.memdesc<128x32xf16, #padded, #smem, mutable, 256x128>
    tt.return
  }
}
