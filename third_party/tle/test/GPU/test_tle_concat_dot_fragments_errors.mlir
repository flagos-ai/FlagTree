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

// RUN: triton-opt %s -split-input-file -convert-triton-gpu-to-llvm -verify-diagnostics

// Arity and shape are rejected by the verifier; the dot_op specific constraints
// are rejected when lowering, since operands only pick up their dot_op encoding
// after layout propagation and a #blocked tile is a legal intermediate state.

#blocked = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [8, 4], warpsPerCTA = [1, 1], order = [1, 0]}>

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 1 : i32, ttg.target = "cuda:90", "ttg.threads-per-warp" = 32 : i32} {
  tt.func @reject_too_few_tiles(%a0: tensor<64x64xf32, #blocked>) {
    // expected-error @+1 {{concat_dot_fragments requires at least two input tiles}}
    %m = tle.concat_dot_fragments %a0 {dim = 0 : i32} : tensor<64x64xf32, #blocked> -> tensor<64x64xf32, #blocked>
    tt.return
  }
}

// -----

#blocked = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [8, 4], warpsPerCTA = [1, 1], order = [1, 0]}>

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 1 : i32, ttg.target = "cuda:90", "ttg.threads-per-warp" = 32 : i32} {
  tt.func @reject_wrong_result_shape(%a0: tensor<64x64xf32, #blocked>, %a1: tensor<64x64xf32, #blocked>) {
    // Concatenating two 64x64 tiles along dim 1 must produce 64x128; claiming
    // 64x64 is a shape mismatch.
    // expected-error @+1 {{result shape mismatch at dimension 1 (expected 128, got 64)}}
    %m = tle.concat_dot_fragments %a0, %a1 {dim = 1 : i32} : tensor<64x64xf32, #blocked>, tensor<64x64xf32, #blocked> -> tensor<64x64xf32, #blocked>
    tt.return
  }
}

// -----

#blocked = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [8, 4], warpsPerCTA = [1, 1], order = [1, 0]}>

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 1 : i32, ttg.target = "cuda:90", "ttg.threads-per-warp" = 32 : i32} {
  tt.func @reject_out_of_range_dim(%a0: tensor<64x64xf32, #blocked>, %a1: tensor<64x64xf32, #blocked>) {
    // expected-error @+1 {{dim 2 out of range for rank 2}}
    %m = tle.concat_dot_fragments %a0, %a1 {dim = 2 : i32} : tensor<64x64xf32, #blocked>, tensor<64x64xf32, #blocked> -> tensor<64x128xf32, #blocked>
    tt.return
  }
}

// -----

#blocked = #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [8, 4], warpsPerCTA = [4, 1], order = [1, 0]}>

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:90", "ttg.threads-per-warp" = 32 : i32} {
  tt.func @reject_non_dot_encoding(%a0: tensor<64x32xf32, #blocked>, %a1: tensor<64x32xf32, #blocked>) {
    // expected-error @+2 {{expects dot_op encoded operands}}
    // expected-error @+1 {{failed to legalize operation 'tle.concat_dot_fragments'}}
    %m = tle.concat_dot_fragments %a0, %a1 {dim = 1 : i32} : tensor<64x32xf32, #blocked>, tensor<64x32xf32, #blocked> -> tensor<64x64xf32, #blocked>
    tt.return
  }
}

// -----

#mma = #ttg.nvidia_mma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [16, 16, 16]}>
#dot0 = #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 2}>

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:90", "ttg.threads-per-warp" = 32 : i32} {
  tt.func @reject_non_contraction_axis(%a0: tensor<128x16xbf16, #dot0>, %a1: tensor<128x16xbf16, #dot0>) {
    // dim 0 is M for opIdx 0; extending it would change which lane owns an element.
    // expected-error @+2 {{dim 0 is not the contraction axis (expected 1 for opIdx 0)}}
    // expected-error @+1 {{failed to legalize operation 'tle.concat_dot_fragments'}}
    %m = tle.concat_dot_fragments %a0, %a1 {dim = 0 : i32} : tensor<128x16xbf16, #dot0>, tensor<128x16xbf16, #dot0> -> tensor<256x16xbf16, #dot0>
    tt.return
  }
}

// -----

#mma = #ttg.nvidia_mma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [16, 16, 16]}>
#dot0 = #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 2}>

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:90", "ttg.threads-per-warp" = 32 : i32} {
  tt.func @reject_tile_below_min_k(%a0: tensor<128x1xbf16, #dot0>, %a1: tensor<128x1xbf16, #dot0>) {
    // expected-error @+2 {{tile K extent 1 is below the minimum 16 (8 * kWidth)}}
    // expected-error @+1 {{failed to legalize operation 'tle.concat_dot_fragments'}}
    %m = tle.concat_dot_fragments %a0, %a1 {dim = 1 : i32} : tensor<128x1xbf16, #dot0>, tensor<128x1xbf16, #dot0> -> tensor<128x2xbf16, #dot0>
    tt.return
  }
}

// -----

#mma = #ttg.nvidia_mma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [16, 16, 16]}>
#dot0 = #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 2}>

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:90", "ttg.threads-per-warp" = 32 : i32} {
  tt.func @reject_broadcast_along_indexed_dim(%a0: tensor<8x16xf16, #dot0>, %a1: tensor<8x16xf16, #dot0>) {
    // With M below the mma M extent the layout broadcasts, so several registers
    // hold the same element and the index-based relabel would drop some of them.
    // expected-error @+2 {{maps several registers to the same element}}
    // expected-error @+1 {{failed to legalize operation 'tle.concat_dot_fragments'}}
    %m = tle.concat_dot_fragments %a0, %a1 {dim = 1 : i32} : tensor<8x16xf16, #dot0>, tensor<8x16xf16, #dot0> -> tensor<8x32xf16, #dot0>
    tt.return
  }
}
