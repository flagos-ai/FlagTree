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

// RUN: triton-opt %s -split-input-file --tritongpu-accelerate-matmul | FileCheck %s

// Direct A/B chained dot should keep the one-axis warp assignment.
#blocked = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [2, 16], warpsPerCTA = [8, 1], order = [1, 0]}>
#dotOp0 = #ttg.dot_op<{opIdx = 0, parent = #blocked}>
#dotOp1 = #ttg.dot_op<{opIdx = 1, parent = #blocked}>
// CHECK: #mma = #ttg.nvidia_mma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [8, 1], instrShape = [16, 32, 16]}>
// CHECK: #mma1 = #ttg.nvidia_mma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 2], instrShape = [16, 64, 16]}>
// CHECK-LABEL: @chain_dot_through_a
// CHECK: %{{.*}} = ttng.warp_group_dot {{.*}} -> tensor<64x64xf32, #mma>
// CHECK: %{{.*}} = ttng.warp_group_dot {{.*}} -> tensor<64x128xf32, #mma1>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 8 : i32, ttg.target = "cuda:90", "ttg.threads-per-warp" = 32 : i32} {
  tt.func public @chain_dot_through_a(%q: tensor<64x128xf16, #dotOp0>,
                                      %k: tensor<128x64xf16, #dotOp1>,
                                      %v: tensor<64x128xf16, #dotOp1>) -> tensor<64x128xf32, #blocked> {
    %zero_qk = arith.constant dense<0.000000e+00> : tensor<64x64xf32, #blocked>
    %zero_o = arith.constant dense<0.000000e+00> : tensor<64x128xf32, #blocked>
    %qk = tt.dot %q, %k, %zero_qk : tensor<64x128xf16, #dotOp0> * tensor<128x64xf16, #dotOp1> -> tensor<64x64xf32, #blocked>
    %qk_f16 = arith.truncf %qk : tensor<64x64xf32, #blocked> to tensor<64x64xf16, #blocked>
    %p = ttg.convert_layout %qk_f16 : tensor<64x64xf16, #blocked> -> tensor<64x64xf16, #dotOp0>
    %o = tt.dot %p, %v, %zero_o : tensor<64x64xf16, #dotOp0> * tensor<64x128xf16, #dotOp1> -> tensor<64x128xf32, #blocked>
    tt.return %o : tensor<64x128xf32, #blocked>
  }
}

// -----

// A later dot that only uses the current dot result as C must not trigger the
// chained-dot heuristic. This should use the generic MMAv3 warp allocation.
#blocked = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [2, 16], warpsPerCTA = [8, 1], order = [1, 0]}>
#dotOp0 = #ttg.dot_op<{opIdx = 0, parent = #blocked}>
#dotOp1 = #ttg.dot_op<{opIdx = 1, parent = #blocked}>
// CHECK: #mma = #ttg.nvidia_mma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 2], instrShape = [16, 32, 16]}>
// CHECK-LABEL: @dot_result_only_as_c
// CHECK: %{{.*}} = ttng.warp_group_dot {{.*}} -> tensor<64x64xf32, #mma>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 8 : i32, ttg.target = "cuda:90", "ttg.threads-per-warp" = 32 : i32} {
  tt.func public @dot_result_only_as_c(%q: tensor<64x128xf16, #dotOp0>,
                                       %k: tensor<128x64xf16, #dotOp1>,
                                       %a: tensor<64x64xf16, #dotOp0>,
                                       %b: tensor<64x64xf16, #dotOp1>) -> tensor<64x64xf32, #blocked> {
    %zero = arith.constant dense<0.000000e+00> : tensor<64x64xf32, #blocked>
    %qk = tt.dot %q, %k, %zero : tensor<64x128xf16, #dotOp0> * tensor<128x64xf16, #dotOp1> -> tensor<64x64xf32, #blocked>
    %o = tt.dot %a, %b, %qk : tensor<64x64xf16, #dotOp0> * tensor<64x64xf16, #dotOp1> -> tensor<64x64xf32, #blocked>
    tt.return %o : tensor<64x64xf32, #blocked>
  }
}
