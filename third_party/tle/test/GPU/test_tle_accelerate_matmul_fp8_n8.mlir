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

// Narrow FP8 dots retain native Hopper K32 accumulation; precise and
// non-FP8 controls retain their existing selection.

#blocked = #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [4, 8], warpsPerCTA = [4, 1], order = [1, 0]}>
// CHECK: #{{.*}} = #ttg.nvidia_mma<{versionMajor = 3,
// CHECK-SAME: instrShape = [16, 8, 32]
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:90", "ttg.threads-per-warp" = 32 : i32} {
  // CHECK-LABEL: tt.func @e4m3_n8
  tt.func @e4m3_n8(%a: tensor<64x128xf8E4M3FN, #blocked>, %b: tensor<128x8xf8E4M3FN, #blocked>) -> tensor<64x8xf32, #blocked> {
    %acc = arith.constant dense<0.000000e+00> : tensor<64x8xf32, #blocked>
    %ad = ttg.convert_layout %a : tensor<64x128xf8E4M3FN, #blocked> -> tensor<64x128xf8E4M3FN, #ttg.dot_op<{opIdx = 0, parent = #blocked}>>
    %bd = ttg.convert_layout %b : tensor<128x8xf8E4M3FN, #blocked> -> tensor<128x8xf8E4M3FN, #ttg.dot_op<{opIdx = 1, parent = #blocked}>>
    // CHECK: ttng.warp_group_dot
    // CHECK-SAME: f8E4M3FN
    %out = tt.dot %ad, %bd, %acc {maxNumImpreciseAcc = 1073741824 : i32} : tensor<64x128xf8E4M3FN, #ttg.dot_op<{opIdx = 0, parent = #blocked}>> * tensor<128x8xf8E4M3FN, #ttg.dot_op<{opIdx = 1, parent = #blocked}>> -> tensor<64x8xf32, #blocked>
    tt.return %out : tensor<64x8xf32, #blocked>
  }
}

// -----

#blocked = #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [4, 8], warpsPerCTA = [4, 1], order = [1, 0]}>
// CHECK: #{{.*}} = #ttg.nvidia_mma<{versionMajor = 3,
// CHECK-SAME: instrShape = [16, 8, 32]
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:90", "ttg.threads-per-warp" = 32 : i32} {
  // CHECK-LABEL: tt.func @e5m2_n8
  tt.func @e5m2_n8(%a: tensor<64x128xf8E5M2, #blocked>, %b: tensor<128x8xf8E5M2, #blocked>) -> tensor<64x8xf32, #blocked> {
    %acc = arith.constant dense<0.000000e+00> : tensor<64x8xf32, #blocked>
    %ad = ttg.convert_layout %a : tensor<64x128xf8E5M2, #blocked> -> tensor<64x128xf8E5M2, #ttg.dot_op<{opIdx = 0, parent = #blocked}>>
    %bd = ttg.convert_layout %b : tensor<128x8xf8E5M2, #blocked> -> tensor<128x8xf8E5M2, #ttg.dot_op<{opIdx = 1, parent = #blocked}>>
    // CHECK: ttng.warp_group_dot
    // CHECK-SAME: f8E5M2
    %out = tt.dot %ad, %bd, %acc {maxNumImpreciseAcc = 1073741824 : i32} : tensor<64x128xf8E5M2, #ttg.dot_op<{opIdx = 0, parent = #blocked}>> * tensor<128x8xf8E5M2, #ttg.dot_op<{opIdx = 1, parent = #blocked}>> -> tensor<64x8xf32, #blocked>
    tt.return %out : tensor<64x8xf32, #blocked>
  }
}

// -----

#blocked = #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [4, 8], warpsPerCTA = [4, 2], order = [1, 0]}>
// CHECK: #{{.*}} = #ttg.nvidia_mma<{versionMajor = 3,
// CHECK-SAME: instrShape = [16, 8, 32]
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 8 : i32, ttg.target = "cuda:90", "ttg.threads-per-warp" = 32 : i32} {
  // CHECK-LABEL: tt.func @e4m3_n8_w8
  tt.func @e4m3_n8_w8(%a: tensor<64x128xf8E4M3FN, #blocked>, %b: tensor<128x8xf8E4M3FN, #blocked>) -> tensor<64x8xf32, #blocked> {
    %acc = arith.constant dense<0.000000e+00> : tensor<64x8xf32, #blocked>
    %ad = ttg.convert_layout %a : tensor<64x128xf8E4M3FN, #blocked> -> tensor<64x128xf8E4M3FN, #ttg.dot_op<{opIdx = 0, parent = #blocked}>>
    %bd = ttg.convert_layout %b : tensor<128x8xf8E4M3FN, #blocked> -> tensor<128x8xf8E4M3FN, #ttg.dot_op<{opIdx = 1, parent = #blocked}>>
    // CHECK: ttng.warp_group_dot
    // CHECK-SAME: f8E4M3FN
    %out = tt.dot %ad, %bd, %acc {maxNumImpreciseAcc = 1073741824 : i32} : tensor<64x128xf8E4M3FN, #ttg.dot_op<{opIdx = 0, parent = #blocked}>> * tensor<128x8xf8E4M3FN, #ttg.dot_op<{opIdx = 1, parent = #blocked}>> -> tensor<64x8xf32, #blocked>
    tt.return %out : tensor<64x8xf32, #blocked>
  }
}

// -----

#blocked = #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [4, 8], warpsPerCTA = [4, 1], order = [1, 0]}>
// CHECK: #{{.*}} = #ttg.nvidia_mma<{versionMajor = 3,
// CHECK-SAME: instrShape = [16, 16, 32]
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:90", "ttg.threads-per-warp" = 32 : i32} {
  // CHECK-LABEL: tt.func @e4m3_n16
  tt.func @e4m3_n16(%a: tensor<64x128xf8E4M3FN, #blocked>, %b: tensor<128x16xf8E4M3FN, #blocked>) -> tensor<64x16xf32, #blocked> {
    %acc = arith.constant dense<0.000000e+00> : tensor<64x16xf32, #blocked>
    %ad = ttg.convert_layout %a : tensor<64x128xf8E4M3FN, #blocked> -> tensor<64x128xf8E4M3FN, #ttg.dot_op<{opIdx = 0, parent = #blocked}>>
    %bd = ttg.convert_layout %b : tensor<128x16xf8E4M3FN, #blocked> -> tensor<128x16xf8E4M3FN, #ttg.dot_op<{opIdx = 1, parent = #blocked}>>
    // CHECK: ttng.warp_group_dot
    // CHECK-SAME: f8E4M3FN
    %out = tt.dot %ad, %bd, %acc {maxNumImpreciseAcc = 1073741824 : i32} : tensor<64x128xf8E4M3FN, #ttg.dot_op<{opIdx = 0, parent = #blocked}>> * tensor<128x16xf8E4M3FN, #ttg.dot_op<{opIdx = 1, parent = #blocked}>> -> tensor<64x16xf32, #blocked>
    tt.return %out : tensor<64x16xf32, #blocked>
  }
}

// -----

#blocked = #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [4, 8], warpsPerCTA = [4, 1], order = [1, 0]}>
// CHECK: #{{.*}} = #ttg.nvidia_mma<{versionMajor = 2,
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:90", "ttg.threads-per-warp" = 32 : i32} {
  // CHECK-LABEL: tt.func @e4m3_n8_precise
  tt.func @e4m3_n8_precise(%a: tensor<64x128xf8E4M3FN, #blocked>, %b: tensor<128x8xf8E4M3FN, #blocked>) -> tensor<64x8xf32, #blocked> {
    %acc = arith.constant dense<0.000000e+00> : tensor<64x8xf32, #blocked>
    %ad = ttg.convert_layout %a : tensor<64x128xf8E4M3FN, #blocked> -> tensor<64x128xf8E4M3FN, #ttg.dot_op<{opIdx = 0, parent = #blocked}>>
    %bd = ttg.convert_layout %b : tensor<128x8xf8E4M3FN, #blocked> -> tensor<128x8xf8E4M3FN, #ttg.dot_op<{opIdx = 1, parent = #blocked}>>
    // CHECK: tt.dot
    %out = tt.dot %ad, %bd, %acc {maxNumImpreciseAcc = 0 : i32} : tensor<64x128xf8E4M3FN, #ttg.dot_op<{opIdx = 0, parent = #blocked}>> * tensor<128x8xf8E4M3FN, #ttg.dot_op<{opIdx = 1, parent = #blocked}>> -> tensor<64x8xf32, #blocked>
    tt.return %out : tensor<64x8xf32, #blocked>
  }
}

// -----

#blocked = #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [4, 8], warpsPerCTA = [4, 1], order = [1, 0]}>
// CHECK: #{{.*}} = #ttg.nvidia_mma<{versionMajor = 2,
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:90", "ttg.threads-per-warp" = 32 : i32} {
  // CHECK-LABEL: tt.func @bf16_n8
  tt.func @bf16_n8(%a: tensor<64x128xbf16, #blocked>, %b: tensor<128x8xbf16, #blocked>) -> tensor<64x8xf32, #blocked> {
    %acc = arith.constant dense<0.000000e+00> : tensor<64x8xf32, #blocked>
    %ad = ttg.convert_layout %a : tensor<64x128xbf16, #blocked> -> tensor<64x128xbf16, #ttg.dot_op<{opIdx = 0, parent = #blocked}>>
    %bd = ttg.convert_layout %b : tensor<128x8xbf16, #blocked> -> tensor<128x8xbf16, #ttg.dot_op<{opIdx = 1, parent = #blocked}>>
    // CHECK: tt.dot
    %out = tt.dot %ad, %bd, %acc {maxNumImpreciseAcc = 1073741824 : i32} : tensor<64x128xbf16, #ttg.dot_op<{opIdx = 0, parent = #blocked}>> * tensor<128x8xbf16, #ttg.dot_op<{opIdx = 1, parent = #blocked}>> -> tensor<64x8xf32, #blocked>
    tt.return %out : tensor<64x8xf32, #blocked>
  }
}

// -----

#blocked = #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [4, 8], warpsPerCTA = [4, 1], order = [1, 0]}>
// CHECK: #{{.*}} = #ttg.nvidia_mma<{versionMajor = 2,
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:90", "ttg.threads-per-warp" = 32 : i32} {
  // CHECK-LABEL: tt.func @f16_n8
  tt.func @f16_n8(%a: tensor<64x128xf16, #blocked>, %b: tensor<128x8xf16, #blocked>) -> tensor<64x8xf32, #blocked> {
    %acc = arith.constant dense<0.000000e+00> : tensor<64x8xf32, #blocked>
    %ad = ttg.convert_layout %a : tensor<64x128xf16, #blocked> -> tensor<64x128xf16, #ttg.dot_op<{opIdx = 0, parent = #blocked}>>
    %bd = ttg.convert_layout %b : tensor<128x8xf16, #blocked> -> tensor<128x8xf16, #ttg.dot_op<{opIdx = 1, parent = #blocked}>>
    // CHECK: tt.dot
    %out = tt.dot %ad, %bd, %acc {maxNumImpreciseAcc = 1073741824 : i32} : tensor<64x128xf16, #ttg.dot_op<{opIdx = 0, parent = #blocked}>> * tensor<128x8xf16, #ttg.dot_op<{opIdx = 1, parent = #blocked}>> -> tensor<64x8xf32, #blocked>
    tt.return %out : tensor<64x8xf32, #blocked>
  }
}

// -----

#blocked = #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [4, 8], warpsPerCTA = [4, 1], order = [1, 0]}>
// CHECK: #{{.*}} = #ttg.nvidia_mma<{versionMajor = 2,
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:90", "ttg.threads-per-warp" = 32 : i32} {
  // CHECK-LABEL: tt.func @e4m3_n8_short_k
  tt.func @e4m3_n8_short_k(%a: tensor<64x16xf8E4M3FN, #blocked>, %b: tensor<16x8xf8E4M3FN, #blocked>) -> tensor<64x8xf32, #blocked> {
    %acc = arith.constant dense<0.000000e+00> : tensor<64x8xf32, #blocked>
    %ad = ttg.convert_layout %a : tensor<64x16xf8E4M3FN, #blocked> -> tensor<64x16xf8E4M3FN, #ttg.dot_op<{opIdx = 0, parent = #blocked}>>
    %bd = ttg.convert_layout %b : tensor<16x8xf8E4M3FN, #blocked> -> tensor<16x8xf8E4M3FN, #ttg.dot_op<{opIdx = 1, parent = #blocked}>>
    // CHECK: tt.dot
    %out = tt.dot %ad, %bd, %acc {maxNumImpreciseAcc = 1073741824 : i32} : tensor<64x16xf8E4M3FN, #ttg.dot_op<{opIdx = 0, parent = #blocked}>> * tensor<16x8xf8E4M3FN, #ttg.dot_op<{opIdx = 1, parent = #blocked}>> -> tensor<64x8xf32, #blocked>
    tt.return %out : tensor<64x8xf32, #blocked>
  }
}
