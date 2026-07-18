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

// RUN: triton-opt %s  -split-input-file --convert-triton-amdgpu-to-llvm="arch=gfx942" | FileCheck %s

// CHECK-LABEL:mfma_xf32

#blocked = #ttg.blocked<{sizePerThread = [8, 1], threadsPerWarp = [8, 8], warpsPerCTA = [1, 8], order = [0, 1]}>
#blocked1 = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [8, 8], warpsPerCTA = [8, 1], order = [1, 0]}>
#mma = #ttg.amd_mfma<{version = 3, warpsPerCTA = [2, 4], instrShape = [16, 16, 8], isTransposed = true}>
module attributes {"ttg.compute-capability" = 0 : i32, "ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 8 : i32, "ttg.threads-per-warp" = 64 : i32} {
  tt.func public @mfma_xf32(
    %arg0: tensor<64x128xf32, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 1}>>,
    %arg1: tensor<128x64xf32, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 1}>>) {
    %cst_0 = arith.constant dense<0.000000e+00> : tensor<64x64xf32, #mma>
    // Check that we generate xf32 instructions
    // CHECK: rocdl.mfma.f32.16x16x8.xf32
    %dot = tt.dot %arg0, %arg1, %cst_0, inputPrecision = tf32 :
      tensor<64x128xf32, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 1}>> * tensor<128x64xf32, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 1}>> -> tensor<64x64xf32, #mma>
    tt.return
  }
}

// -----

// CHECK-LABEL:mfma_not_xf32

#blocked = #ttg.blocked<{sizePerThread = [8, 1], threadsPerWarp = [8, 8], warpsPerCTA = [1, 8], order = [0, 1]}>
#blocked1 = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [8, 8], warpsPerCTA = [8, 1], order = [1, 0]}>
#mma = #ttg.amd_mfma<{version = 3, warpsPerCTA = [2, 4], instrShape = [16, 16, 4], isTransposed = true}>
module attributes {"ttg.compute-capability" = 0 : i32, "ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 8 : i32, "ttg.threads-per-warp" = 64 : i32} {
  tt.func public @mfma_not_xf32(
    %arg0: tensor<64x128xf32, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 1}>>,
    %arg1: tensor<128x64xf32, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 1}>>) {
    %cst_0 = arith.constant dense<0.000000e+00> : tensor<64x64xf32, #mma>
    // Check that we don't generate xf32 instructions if the input precision is "ieee"
    // CHECK: rocdl.mfma.f32.16x16x4f32
    %dot = tt.dot %arg0, %arg1, %cst_0, inputPrecision = ieee :
      tensor<64x128xf32, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 1}>> * tensor<128x64xf32, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 1}>> -> tensor<64x64xf32, #mma>
    tt.return
  }
}
