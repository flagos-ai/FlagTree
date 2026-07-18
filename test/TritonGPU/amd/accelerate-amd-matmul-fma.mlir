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

// RUN: triton-opt %s -split-input-file --tritonamdgpu-accelerate-matmul="arch-generation-name=gfx942" | FileCheck %s

// CHECK: fma_dot_fp16_fp16
// CHECK: %[[D:.*]] = tt.dot {{.*}} : tensor<2x64xf16, {{.*}}> * tensor<64x64xf16, {{.*}}> -> tensor<2x64xf16, {{.*}}>
#blocked = #ttg.blocked<{sizePerThread = [4, 4], threadsPerWarp = [8, 8], warpsPerCTA = [2, 2], order = [1, 0]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "hip:gfx942", "ttg.threads-per-warp" = 64 : i32} {
  tt.func public @fma_dot_fp16_fp16(
      %arg0: tensor<2x64xf16, #ttg.dot_op<{opIdx = 0, parent = #blocked}>>,
      %arg1: tensor<64x64xf16, #ttg.dot_op<{opIdx = 1, parent = #blocked}>>,
      %arg2: tensor<2x64x!tt.ptr<f16>, #blocked> ) {
    %cst = arith.constant dense<0.0> : tensor<2x64xf16, #blocked>
    %1 = tt.dot %arg0, %arg1, %cst : tensor<2x64xf16, #ttg.dot_op<{opIdx = 0, parent = #blocked}>> * tensor<64x64xf16, #ttg.dot_op<{opIdx = 1, parent = #blocked}>> -> tensor<2x64xf16, #blocked>
    tt.store %arg2, %1 : tensor<2x64x!tt.ptr<f16>, #blocked>
    tt.return
  }
}

// -----

// CHECK: fma_dot_fp32_fp32
// CHECK: tt.dot {{.*}} : tensor<2x64xf32, {{.*}}> * tensor<64x64xf32, {{.*}}> -> tensor<2x64xf32, {{.*}}>
#blocked = #ttg.blocked<{sizePerThread = [4, 4], threadsPerWarp = [8, 8], warpsPerCTA = [2, 2], order = [1, 0]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "hip:gfx942", "ttg.threads-per-warp" = 64 : i32} {
  tt.func public @fma_dot_fp32_fp32(
      %arg0: tensor<2x64xf32, #ttg.dot_op<{opIdx = 0, parent = #blocked}>>,
      %arg1: tensor<64x64xf32, #ttg.dot_op<{opIdx = 1, parent = #blocked}>>,
      %arg2: tensor<2x64x!tt.ptr<f32>, #blocked> ) {
    %cst = arith.constant dense<0.0> : tensor<2x64xf32, #blocked>
    %1 = tt.dot %arg0, %arg1, %cst : tensor<2x64xf32, #ttg.dot_op<{opIdx = 0, parent = #blocked}>> * tensor<64x64xf32, #ttg.dot_op<{opIdx = 1, parent = #blocked}>> -> tensor<2x64xf32, #blocked>
    tt.store %arg2, %1 : tensor<2x64x!tt.ptr<f32>, #blocked>
    tt.return
  }
}

// -----

// CHECK: #[[BLOCKED:.*]] = #ttg.blocked<{sizePerThread = [4, 4], threadsPerWarp = [8, 8], warpsPerCTA = [2, 2], order = [1, 0]}>
// CHECK: fma_dot_i8
// CHECK: tt.dot {{.*}} : tensor<2x64xi8, #ttg.dot_op<{opIdx = 0, parent = #[[BLOCKED]]}>> * tensor<64x64xi8, #ttg.dot_op<{opIdx = 1, parent = #[[BLOCKED]]}>> -> tensor<2x64xi32, #[[BLOCKED]]>
#blocked = #ttg.blocked<{sizePerThread = [4, 4], threadsPerWarp = [8, 8], warpsPerCTA = [2, 2], order = [1, 0]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "hip:gfx942", "ttg.threads-per-warp" = 64 : i32} {
  tt.func public @fma_dot_i8(
      %arg0: tensor<2x64xi8, #ttg.dot_op<{opIdx = 0, parent = #blocked}>>,
      %arg1: tensor<64x64xi8, #ttg.dot_op<{opIdx = 1, parent = #blocked}>>,
      %arg2: tensor<2x64x!tt.ptr<i32>, #blocked> ) {
    %cst = arith.constant dense<0> : tensor<2x64xi32, #blocked>
    %1 = tt.dot %arg0, %arg1, %cst : tensor<2x64xi8, #ttg.dot_op<{opIdx = 0, parent = #blocked}>> * tensor<64x64xi8, #ttg.dot_op<{opIdx = 1, parent = #blocked}>> -> tensor<2x64xi32, #blocked>
    tt.store %arg2, %1 : tensor<2x64x!tt.ptr<i32>, #blocked>
    tt.return
  }
}

// -----

// CHECK: #[[BLOCKED:.*]] = #ttg.blocked<{sizePerThread = [4, 4], threadsPerWarp = [8, 8], warpsPerCTA = [2, 2], order = [1, 0]}>
// CHECK: fma_dot_f16
// CHECK: tt.dot {{.*}} : tensor<2x64xf16, #ttg.dot_op<{opIdx = 0, parent = #[[BLOCKED]]}>> * tensor<64x64xf16, #ttg.dot_op<{opIdx = 1, parent = #[[BLOCKED]]}>> -> tensor<2x64xf32, #[[BLOCKED]]>
#blocked = #ttg.blocked<{sizePerThread = [4, 4], threadsPerWarp = [8, 8], warpsPerCTA = [2, 2], order = [1, 0]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "hip:gfx942", "ttg.threads-per-warp" = 64 : i32} {
  tt.func public @fma_dot_f16(
      %arg0: tensor<2x64xf16, #ttg.dot_op<{opIdx = 0, parent = #blocked}>>,
      %arg1: tensor<64x64xf16, #ttg.dot_op<{opIdx = 1, parent = #blocked}>>,
      %arg2: tensor<2x64x!tt.ptr<f32>, #blocked> ) {
    %cst = arith.constant dense<0.0> : tensor<2x64xf32, #blocked>
    %1 = tt.dot %arg0, %arg1, %cst : tensor<2x64xf16, #ttg.dot_op<{opIdx = 0, parent = #blocked}>> * tensor<64x64xf16, #ttg.dot_op<{opIdx = 1, parent = #blocked}>> -> tensor<2x64xf32, #blocked>
    tt.store %arg2, %1 : tensor<2x64x!tt.ptr<f32>, #blocked>
    tt.return
  }
}

// -----

// CHECK: #[[BLOCKED:.*]] = #ttg.blocked<{sizePerThread = [4, 4], threadsPerWarp = [8, 8], warpsPerCTA = [2, 2], order = [1, 0]}>
// CHECK: fma_dot_f8
// CHECK: tt.dot {{.*}} : tensor<2x64xf32, #ttg.dot_op<{opIdx = 0, parent = #[[BLOCKED]]}>> * tensor<64x64xf32, #ttg.dot_op<{opIdx = 1, parent = #[[BLOCKED]]}>> -> tensor<2x64xf32, #[[BLOCKED]]>
#blocked = #ttg.blocked<{sizePerThread = [4, 4], threadsPerWarp = [8, 8], warpsPerCTA = [2, 2], order = [1, 0]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "hip:gfx942", "ttg.threads-per-warp" = 64 : i32} {
  tt.func public @fma_dot_f8(
      %arg0: tensor<2x64xf8E5M2, #ttg.dot_op<{opIdx = 0, parent = #blocked}>>,
      %arg1: tensor<64x64xf8E5M2, #ttg.dot_op<{opIdx = 1, parent = #blocked}>>,
      %arg2: tensor<2x64x!tt.ptr<f32>, #blocked> ) {
    %cst = arith.constant dense<0.0> : tensor<2x64xf32, #blocked>
    %1 = tt.dot %arg0, %arg1, %cst : tensor<2x64xf8E5M2, #ttg.dot_op<{opIdx = 0, parent = #blocked}>> * tensor<64x64xf8E5M2, #ttg.dot_op<{opIdx = 1, parent = #blocked}>> -> tensor<2x64xf32, #blocked>
    tt.store %arg2, %1 : tensor<2x64x!tt.ptr<f32>, #blocked>
    tt.return
  }
}

// -----

// CHECK: fma_dot_i8_i8
// CHECK-DAG: %[[A:.*]] = arith.sitofp
// CHECK-DAG: %[[B:.*]] = arith.sitofp
// CHECK: %[[D:.*]] = tt.dot %[[A]], %[[B]], {{.*}} : tensor<2x64xf16, {{.*}}> * tensor<64x64xf16, {{.*}}> -> tensor<2x64xf16, {{.*}}>
#blocked = #ttg.blocked<{sizePerThread = [4, 4], threadsPerWarp = [8, 8], warpsPerCTA = [2, 2], order = [1, 0]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "hip:gfx942", "ttg.threads-per-warp" = 64 : i32} {
  tt.func public @fma_dot_i8_i8(
      %arg0: tensor<2x64xi8, #ttg.dot_op<{opIdx = 0, parent = #blocked}>>,
      %arg1: tensor<64x64xi8, #ttg.dot_op<{opIdx = 1, parent = #blocked}>>,
      %arg2: tensor<2x64x!tt.ptr<i8>, #blocked> ) {
    %cst = arith.constant dense<0> : tensor<2x64xi8, #blocked>
    %1 = tt.dot %arg0, %arg1, %cst : tensor<2x64xi8, #ttg.dot_op<{opIdx = 0, parent = #blocked}>> * tensor<64x64xi8, #ttg.dot_op<{opIdx = 1, parent = #blocked}>> -> tensor<2x64xi8, #blocked>
    tt.store %arg2, %1 : tensor<2x64x!tt.ptr<i8>, #blocked>
    tt.return
  }
}
