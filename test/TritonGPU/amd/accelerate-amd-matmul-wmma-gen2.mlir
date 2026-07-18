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

// RUN: triton-opt %s -split-input-file --tritonamdgpu-accelerate-matmul="arch-generation-name=gfx1200 matrix-instruction-size=0" | FileCheck %s

// CHECK: #[[DOT_OP_PARENT:.+]] = #ttg.blocked<{{.*}}>
// CHECK: #[[WMMA_0:.+]] = #ttg.amd_wmma<{version = 2, isTranspose = true, warpsPerCTA = [1, 4]}>
#blocked = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [4, 8], warpsPerCTA = [4, 1], order = [1, 0]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 32 : i32} {
  tt.func public @wmma_dot_cf32(
   // CHECK: %[[DOT0_ARG_A:.+]]: tensor<128x64xf16, #ttg.dot_op<{opIdx = 0, parent = #[[DOT_OP_PARENT]]}>>
   %0: tensor<128x64xf16, #ttg.dot_op<{opIdx = 0, parent = #blocked}>>,
   // CHECK-SAME: %[[DOT0_ARG_B:.+]]: tensor<64x256xf16, #ttg.dot_op<{opIdx = 1, parent = #[[DOT_OP_PARENT]]}>>
   %1: tensor<64x256xf16, #ttg.dot_op<{opIdx = 1, parent = #blocked}>>,
   %2: tensor<128x256x!tt.ptr<f32>, #blocked>) {
    // CHECK: %[[DOT0_ARG_C:.+]] = arith.constant dense<0.000000e+00> : tensor<128x256xf32, #[[DOT_OP_PARENT]]>
    // CHECK: %[[DOT0_OP_C:.+]] = ttg.convert_layout %[[DOT0_ARG_C]]
    // CHECK-SAME: -> tensor<128x256xf32, #[[WMMA_0]]
    %3 = arith.constant dense<0.000000e+00> : tensor<128x256xf32, #blocked>
    // CHECK: %[[DOT0_OP_A:.+]] = ttg.convert_layout %[[DOT0_ARG_A]]
    // CHECK-SAME: -> tensor<128x64xf16, #ttg.dot_op<{opIdx = 0, parent = #[[WMMA_0]]
    // CHECK: %[[DOT0_OP_B:.+]] = ttg.convert_layout %[[DOT0_ARG_B]]
    // CHECK-SAME: -> tensor<64x256xf16, #ttg.dot_op<{opIdx = 1, parent = #[[WMMA_0]]
    // CHECK: %[[DOT0_WMMA_RES:.+]] = tt.dot %[[DOT0_OP_A]], %[[DOT0_OP_B]], %[[DOT0_OP_C]]
    // CHECK-SAME: -> tensor<128x256xf32, #[[WMMA_0]]
    %4 = tt.dot %0, %1, %3 : tensor<128x64xf16, #ttg.dot_op<{opIdx = 0, parent = #blocked}>> * tensor<64x256xf16, #ttg.dot_op<{opIdx = 1, parent = #blocked}>> -> tensor<128x256xf32, #blocked>
    // CHECK: ttg.convert_layout %[[DOT0_WMMA_RES]]
    // CHECK-SAME: -> tensor<128x256xf32, #[[DOT_OP_PARENT]]>
    tt.store %2, %4 : tensor<128x256x!tt.ptr<f32>, #blocked>
    tt.return
  }
}

// -----

// CHECK: #[[DOT_OP_PARENT:.+]] = #ttg.blocked<{{.*}}>
// CHECK: #[[WMMA_1:.+]] = #ttg.amd_wmma<{version = 2, isTranspose = true, warpsPerCTA = [2, 2]}>
#blocked = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [4, 8], warpsPerCTA = [4, 1], order = [1, 0]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 32 : i32} {
  tt.func public @wmma_dot_cf16(
   // CHECK: %[[DOT1_ARG_A:.+]]: tensor<32x64xf16, #ttg.dot_op<{opIdx = 0, parent = #[[DOT_OP_PARENT]]}>>
   %0: tensor<32x64xf16, #ttg.dot_op<{opIdx = 0, parent = #blocked}>>,
   // CHECK-SAME: %[[DOT1_ARG_B:.+]]: tensor<64x32xf16, #ttg.dot_op<{opIdx = 1, parent = #[[DOT_OP_PARENT]]}>>
   %1: tensor<64x32xf16, #ttg.dot_op<{opIdx = 1, parent = #blocked}>>,
   %2: tensor<32x32x!tt.ptr<f16>, #blocked>) {
    // CHECK: %[[DOT1_ARG_C:.+]] = arith.constant dense<0.000000e+00> : tensor<32x32xf16, #[[DOT_OP_PARENT]]>
    // CHECK: %[[DOT1_OP_C:.+]] = ttg.convert_layout %[[DOT1_ARG_C]]
    // CHECK-SAME: -> tensor<32x32xf16, #[[WMMA_1]]>
    // CHECK: %[[DOT1_OP_C_EXT:.+]] = arith.extf %[[DOT1_OP_C]]
    // CHECK-SAME: to tensor<32x32xf32, #[[WMMA_1]]>
    %3 = arith.constant dense<0.000000e+00> : tensor<32x32xf16, #blocked>
    // CHECK: %[[DOT1_OP_A:.+]] = ttg.convert_layout %[[DOT1_ARG_A]]
    // CHECK-SAME: -> tensor<32x64xf16, #ttg.dot_op<{opIdx = 0, parent = #[[WMMA_1]]
    // CHECK: %[[DOT1_OP_B:.+]] = ttg.convert_layout %[[DOT1_ARG_B]]
    // CHECK-SAME: -> tensor<64x32xf16, #ttg.dot_op<{opIdx = 1, parent = #[[WMMA_1]]
    // CHECK: %[[DOT1_WMMA_RES:.+]] = tt.dot %[[DOT1_OP_A]], %[[DOT1_OP_B]], %[[DOT1_OP_C_EXT]]
    // CHECK-SAME: -> tensor<32x32xf32, #[[WMMA_1]]
    %4 = tt.dot %0, %1, %3 : tensor<32x64xf16, #ttg.dot_op<{opIdx = 0, parent = #blocked}>> * tensor<64x32xf16, #ttg.dot_op<{opIdx = 1, parent = #blocked}>> -> tensor<32x32xf16, #blocked>
    // CHECK: %[[CONVERTED_RES:.+]] = ttg.convert_layout %[[DOT1_WMMA_RES]]
    // CHECK-SAME: -> tensor<32x32xf32, #[[DOT_OP_PARENT]]>
    // CHECK: arith.truncf %[[CONVERTED_RES]]
    // CHECK-SAME: to tensor<32x32xf16, #[[DOT_OP_PARENT]]>
    tt.store %2, %4 : tensor<32x32x!tt.ptr<f16>, #blocked>
    tt.return
  }
}

// -----

// CHECK: #[[DOT_OP_PARENT:.+]] = #ttg.blocked<{{.*}}>
// CHECK: #[[WMMA_0:.+]] = #ttg.amd_wmma<{version = 2, isTranspose = true, warpsPerCTA = [1, 4]}>
#blocked = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [4, 8], warpsPerCTA = [4, 1], order = [1, 0]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 32 : i32} {
  tt.func public @wmma_dot_ab8_cf16(
   // CHECK: %[[DOT2_ARG_A:.+]]: tensor<32x128xf8E5M2, #ttg.dot_op<{opIdx = 0, parent = #[[DOT_OP_PARENT]]}>>
   %0: tensor<32x128xf8E5M2, #ttg.dot_op<{opIdx = 0, parent = #blocked}>>,
   // CHECK-SAME: %[[DOT2_ARG_B:.+]]: tensor<128x64xf8E5M2, #ttg.dot_op<{opIdx = 1, parent = #[[DOT_OP_PARENT]]}>>
   %1: tensor<128x64xf8E5M2, #ttg.dot_op<{opIdx = 1, parent = #blocked}>>,
   %2: tensor<32x64x!tt.ptr<f16>, #blocked>) {
    // CHECK: %[[DOT2_ARG_C:.+]] = arith.constant dense<0.000000e+00> : tensor<32x64xf16, #[[DOT_OP_PARENT]]>
    // CHECK: %[[DOT2_OP_C:.+]] = ttg.convert_layout %[[DOT2_ARG_C]]
    // CHECK-SAME: -> tensor<32x64xf16, #[[WMMA_0]]>
    // CHECK: %[[DOT2_OP_C_EXT:.+]] = arith.extf %[[DOT2_OP_C]]
    // CHECK-SAME: to tensor<32x64xf32, #[[WMMA_0]]>
    %3 = arith.constant dense<0.000000e+00> : tensor<32x64xf16, #blocked>
    // CHECK: %[[DOT2_OP_A:.+]] = ttg.convert_layout %[[DOT2_ARG_A]]
    // CHECK-SAME: -> tensor<32x128xf8E5M2, #ttg.dot_op<{opIdx = 0, parent = #[[WMMA_0]]
    // CHECK: %[[DOT2_OP_B:.+]] = ttg.convert_layout %[[DOT2_ARG_B]]
    // CHECK-SAME: -> tensor<128x64xf8E5M2, #ttg.dot_op<{opIdx = 1, parent = #[[WMMA_0]]
    // CHECK: %[[DOT2_WMMA_RES:.+]] = tt.dot %[[DOT2_OP_A]], %[[DOT2_OP_B]], %[[DOT2_OP_C_EXT]]
    // CHECK-SAME: -> tensor<32x64xf32, #[[WMMA_0]]
    %4 = tt.dot %0, %1, %3 : tensor<32x128xf8E5M2, #ttg.dot_op<{opIdx = 0, parent = #blocked}>> * tensor<128x64xf8E5M2, #ttg.dot_op<{opIdx = 1, parent = #blocked}>> -> tensor<32x64xf16, #blocked>
    // CHECK: %[[CONVERTED_RES:.+]] = ttg.convert_layout %[[DOT2_WMMA_RES]]
    // CHECK-SAME: -> tensor<32x64xf32, #[[DOT_OP_PARENT]]>
    // CHECK: arith.truncf %[[CONVERTED_RES]]
    // CHECK-SAME: to tensor<32x64xf16, #[[DOT_OP_PARENT]]>
    tt.store %2, %4 : tensor<32x64x!tt.ptr<f16>, #blocked>
        tt.return
  }
}

// -----

// CHECK: #[[DOT_OP_PARENT:.+]] = #ttg.blocked<{{.*}}>
// CHECK: #[[WMMA_1:.+]] = #ttg.amd_wmma<{version = 2, isTranspose = true, warpsPerCTA = [2, 2]}>
#blocked = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [4, 8], warpsPerCTA = [4, 1], order = [1, 0]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 32 : i32} {
  tt.func public @wmma_dot_i8_i32(
   // CHECK: %[[DOT1_ARG_A:.+]]: tensor<32x64xi8, #ttg.dot_op<{opIdx = 0, parent = #[[DOT_OP_PARENT]]}>>
   %0: tensor<32x64xi8, #ttg.dot_op<{opIdx = 0, parent = #blocked}>>,
   // CHECK-SAME: %[[DOT1_ARG_B:.+]]: tensor<64x32xi8, #ttg.dot_op<{opIdx = 1, parent = #[[DOT_OP_PARENT]]}>>
   %1: tensor<64x32xi8, #ttg.dot_op<{opIdx = 1, parent = #blocked}>>,
   %2: tensor<32x32x!tt.ptr<i32>, #blocked>) {
    // CHECK: %[[DOT1_ARG_C:.+]] = arith.constant dense<0> : tensor<32x32xi32, #[[DOT_OP_PARENT]]>
    // CHECK: %[[DOT1_OP_C:.+]] = ttg.convert_layout %[[DOT1_ARG_C]]
    // CHECK-SAME: -> tensor<32x32xi32, #[[WMMA_1]]
    %3 = arith.constant dense<0> : tensor<32x32xi32, #blocked>
    // CHECK: %[[DOT1_OP_A:.+]] = ttg.convert_layout %[[DOT1_ARG_A]]
    // CHECK-SAME: -> tensor<32x64xi8, #ttg.dot_op<{opIdx = 0, parent = #[[WMMA_1]]
    // CHECK: %[[DOT1_OP_B:.+]] = ttg.convert_layout %[[DOT1_ARG_B]]
    // CHECK-SAME: -> tensor<64x32xi8, #ttg.dot_op<{opIdx = 1, parent = #[[WMMA_1]]
    // CHECK: %[[DOT1_WMMA_RES:.+]] = tt.dot %[[DOT1_OP_A]], %[[DOT1_OP_B]], %[[DOT1_OP_C]]
    // CHECK-SAME: -> tensor<32x32xi32, #[[WMMA_1]]
    %4 = tt.dot %0, %1, %3 : tensor<32x64xi8, #ttg.dot_op<{opIdx = 0, parent = #blocked}>> * tensor<64x32xi8, #ttg.dot_op<{opIdx = 1, parent = #blocked}>> -> tensor<32x32xi32, #blocked>
    // CHECK: ttg.convert_layout %[[DOT1_WMMA_RES]]
    // CHECK-SAME: -> tensor<32x32xi32, #[[DOT_OP_PARENT]]>
    tt.store %2, %4 : tensor<32x32x!tt.ptr<i32>, #blocked>
    tt.return
  }
}
