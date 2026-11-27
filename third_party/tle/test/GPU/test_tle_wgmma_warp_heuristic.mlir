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

// RUN: triton-opt %s -split-input-file -triton-tle-lower-wgmma | FileCheck %s

#blocked = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [4, 2], order = [1, 0]}>
#shared = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 16}>
#smem = #ttg.shared_memory

// CHECK: #[[$MMA:.+]] = #ttg.nvidia_mma<{{.*}}warpsPerCTA = [4, 2], instrShape = [16, 32, 16]
module attributes {"ttg.target" = "cuda:90", "ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 8 : i32, "ttg.threads-per-warp" = 32 : i32} {
  // CHECK-LABEL: tt.func @unchained_wgmma_uses_n_split
  tt.func @unchained_wgmma_uses_n_split(
      %a: !ttg.memdesc<64x64xf16, #shared, #smem, mutable>,
      %b: !ttg.memdesc<64x64xf16, #shared, #smem, mutable>) {
    %zero = arith.constant dense<0.000000e+00> : tensor<64x64xf32, #blocked>
    // CHECK: ttng.warp_group_dot {{.*}} -> tensor<64x64xf32, #[[$MMA]]>
    %dot = tle.wgmma %a, %b, %zero {inputPrecision = 0 : i32, isAsync = true, maxNumImpreciseAcc = 0 : i32} : !ttg.memdesc<64x64xf16, #shared, #smem, mutable> * !ttg.memdesc<64x64xf16, #shared, #smem, mutable>, tensor<64x64xf32, #blocked> -> tensor<64x64xf32, #blocked>
    %wait = tle.wgmma_wait %dot {pendings = 0 : i32} : tensor<64x64xf32, #blocked> -> tensor<64x64xf32, #blocked>
    tt.return
  }
}

// -----

#blocked = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [4, 2], order = [1, 0]}>
#carried = #ttg.nvidia_mma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [8, 1], instrShape = [16, 16, 32]}>
#shared = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = true, elementBitWidth = 16}>
#smem = #ttg.shared_memory

// CHECK-DAG: #[[$C:.+]] = #ttg.nvidia_mma<{{.*}}warpsPerCTA = [8, 1], instrShape = [16, 16, 32]
// CHECK-DAG: #[[$A:.+]] = #ttg.nvidia_mma<{{.*}}warpsPerCTA = [8, 1], instrShape = [16, 16, 16]
module attributes {"ttg.target" = "cuda:90", "ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 8 : i32, "ttg.threads-per-warp" = 32 : i32} {
  // CHECK-LABEL: tt.func @carried_c_preserves_mn_but_not_k
  // CHECK-SAME: (%[[ARG_A:[^:]+]]:
  // CHECK-SAME: %[[ARG_B:[^:]+]]:
  // CHECK-SAME: %[[ARG_C:[^:]+]]:
  tt.func @carried_c_preserves_mn_but_not_k(
      %a: tensor<64x64xbf16, #blocked>,
      %b: !ttg.memdesc<64x64xbf16, #shared, #smem, mutable>,
      %acc: tensor<64x64xf32, #carried>) {
    // CHECK: %[[REG_A:.+]] = ttg.convert_layout %[[ARG_A]] : {{.*}} -> tensor<64x64xbf16, #ttg.dot_op<{opIdx = 0, parent = #[[$A]], kWidth = 2}>>
    // CHECK-NEXT: ttng.warp_group_dot %[[REG_A]], %[[ARG_B]], %[[ARG_C]] {{.*}} -> tensor<64x64xf32, #[[$C]]>
    %dot = tle.wgmma %a, %b, %acc {inputPrecision = 0 : i32, isAsync = true, maxNumImpreciseAcc = 0 : i32} : tensor<64x64xbf16, #blocked> * !ttg.memdesc<64x64xbf16, #shared, #smem, mutable>, tensor<64x64xf32, #carried> -> tensor<64x64xf32, #carried>
    %wait = tle.wgmma_wait %dot {pendings = 0 : i32} : tensor<64x64xf32, #carried> -> tensor<64x64xf32, #carried>
    tt.return
  }
}
