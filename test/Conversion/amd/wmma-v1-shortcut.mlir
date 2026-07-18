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

// RUN: triton-opt %s --tritongpu-reduce-data-duplication --allocate-shared-memory --convert-triton-amdgpu-to-llvm=arch="gfx1100" -split-input-file | FileCheck %s

#wmmaT = #ttg.amd_wmma<{version = 1, warpsPerCTA = [1, 1], isTranspose = true}>
#dotop0 = #ttg.dot_op<{opIdx = 0, parent = #wmmaT, kWidth=16}>

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 1 : i32, "ttg.threads-per-warp" = 32 : i32} {
  // CHECK-LABEL: wmma_dot_cvt_bf16_wmma
  tt.func public @wmma_dot_cvt_bf16_wmma(%arg0: tensor<16x16xbf16, #wmmaT>) {
    // CHECK-NOT: store
    // CHECK-NOT: load
    // CHECK-COUNT-4: rocdl.permlanex16
    // CHECK: llvm.return
    %0 = ttg.convert_layout %arg0 : tensor<16x16xbf16, #wmmaT> -> tensor<16x16xbf16, #dotop0>
    tt.return
  }
}
