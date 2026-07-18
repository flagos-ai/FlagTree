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

// RUN: triton-opt %s --tritongpu-reduce-data-duplication --allocate-shared-memory --convert-triton-amdgpu-to-llvm=arch="gfx1200" -reconcile-unrealized-casts -split-input-file | FileCheck %s

#wmmaTv2 = #ttg.amd_wmma<{version = 2, warpsPerCTA = [1, 1], isTranspose = true}>
#dotop0v2 = #ttg.dot_op<{opIdx = 0, parent = #wmmaTv2, kWidth=8}>

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 1 : i32, "ttg.threads-per-warp" = 32 : i32} {
  // CHECK-LABEL: wmma_dot_cvt_bf16_wmma_v2
  tt.func @wmma_dot_cvt_bf16_wmma_v2(%arg0: tensor<16x16xbf16, #wmmaTv2>) {
    // CHECK-NOT: %0
    %0 = ttg.convert_layout %arg0 : tensor<16x16xbf16, #wmmaTv2> -> tensor<16x16xbf16, #dotop0v2>
    tt.return
  }
}
