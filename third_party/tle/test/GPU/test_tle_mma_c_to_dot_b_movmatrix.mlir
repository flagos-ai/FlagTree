// Copyright 2025-     FlagOS Contributors
//
// Permission is hereby granted, free of charge, to any person obtaining
// a copy of this software and associated documentation files (the "Software"),
// to deal in the Software without restriction, including without limitation
// the rights to use, copy, modify, merge, publish, distribute, sublicense,
// and/or sell copies of the Software, and to permit persons to whom the Software
// is furnished to do so, subject to the following conditions:
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

// RUN: triton-opt %s -split-input-file -allow-unregistered-dialect --allocate-shared-memory-nv='compute-capability=90 ptx-version=81' --convert-triton-gpu-to-llvm='compute-capability=90 ptx-version=81' -reconcile-unrealized-casts | FileCheck %s

#mma = #ttg.nvidia_mma<{versionMajor = 2, versionMinor = 0, warpsPerCTA = [1, 4], instrShape = [16, 8]}>
#dot_b = #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 2}>

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:90", "ttg.threads-per-warp" = 32 : i32} {
  // CHECK: module attributes {{.*}}ttg.shared = 0
  // CHECK-LABEL: llvm.func @mma_c_to_dot_b_uses_movmatrix
  // CHECK: llvm.inline_asm
  // CHECK-SAME: movmatrix.sync.aligned.m8n8.trans.b16
  // CHECK-NOT: nvvm.barrier0
  tt.func @mma_c_to_dot_b_uses_movmatrix(%src: tensor<128x128xbf16, #mma>) {
    %dst = ttg.convert_layout %src : tensor<128x128xbf16, #mma> -> tensor<128x128xbf16, #dot_b>
    "consume"(%dst) : (tensor<128x128xbf16, #dot_b>) -> ()
    tt.return
  }
}

// -----

#mma = #ttg.nvidia_mma<{versionMajor = 2, versionMinor = 0, warpsPerCTA = [1, 4], instrShape = [16, 8]}>
#dot_b = #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 2}>

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:90", "ttg.threads-per-warp" = 32 : i32} {
  // CHECK-LABEL: llvm.func @small_f16_does_not_use_movmatrix_path
  // CHECK-NOT: movmatrix.sync.aligned.m8n8.trans.b16
  tt.func @small_f16_does_not_use_movmatrix_path(%src: tensor<8x16xf16, #mma>) {
    %dst = ttg.convert_layout %src : tensor<8x16xf16, #mma> -> tensor<8x16xf16, #dot_b>
    "consume"(%dst) : (tensor<8x16xf16, #dot_b>) -> ()
    tt.return
  }
}

// -----

#mma = #ttg.nvidia_mma<{versionMajor = 2, versionMinor = 0, warpsPerCTA = [1, 4], instrShape = [16, 8]}>
#dot_b = #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 2}>

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:90", "ttg.threads-per-warp" = 32 : i32} {
  // CHECK-LABEL: llvm.func @mma_c_to_dot_b_f16_uses_movmatrix
  // CHECK: llvm.inline_asm
  // CHECK-SAME: movmatrix.sync.aligned.m8n8.trans.b16
  tt.func @mma_c_to_dot_b_f16_uses_movmatrix(%src: tensor<128x128xf16, #mma>) {
    %dst = ttg.convert_layout %src : tensor<128x128xf16, #mma> -> tensor<128x128xf16, #dot_b>
    "consume"(%dst) : (tensor<128x128xf16, #dot_b>) -> ()
    tt.return
  }
}

// -----

#mma = #ttg.nvidia_mma<{versionMajor = 2, versionMinor = 1, warpsPerCTA = [1, 4], instrShape = [16, 8]}>
#dot_b = #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 2}>

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:75", "ttg.threads-per-warp" = 32 : i32} {
  // CHECK-LABEL: llvm.func @turing_mma_c_to_dot_b_uses_movmatrix
  // CHECK: llvm.inline_asm
  // CHECK-SAME: movmatrix.sync.aligned.m8n8.trans.b16
  tt.func @turing_mma_c_to_dot_b_uses_movmatrix(%src: tensor<128x128xf16, #mma>) {
    %dst = ttg.convert_layout %src : tensor<128x128xf16, #mma> -> tensor<128x128xf16, #dot_b>
    "consume"(%dst) : (tensor<128x128xf16, #dot_b>) -> ()
    tt.return
  }
}
