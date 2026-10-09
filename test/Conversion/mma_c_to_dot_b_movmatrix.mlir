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

// RUN: split-file %s %t
// RUN: triton-opt %t/sm90.mlir -split-input-file -allow-unregistered-dialect --allocate-shared-memory-nv='compute-capability=90 ptx-version=81' --convert-triton-gpu-to-llvm='compute-capability=90 ptx-version=81' -reconcile-unrealized-casts | FileCheck %t/sm90.mlir --check-prefix=SM90
// RUN: triton-opt %t/sm75.mlir -allow-unregistered-dialect --allocate-shared-memory-nv='compute-capability=75 ptx-version=78' --convert-triton-gpu-to-llvm='compute-capability=75 ptx-version=78' -reconcile-unrealized-casts | FileCheck %t/sm75.mlir --check-prefix=SM75
// RUN: triton-opt %t/sm75.mlir -allow-unregistered-dialect --allocate-shared-memory-nv='compute-capability=75 ptx-version=77' --convert-triton-gpu-to-llvm='compute-capability=75 ptx-version=77' -reconcile-unrealized-casts | FileCheck %t/sm75.mlir --check-prefix=FALLBACK
// RUN: sed 's/cuda:75/cuda:70/' %t/sm75.mlir > %t/sm70.mlir
// RUN: triton-opt %t/sm70.mlir -allow-unregistered-dialect --allocate-shared-memory-nv='compute-capability=70 ptx-version=78' --convert-triton-gpu-to-llvm='compute-capability=70 ptx-version=78' -reconcile-unrealized-casts | FileCheck %t/sm70.mlir --check-prefix=FALLBACK

// movmatrix requires SM75 and PTX 7.8. Check both lower bounds independently.

//--- sm90.mlir

#mma = #ttg.nvidia_mma<{versionMajor = 2, versionMinor = 0, warpsPerCTA = [1, 4], instrShape = [16, 8]}>
#dot_b = #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 2}>

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:90", "ttg.threads-per-warp" = 32 : i32} {
  // SM90: module attributes {{.*}}ttg.shared = 0
  // SM90-LABEL: llvm.func @mma_c_to_dot_b_uses_movmatrix
  // SM90: llvm.inline_asm
  // SM90-SAME: movmatrix.sync.aligned.m8n8.trans.b16
  // SM90-NOT: nvvm.barrier0
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
  // SM90-LABEL: llvm.func @small_f16_does_not_use_movmatrix_path
  // SM90-NOT: movmatrix.sync.aligned.m8n8.trans.b16
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
  // SM90-LABEL: llvm.func @mma_c_to_dot_b_f16_uses_movmatrix
  // SM90: llvm.inline_asm
  // SM90-SAME: movmatrix.sync.aligned.m8n8.trans.b16
  tt.func @mma_c_to_dot_b_f16_uses_movmatrix(%src: tensor<128x128xf16, #mma>) {
    %dst = ttg.convert_layout %src : tensor<128x128xf16, #mma> -> tensor<128x128xf16, #dot_b>
    "consume"(%dst) : (tensor<128x128xf16, #dot_b>) -> ()
    tt.return
  }
}

// -----

#mma = #ttg.nvidia_mma<{versionMajor = 2, versionMinor = 0, warpsPerCTA = [2, 2], instrShape = [16, 8]}>
#dot_b = #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 2}>

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:90", "ttg.threads-per-warp" = 32 : i32} {
  // A row warp replicated by the small shape still permits a warp-local move.
  // SM90-LABEL: llvm.func @replicated_row_warp_uses_movmatrix
  // SM90: llvm.inline_asm
  // SM90-SAME: movmatrix.sync.aligned.m8n8.trans.b16
  tt.func @replicated_row_warp_uses_movmatrix(%src: tensor<16x32xbf16, #mma>) {
    %dst = ttg.convert_layout %src : tensor<16x32xbf16, #mma> -> tensor<16x32xbf16, #dot_b>
    "consume"(%dst) : (tensor<16x32xbf16, #dot_b>) -> ()
    tt.return
  }
}

// -----

#mma = #ttg.nvidia_mma<{versionMajor = 2, versionMinor = 0, warpsPerCTA = [2, 2], instrShape = [16, 8]}>
#dot_b = #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 2}>

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:90", "ttg.threads-per-warp" = 32 : i32} {
  // The same warp layout requires cross-warp communication for a larger tile.
  // SM90-LABEL: llvm.func @cross_warp_conversion_does_not_use_movmatrix
  // SM90-NOT: movmatrix.sync.aligned.m8n8.trans.b16
  tt.func @cross_warp_conversion_does_not_use_movmatrix(%src: tensor<32x64xbf16, #mma>) {
    %dst = ttg.convert_layout %src : tensor<32x64xbf16, #mma> -> tensor<32x64xbf16, #dot_b>
    "consume"(%dst) : (tensor<32x64xbf16, #dot_b>) -> ()
    tt.return
  }
}

//--- sm75.mlir

#mma = #ttg.nvidia_mma<{versionMajor = 2, versionMinor = 1, warpsPerCTA = [1, 4], instrShape = [16, 8]}>
#dot_b = #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 2}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:75", "ttg.threads-per-warp" = 32 : i32} {
  // FALLBACK-LABEL: llvm.func @turing_mma_c_to_dot_b_uses_movmatrix
  // FALLBACK-NOT: movmatrix.sync.aligned.m8n8.trans.b16
  // SM75-LABEL: llvm.func @turing_mma_c_to_dot_b_uses_movmatrix
  // SM75: llvm.inline_asm
  // SM75-SAME: movmatrix.sync.aligned.m8n8.trans.b16
  tt.func @turing_mma_c_to_dot_b_uses_movmatrix(%src: tensor<128x128xf16, #mma>) {
    %dst = ttg.convert_layout %src : tensor<128x128xf16, #mma> -> tensor<128x128xf16, #dot_b>
    "consume"(%dst) : (tensor<128x128xf16, #dot_b>) -> ()
    tt.return
  }
}
