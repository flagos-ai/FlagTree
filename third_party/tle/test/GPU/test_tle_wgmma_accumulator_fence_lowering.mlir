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

// RUN: triton-opt %s -split-input-file --allocate-shared-memory-nv='compute-capability=90 ptx-version=81' --convert-triton-gpu-to-llvm='compute-capability=90 ptx-version=81' | FileCheck %s
// RUN: triton-opt %s -split-input-file --allocate-shared-memory-nv='compute-capability=90 ptx-version=81' --convert-triton-gpu-to-llvm='compute-capability=90 ptx-version=81' --convert-nv-gpu-to-llvm | FileCheck %s --check-prefix=LLVM

#mma = #ttg.nvidia_mma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [16, 64, 16]}>
#shared = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = true, elementBitWidth = 16}>
#smem = #ttg.shared_memory
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32} {
  // CHECK-LABEL: @dot_nonzero_acc_precedes_wgmma_fence
  // CHECK: llvm.insertvalue
  // CHECK: nvvm.wgmma.fence.aligned
  // CHECK-NOT: llvm.insertvalue
  // CHECK: nvg.wgmma
  // LLVM-LABEL: @dot_nonzero_acc_precedes_wgmma_fence
  // LLVM: "wgmma.fence.sync.aligned;
  // LLVM-SAME: add.u64 __tle_wgmma_desc_a
  // LLVM-SAME: add.u64 __tle_wgmma_desc_b
  // LLVM-SAME: wgmma.mma_async.sync.aligned
  tt.func @dot_nonzero_acc_precedes_wgmma_fence(%a: !ttg.memdesc<64x64xf16, #shared, #smem>, %b: !ttg.memdesc<64x64xf16, #shared, #smem>, %acc: tensor<64x64xf32, #mma>) {
    %m = ttng.warp_group_dot %a, %b, %acc { inputPrecision = 0 : i32 }:
      !ttg.memdesc<64x64xf16, #shared, #smem> * !ttg.memdesc<64x64xf16, #shared, #smem> -> tensor<64x64xf32, #mma>
    tt.return
  }
}

// -----

#mma = #ttg.nvidia_mma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [16, 64, 16]}>
#shared = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = true, elementBitWidth = 16}>
#smem = #ttg.shared_memory
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32} {
  // CHECK-LABEL: @dot_chain_c_consumer_omits_wgmma_fence
  // CHECK: nvvm.wgmma.fence.aligned
  // CHECK-NOT: nvvm.wgmma.fence.aligned
  // CHECK: llvm.return
  // LLVM-LABEL: @dot_chain_c_consumer_omits_wgmma_fence
  // LLVM: "wgmma.fence.sync.aligned;
  // LLVM-NOT: "wgmma.fence.sync.aligned;
  // LLVM: llvm.return
  tt.func @dot_chain_c_consumer_omits_wgmma_fence(
      %a0: !ttg.memdesc<64x64xf16, #shared, #smem>,
      %b0: !ttg.memdesc<64x64xf16, #shared, #smem>,
      %a1: !ttg.memdesc<64x64xf16, #shared, #smem>,
      %b1: !ttg.memdesc<64x64xf16, #shared, #smem>,
      %acc: tensor<64x64xf32, #mma>) {
    %m0 = ttng.warp_group_dot %a0, %b0, %acc {inputPrecision = 0 : i32, isAsync = true, tle.explicit_wgmma_commit} :
      !ttg.memdesc<64x64xf16, #shared, #smem> * !ttg.memdesc<64x64xf16, #shared, #smem> -> tensor<64x64xf32, #mma>
    %m1 = ttng.warp_group_dot %a1, %b1, %m0 {inputPrecision = 0 : i32, isAsync = true, tle.explicit_wgmma_commit, tle.wgmma_accumulator_chain_c} :
      !ttg.memdesc<64x64xf16, #shared, #smem> * !ttg.memdesc<64x64xf16, #shared, #smem> -> tensor<64x64xf32, #mma>
    tt.return
  }
}
