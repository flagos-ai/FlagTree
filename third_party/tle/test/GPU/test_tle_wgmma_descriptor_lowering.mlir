/*
* Copyright 2025-     FlagOS Contributors
*
* Permission is hereby granted, free of charge, to any person obtaining
* a copy of this software and associated documentation files
* (the "Software"), to deal in the Software without restriction,
* including without limitation the rights to use, copy, modify, merge,
* publish, distribute, sublicense, and/or sell copies of the Software,
* and to permit persons to whom the Software is furnished to do so,
* subject to the following conditions:
*
* The above copyright notice and this permission notice shall be
* included in all copies or substantial portions of the Software.
*
* THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
* EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
* MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
* IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY
* CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,
* TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE
* SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
*/

// RUN: triton-opt %s -split-input-file --allocate-shared-memory-nv='compute-capability=90 ptx-version=81' --convert-triton-gpu-to-llvm='compute-capability=90 ptx-version=81' | FileCheck %s --check-prefix=NVGPU
// RUN: triton-opt %s -split-input-file --allocate-shared-memory-nv='compute-capability=90 ptx-version=81' --convert-triton-gpu-to-llvm='compute-capability=90 ptx-version=81' --convert-nv-gpu-to-llvm | FileCheck %s --check-prefix=LLVM

#mma = #ttg.nvidia_mma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [16, 64, 16]}>
#shared_a = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 16}>
#shared_b = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = true, elementBitWidth = 16}>
#smem = #ttg.shared_memory
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:90", "ttg.threads-per-warp" = 32 : i32} {
  // NVGPU-LABEL: @descriptor_arithmetic_remains_visible_to_ptxas
  // NVGPU: nvg.wgmma
  // NVGPU-SAME: tle.wgmma_operand_a_desc_imm
  // NVGPU-SAME: tle.wgmma_operand_b_desc_imm
  // LLVM-LABEL: @descriptor_arithmetic_remains_visible_to_ptxas
  // LLVM: llvm.inline_asm
  // LLVM-SAME: add.u64 __tle_wgmma_desc_a
  // LLVM-SAME: add.u64 __tle_wgmma_desc_b
  // LLVM-SAME: wgmma.mma_async.sync.aligned
  // LLVM-SAME: __tle_wgmma_desc_a, __tle_wgmma_desc_b
  tt.func @descriptor_arithmetic_remains_visible_to_ptxas(
      %a: !ttg.memdesc<64x64xf16, #shared_a, #smem>,
      %b: !ttg.memdesc<64x64xf16, #shared_b, #smem>,
      %acc: tensor<64x64xf32, #mma>) {
    %m = ttng.warp_group_dot %a, %b, %acc { inputPrecision = 0 : i32 }:
      !ttg.memdesc<64x64xf16, #shared_a, #smem> * !ttg.memdesc<64x64xf16, #shared_b, #smem> -> tensor<64x64xf32, #mma>
    tt.return
  }
}
