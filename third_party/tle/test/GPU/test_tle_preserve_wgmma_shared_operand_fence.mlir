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

// RUN: triton-opt %s -triton-nvidia-gpu-fence-insertion=compute-capability=90 --canonicalize | FileCheck %s

#blocked = #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [1, 32], warpsPerCTA = [4, 1], order = [1, 0]}>
#mma = #ttg.nvidia_mma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [16, 64, 16]}>
#dot = #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 2}>
#shared_in = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 16}>
#shared_out = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = true, elementBitWidth = 16}>
#smem = #ttg.shared_memory

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:90", "ttg.threads-per-warp" = 32 : i32} {
  // CHECK-LABEL: tt.func @preserve_async_copy_wgmma_fence
  tt.func @preserve_async_copy_wgmma_fence(
      %a: tensor<64x64xbf16, #dot>,
      %src: tensor<64x64x!tt.ptr<bf16>, #blocked>) -> tensor<64x64xf32, #mma> {
    %c0 = arith.constant 0 : i32
    %acc = arith.constant dense<0.000000e+00> : tensor<64x64xf32, #mma>
    %base = ttg.local_alloc : () -> !ttg.memdesc<2x64x64xbf16, #shared_in, #smem, mutable>
    %slot = ttg.memdesc_index %base[%c0] : !ttg.memdesc<2x64x64xbf16, #shared_in, #smem, mutable> -> !ttg.memdesc<64x64xbf16, #shared_in, #smem, mutable>
    %tok = ttg.async_copy_global_to_local %src, %slot : tensor<64x64x!tt.ptr<bf16>, #blocked> -> <64x64xbf16, #shared_in, #smem, mutable>
    %tok2 = ttg.async_commit_group tokens %tok
    %view = tle.memdesc_wgmma_view %slot {order = array<i32: 1, 0>} : !ttg.memdesc<64x64xbf16, #shared_in, #smem, mutable> -> !ttg.memdesc<64x64xbf16, #shared_out, #smem, mutable>
    // CHECK: tle.wgmma_shared_operand_fence
    // CHECK-NEXT: ttng.warp_group_dot
    %out = ttng.warp_group_dot %a, %view, %acc {inputPrecision = 0 : i32} : tensor<64x64xbf16, #dot> * !ttg.memdesc<64x64xbf16, #shared_out, #smem, mutable> -> tensor<64x64xf32, #mma>
    tt.return %out : tensor<64x64xf32, #mma>
  }
}
