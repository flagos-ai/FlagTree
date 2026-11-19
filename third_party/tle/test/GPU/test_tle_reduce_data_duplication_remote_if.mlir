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

// RUN: triton-opt %s -split-input-file -tritongpu-reduce-data-duplication | FileCheck %s

#blocked_local = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [32, 1], warpsPerCTA = [2, 2], order = [0, 1]}>
#blocked_remote = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [2, 16], warpsPerCTA = [4, 1], order = [1, 0]}>
#mma = #ttg.nvidia_mma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [16, 64, 16]}>
#shared = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 16}>
#smem = #ttg.shared_memory

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:90", "ttg.threads-per-warp" = 32 : i32} {
  // CHECK-LABEL: @remote_local_if_preserves_register_dot_conversion
  // CHECK-NOT: ttg.local_alloc
  // CHECK: scf.if
  // CHECK: ttg.convert_layout
  // CHECK: "tle.remote_pointers"
  // CHECK: ttg.convert_layout
  // CHECK-NOT: ttg.local_alloc
  // CHECK: ttng.warp_group_dot
  tt.func @remote_local_if_preserves_register_dot_conversion(
      %local_ptr: tensor<64x16x!tt.ptr<f16, 3>, #blocked_local>,
      %remote_base: tensor<64x16x!tt.ptr<f16, 3>, #blocked_remote>,
      %b: !ttg.memdesc<16x64xf16, #shared, #smem, mutable>,
      %acc: tensor<64x64xf32, #mma>,
      %rank: i32,
      %cond: i1) {
    %a = scf.if %cond -> (tensor<64x16xf16, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 2}>>) {
      %local = tt.load %local_ptr : tensor<64x16x!tt.ptr<f16, 3>, #blocked_local>
      %local_dot = ttg.convert_layout %local : tensor<64x16xf16, #blocked_local> -> tensor<64x16xf16, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 2}>>
      scf.yield %local_dot : tensor<64x16xf16, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 2}>>
    } else {
      %remote_ptr = "tle.remote_pointers"(%remote_base, %rank) : (tensor<64x16x!tt.ptr<f16, 3>, #blocked_remote>, i32) -> tensor<64x16x!tt.ptr<f16, 7>, #blocked_remote>
      %remote = tt.load %remote_ptr : tensor<64x16x!tt.ptr<f16, 7>, #blocked_remote>
      %remote_dot = ttg.convert_layout %remote : tensor<64x16xf16, #blocked_remote> -> tensor<64x16xf16, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 2}>>
      scf.yield %remote_dot : tensor<64x16xf16, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 2}>>
    }
    %out = ttng.warp_group_dot %a, %b, %acc {inputPrecision = 0 : i32} : tensor<64x16xf16, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 2}>> * !ttg.memdesc<16x64xf16, #shared, #smem, mutable> -> tensor<64x64xf32, #mma>
    tt.return
  }
}

// -----

#blocked_local = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [32, 1], warpsPerCTA = [2, 2], order = [0, 1]}>
#mma = #ttg.nvidia_mma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [16, 64, 16]}>

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:90", "ttg.threads-per-warp" = 32 : i32} {
  // CHECK-LABEL: @local_dot_conversion_still_uses_shared_shortcut
  // CHECK: ttg.local_alloc
  // CHECK: ttg.local_load
  tt.func @local_dot_conversion_still_uses_shared_shortcut(%value: tensor<64x16xf16, #blocked_local>) {
    %dot = ttg.convert_layout %value : tensor<64x16xf16, #blocked_local> -> tensor<64x16xf16, #ttg.dot_op<{opIdx = 0, parent = #mma, kWidth = 2}>>
    tt.return
  }
}
