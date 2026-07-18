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

// RUN: triton-opt %s -split-input-file --triton-nvidia-optimize-descriptor-encoding | FileCheck %s

#shared = #ttg.nvmma_shared<{swizzlingByteWidth = 64, transposed = false, elementBitWidth = 16}>
#smem = #ttg.shared_memory

module attributes {"ttg.num-warps" = 4 : i32} {
  // CHECK-LABEL: tt.func @ws_tma_store_desc_capture
  // CHECK-SAME: %{{.*}}: !tt.tensordesc<tensor<16x16xbf16, #[[DESC_ENC:[A-Za-z0-9_]+]]>>
  tt.func @ws_tma_store_desc_capture(
      %desc: !tt.tensordesc<tensor<16x16xbf16>>,
      %src: !ttg.memdesc<16x16xbf16, #shared, #smem, mutable>) {
    // CHECK: ttg.warp_specialize
    ttg.warp_specialize(%desc, %src)
    default {
      ttg.warp_yield
    }
    // CHECK: partition0(%{{.*}}: !tt.tensordesc<tensor<16x16xbf16, #[[DESC_ENC]]>>
    partition0(
        %arg0: !tt.tensordesc<tensor<16x16xbf16>>,
        %arg1: !ttg.memdesc<16x16xbf16, #shared, #smem, mutable>) num_warps(4) {
      %c0_i32 = arith.constant 0 : i32
      ttg.tma_copy %arg1, %arg0, [%c0_i32, %c0_i32]
          : !ttg.memdesc<16x16xbf16, #shared, #smem, mutable>, !tt.tensordesc<tensor<16x16xbf16>>
      ttg.warp_return
    } : (!tt.tensordesc<tensor<16x16xbf16>>, !ttg.memdesc<16x16xbf16, #shared, #smem, mutable>) -> ()
    tt.return
  }
}
