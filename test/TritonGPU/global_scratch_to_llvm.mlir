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

// RUN: triton-opt %s -allow-unregistered-dialect --tritongpu-global-scratch-memory-allocation --convert-triton-gpu-to-llvm | FileCheck %s

module attributes {"ttg.num-warps" = 4 : i32} {
  // CHECK-LABEL: @global_scratch_alloc_warpgroup(%arg0: !llvm.ptr<1>, %arg1: !llvm.ptr<1>)
  tt.func @global_scratch_alloc_warpgroup() {
    // CHECK-NEXT: ttg.warp_specialize(%arg0)
    ttg.warp_specialize()
    default {
      ttg.warp_yield
    }
    // CHECK: partition0(%arg2: !llvm.ptr<1>)
    partition0() num_warps(1) {
      // CHECK-COUNT-2: llvm.getelementptr %arg2
      %0 = ttg.global_scratch_alloc {alignment = 8 : i32, nbytes = 100 : i32, ttg.global_scratch_memory_offset = 0 : i32} : !tt.ptr<i8>
      %1 = ttg.global_scratch_alloc {alignment = 8 : i32, nbytes = 100 : i32, ttg.global_scratch_memory_offset = 0 : i32} : !tt.ptr<i8>
      "use"(%0, %1) : (!tt.ptr<i8>, !tt.ptr<i8>) -> ()
      ttg.warp_return
    } : () -> ()
    tt.return
  }
}
