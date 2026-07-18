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

// RUN: triton-opt %s -inline | FileCheck %s

#smem = #ttg.shared_memory
#shared = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>

// CHECK-LABEL: @inline_in_warp_specialize
tt.func public @inline_in_warp_specialize(%arg0: !ttg.memdesc<1xi32, #shared, #smem, mutable>) {
  ttg.warp_specialize(%arg0)
  default {
    ttg.warp_yield
  }
  // CHECK: partition0
  partition0(%arg1: !ttg.memdesc<1xi32, #shared, #smem, mutable>) num_warps(4) {
    // CHECK-NEXT: %cst = arith.constant dense<1> : tensor<1xi32>
    // CHECK-NEXT: local_store %cst, %arg1
    tt.call @store_1(%arg1) : (!ttg.memdesc<1xi32, #shared, #smem, mutable>) -> ()
    // CHECK-NEXT: warp_return
    ttg.warp_return
  } : (!ttg.memdesc<1xi32, #shared, #smem, mutable>) -> ()
  tt.return
}

tt.func private @store_1(%arg0: !ttg.memdesc<1xi32, #shared, #smem, mutable>) {
  %cst = arith.constant dense<1> : tensor<1xi32>
  ttg.local_store %cst, %arg0 : tensor<1xi32> -> !ttg.memdesc<1xi32, #shared, #smem, mutable>
  tt.return
}
