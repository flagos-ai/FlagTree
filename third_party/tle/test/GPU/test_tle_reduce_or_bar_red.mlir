// MIT License
//
// Copyright (c) 2025 The FlagOS Contributors
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
//
// flagtree tle

// RUN: triton-opt %s -split-input-file --allocate-shared-memory-nv | FileCheck %s --check-prefix=ALLOC
// RUN: triton-opt %s -split-input-file --allocate-shared-memory-nv --convert-triton-gpu-to-llvm -reconcile-unrealized-casts | FileCheck %s

#blocked = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>
module attributes {"ttg.target" = "cuda:90", "ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 32 : i32} {
  // ALLOC-LABEL: tt.func @reduce_or_bar_red
  // ALLOC-NOT: allocation.offset
  // CHECK-LABEL: llvm.func @reduce_or_bar_red
  // CHECK: llvm.inline_asm
  // CHECK-SAME: bar.red.or.pred
  tt.func @reduce_or_bar_red(%arg0: tensor<128xi1, #blocked>, %out_ptr: !tt.ptr<i1>) {
    %0 = "tt.reduce"(%arg0) <{axis = 0 : i32}> ({
    ^bb0(%lhs: i1, %rhs: i1):
      %1 = arith.ori %lhs, %rhs : i1
      tt.reduce.return %1 : i1
    }) : (tensor<128xi1, #blocked>) -> i1
    %2 = tt.make_range {end = 128 : i32, start = 0 : i32} : tensor<128xi32, #blocked>
    %3 = arith.constant dense<0> : tensor<128xi32, #blocked>
    %4 = arith.cmpi eq, %2, %3 : tensor<128xi32, #blocked>
    %5 = tt.splat %out_ptr : !tt.ptr<i1> -> tensor<128x!tt.ptr<i1>, #blocked>
    %6 = tt.splat %0 : i1 -> tensor<128xi1, #blocked>
    tt.store %5, %6, %4 : tensor<128x!tt.ptr<i1>, #blocked>
    tt.return
  }
}
