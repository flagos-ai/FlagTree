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

// RUN: triton-opt %s -convert-triton-to-tritongpu='target=cuda:90 num-warps=4' | FileCheck %s

// CHECK-DAG: #[[$NW4:.*]] = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>
// CHECK-DAG: #[[$NW1:.*]] = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [1], order = [0]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32} {
  // CHECK-LABEL: tt.func private @worker_nw1
  // CHECK-SAME: attributes {noinline = false, "ttg.num-warps" = 1 : i32}
  tt.func private @worker_nw1() attributes {noinline = false, "ttg.num-warps" = 1 : i32} {
    // CHECK: tt.make_range {{.*}} : tensor<64xi32, #[[$NW1]]>
    %0 = tt.make_range {end = 64 : i32, start = 0 : i32} : tensor<64xi32>
    // CHECK: arith.addi {{.*}} : tensor<64xi32, #[[$NW1]]>
    %1 = arith.addi %0, %0 : tensor<64xi32>
    tt.return
  }

  // CHECK-LABEL: tt.func public @kernel
  tt.func public @kernel() attributes {noinline = false} {
    // CHECK: tt.make_range {{.*}} : tensor<64xi32, #[[$NW4]]>
    %0 = tt.make_range {end = 64 : i32, start = 0 : i32} : tensor<64xi32>
    tt.call @worker_nw1() : () -> ()
    tt.return
  }
}
