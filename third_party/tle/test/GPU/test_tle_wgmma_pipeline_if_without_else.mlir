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

// RUN: triton-opt %s -tritongpu-pipeline | FileCheck %s

#blocked = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>

module attributes {"ttg.target" = "cuda:90", "ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 32 : i32} {
  // CHECK-LABEL: tt.func @if_without_else_in_pipelined_loop
  tt.func @if_without_else_in_pipelined_loop(%arg0: !tt.ptr<i32> {tt.divisibility = 16 : i32}, %arg1: !tt.ptr<i32> {tt.divisibility = 16 : i32}, %arg2: i1) attributes {noinline = false} {
    %c1_i32 = arith.constant 1 : i32
    %c0_i32 = arith.constant 0 : i32
    %c17_i32 = arith.constant 17 : i32
    // CHECK: scf.for
    scf.for %iv = %c0_i32 to %c17_i32 step %c1_i32  : i32 {
      %0 = tt.addptr %arg0, %iv : !tt.ptr<i32>, i32
      %1 = tt.splat %0 : !tt.ptr<i32> -> tensor<1x!tt.ptr<i32>, #blocked>
      %2 = tt.load %1 : tensor<1x!tt.ptr<i32>, #blocked>
      // CHECK: scf.if
      scf.if %arg2 {
        %3 = tt.splat %arg1 : !tt.ptr<i32> -> tensor<1x!tt.ptr<i32>, #blocked>
        %4 = tt.addptr %3, %2 : tensor<1x!tt.ptr<i32>, #blocked>, tensor<1xi32, #blocked>
        %5 = arith.addi %iv, %c1_i32 : i32
        %6 = tt.splat %5 : i32 -> tensor<1xi32, #blocked>
        tt.store %4, %6 : tensor<1x!tt.ptr<i32>, #blocked>
      }
    } {tt.num_stages = 1 : i32}
    tt.return
  }
}
