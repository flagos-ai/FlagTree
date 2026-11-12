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
//

// RUN: triton-opt %s -tritongpu-coalesce | FileCheck %s

#blocked = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [1], order = [0]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 1 : i32, ttg.target = "cuda:90", "ttg.threads-per-warp" = 32 : i32} {
  // A tensor load and scalar polling share their CTA coordinate. Coalescing's
  // dependency closure must traverse both scf.while regions through iter args.
  // CHECK-LABEL: tt.func @poll_then_load
  // CHECK: scf.while
  // CHECK: tt.atomic_rmw add, acquire, sys
  // CHECK: tt.load
  // CHECK: tt.store
  // CHECK: tt.return
  tt.func @poll_then_load(%input: !tt.ptr<f32>, %output: !tt.ptr<f32>, %flags: !tt.ptr<i64>, %epoch: i64) {
    %zero = arith.constant 0 : i64
    %true = arith.constant true
    %width = arith.constant 32 : i32
    %cta = tt.get_program_id x : i32
    %flag = tt.addptr %flags, %cta : !tt.ptr<i64>, i32
    %first = tt.atomic_rmw add, acquire, sys, %flag, %zero, %true : (!tt.ptr<i64>, i64, i1) -> i64
    %last = scf.while (%seen = %first) : (i64) -> i64 {
      %pending = arith.cmpi ult, %seen, %epoch : i64
      scf.condition(%pending) %seen : i64
    } do {
    ^bb0(%seen: i64):
      %next = tt.atomic_rmw add, acquire, sys, %flag, %zero, %true : (!tt.ptr<i64>, i64, i1) -> i64
      scf.yield %next : i64
    }
    %base = arith.muli %cta, %width : i32
    %starts = tt.splat %base : i32 -> tensor<32xi32, #blocked>
    %lanes = tt.make_range {start = 0 : i32, end = 32 : i32} : tensor<32xi32, #blocked>
    %indices = arith.addi %starts, %lanes : tensor<32xi32, #blocked>
    %inputs = tt.splat %input : !tt.ptr<f32> -> tensor<32x!tt.ptr<f32>, #blocked>
    %outputs = tt.splat %output : !tt.ptr<f32> -> tensor<32x!tt.ptr<f32>, #blocked>
    %source = tt.addptr %inputs, %indices : tensor<32x!tt.ptr<f32>, #blocked>, tensor<32xi32, #blocked>
    %destination = tt.addptr %outputs, %indices : tensor<32x!tt.ptr<f32>, #blocked>, tensor<32xi32, #blocked>
    %values = tt.load %source : tensor<32x!tt.ptr<f32>, #blocked>
    tt.store %destination, %values : tensor<32x!tt.ptr<f32>, #blocked>
    tt.return
  }
}
