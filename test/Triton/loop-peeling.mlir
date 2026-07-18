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

// RUN: triton-opt %s -split-input-file -allow-unregistered-dialect -triton-test-loop-peeling -canonicalize | FileCheck %s

module attributes {"ttg.num-warps" = 4 : i32, "ttg.num-ctas" = 1 : i32} {
// CHECK-LABEL: @simple_loop_i32
// CHECK: (%[[LB:.*]]: i32, %[[UB:.*]]: i32, %[[STEP:.*]]: i32) -> f32
// CHECK-DAG: %[[CST:.*]] = arith.constant 0.000000e+00 : f32
// CHECK-DAG: %[[ONE:.*]] = arith.constant 1 : i32
// CHECK: %[[NUB:.*]] = arith.subi %[[UB]], %[[STEP]]
// CHECK: %[[FOR:.*]] = scf.for %[[IV:.*]] = %[[LB]] to %[[NUB]] step %[[STEP]]
// CHECK: scf.yield
// CHECK: %[[RANGE:.*]] = arith.subi %[[UB]], %[[LB]]
// CHECK: %[[RANGE_M1:.*]] = arith.subi %[[RANGE]], %[[ONE]]
// CHECK: %[[ITERS_M1:.*]] = arith.divsi %[[RANGE_M1]], %[[STEP]]
// CHECK: %[[DELTA:.*]] = arith.muli %[[ITERS_M1]], %[[STEP]]
// CHECK: %[[LAST_IV:.*]] = arith.addi %[[DELTA]], %[[LB]]
// CHECK: %[[COND:.*]] = arith.cmpi slt, %[[LB]], %[[UB]]
// CHECK: %[[IF:.*]] = scf.if %[[COND]]
// CHECK:   %[[DEF:.*]] = "def"(%[[LAST_IV]]) : (i32) -> f32
// CHECK:   %[[RES:.*]] = arith.addf %[[FOR]], %[[DEF]] : f32
// CHECK:   scf.yield %[[RES]] : f32
// CHECK: else
// CHECK:   scf.yield %[[FOR]] : f32
// CHECK: tt.return %[[IF]] : f32
tt.func @simple_loop_i32(%lb : i32, %ub : i32, %step : i32) -> f32 {
  %init = arith.constant 0.00e+00 : f32
  %loop = scf.for %iv = %lb to %ub step %step iter_args(%acc = %init) -> (f32) : i32 {
    %a = "def"(%iv) : (i32) -> f32
    %res = arith.addf %acc, %a : f32
    scf.yield %res : f32
  } {__test_peel_epilogue}

  tt.return %loop#0 : f32
}
}

// -----

module attributes {"ttg.num-warps" = 4 : i32, "ttg.num-ctas" = 1 : i32} {
// CHECK-LABEL: @simple_loop_i32
// CHECK: (%[[LB:.*]]: i32, %[[UB:.*]]: i32, %[[STEP:.*]]: i32) -> f32
// CHECK-DAG: %[[CST:.*]] = arith.constant 0.000000e+00 : f32
// CHECK-DAG: %[[ONE:.*]] = arith.constant 1 : i32
// CHECK: %[[NUB:.*]] = arith.subi %[[UB]], %[[STEP]]
// CHECK: %[[FOR:.*]] = scf.for %[[IV:.*]] = %[[LB]] to %[[NUB]] step %[[STEP]]
// CHECK: scf.yield
// CHECK: %[[RANGE:.*]] = arith.subi %[[UB]], %[[LB]]
// CHECK: %[[RANGE_M1:.*]] = arith.subi %[[RANGE]], %[[ONE]]
// CHECK: %[[ITERS_M1:.*]] = arith.divsi %[[RANGE_M1]], %[[STEP]]
// CHECK: %[[DELTA:.*]] = arith.muli %[[ITERS_M1]], %[[STEP]]
// CHECK: %[[LAST_IV:.*]] = arith.addi %[[DELTA]], %[[LB]]
// CHECK: %[[COND:.*]] = arith.cmpi slt, %[[LB]], %[[UB]]
// CHECK: %[[IF:.*]] = scf.if %[[COND]]
// CHECK:   %[[DEF:.*]] = "def"(%[[LAST_IV]]) : (i32) -> f32
// CHECK:   %[[RES:.*]] = arith.addf %[[FOR]], %[[DEF]] : f32
// CHECK:   scf.yield %[[RES]] : f32
// CHECK: else
// CHECK:   scf.yield %[[FOR]] : f32
// CHECK: tt.return %[[IF]] : f32
tt.func @simple_loop_i32(%lb : i32, %ub : i32, %step : i32) -> f32 {
  %init = arith.constant 0.00e+00 : f32
  %loop = scf.for %iv = %lb to %ub step %step iter_args(%acc = %init) -> (f32) : i32 {
    %a = "def"(%iv) : (i32) -> f32
    %res = arith.addf %acc, %a : f32
    scf.yield %res : f32
  } {__test_peel_epilogue}

  tt.return %loop#0 : f32
}
}
