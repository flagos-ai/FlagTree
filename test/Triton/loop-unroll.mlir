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

// RUN: triton-opt --split-input-file %s -triton-loop-unroll | FileCheck %s

tt.func @add_kernel_unroll(%arg0: tensor<256x!tt.ptr<f32>>, %arg1: i32) {
  %c1_i32 = arith.constant 1 : i32
  %cst = arith.constant 0.000000e+00 : f32
  %0 = tt.splat %c1_i32 : i32 -> tensor<256xi32>
  %1 = tt.splat %cst : f32 -> tensor<256xf32>
  // Check the loop is unrolled by factor of 2 and is followed by a reminder loop.
  // CHECK-LABEL: add_kernel_unroll
  // CHECK: scf.for
  // CHECK-COUNT-2: tt.load
  // CHECK-NOT: tt.load
  // CHECK: scf.for
  // CHECK: tt.load
  // CHECK-NOT: tt.load
  // CHECK: tt.num_stages = 1 : i32
  %2:2 = scf.for %arg3 = %c1_i32 to %arg1 step %c1_i32 iter_args(%arg4 = %1, %arg5 = %arg0) -> (tensor<256xf32>, tensor<256x!tt.ptr<f32>>)  : i32 {
      %3 = tt.load %arg5 : tensor<256x!tt.ptr<f32>>
    %4 = arith.addf %arg4, %3 : tensor<256xf32>
    %5 = tt.addptr %arg5, %0 : tensor<256x!tt.ptr<f32>>, tensor<256xi32>
    scf.yield %4, %5 : tensor<256xf32>, tensor<256x!tt.ptr<f32>>
  } {tt.loop_unroll_factor = 2 : i32}
  tt.return
}

// -----

tt.func @add_kernel_nounroll(%arg0: tensor<256x!tt.ptr<f32>>, %arg1: i32) {
  %c1_i32 = arith.constant 1 : i32
  %cst = arith.constant 0.000000e+00 : f32
  %0 = tt.splat %c1_i32 : i32 -> tensor<256xi32>
  %1 = tt.splat %cst : f32 -> tensor<256xf32>
  // Check the loop is not unrolled.
  // CHECK-LABEL: add_kernel_nounroll
  // CHECK: scf.for
  // CHECK-COUNT-1: tt.load
  // CHECK-NOT: tt.load
  // CHECK-NOT: scf.for
  %2:2 = scf.for %arg3 = %c1_i32 to %arg1 step %c1_i32 iter_args(%arg4 = %1, %arg5 = %arg0) -> (tensor<256xf32>, tensor<256x!tt.ptr<f32>>)  : i32 {
      %3 = tt.load %arg5 : tensor<256x!tt.ptr<f32>>
    %4 = arith.addf %arg4, %3 : tensor<256xf32>
    %5 = tt.addptr %arg5, %0 : tensor<256x!tt.ptr<f32>>, tensor<256xi32>
    scf.yield %4, %5 : tensor<256xf32>, tensor<256x!tt.ptr<f32>>
  }
  tt.return
}
