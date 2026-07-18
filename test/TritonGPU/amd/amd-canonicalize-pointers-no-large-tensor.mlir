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

// RUN: triton-opt %s -allow-unregistered-dialect -split-input-file -tritonamdgpu-canonicalize-pointers="enable-large-tensor-ptr-canon=false" -canonicalize -verify-diagnostics | FileCheck %s

// this case is copied from amd-canonicalize-pointers-no-large-tensor.mlir. With
// enable-large-tensor-ptr-canon=false, the input is not changed at all.
module attributes {"ttg.num-warps" = 4 : i32} {
  tt.func @conversion1(%arg0: !tt.ptr<f32>) -> tensor<1024xf32> {
    %c1024_i32 = arith.constant 1024 : i32
    %0 = tt.get_program_id x : i32
    %1 = arith.muli %0, %c1024_i32 : i32
    %2 = tt.splat %1 : i32 -> tensor<1024xi32>
    %3 = tt.splat %arg0 : !tt.ptr<f32> -> tensor<1024x!tt.ptr<f32>>
    %4 = tt.addptr %3, %2 : tensor<1024x!tt.ptr<f32>>, tensor<1024xi32>
    %5 = tt.load %4 : tensor<1024x!tt.ptr<f32>>
    tt.return %5 : tensor<1024xf32>
  }
}

// CHECK-LABEL:   tt.func @conversion1
// CHECK: %[[ADDPTR:.*]] = tt.addptr
// CHECK:                = tt.load %[[ADDPTR]]
