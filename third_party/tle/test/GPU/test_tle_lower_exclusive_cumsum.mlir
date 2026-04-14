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

// RUN: triton-opt %s -split-input-file -triton-tle-lower-exclusive-cumsum | FileCheck %s

#blocked = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [1], order = [0]}>

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 1 : i32, ttg.target = "cuda:90", "ttg.threads-per-warp" = 32 : i32} {
  // CHECK-LABEL: tt.func @lower_exclusive_cumsum_i32
  tt.func @lower_exclusive_cumsum_i32(%arg0: tensor<16xi32, #blocked>) -> (tensor<16xi32, #blocked>, i32) {
    // CHECK: %[[SCAN:.*]] = "tt.scan"(%arg0)
    // CHECK: %[[EXCLUSIVE:.*]] = arith.subi %[[SCAN]], %arg0 : tensor<16xi32, #blocked>
    // CHECK: %[[RANGE:.*]] = tt.make_range
    // CHECK: %[[REDUCE:.*]]:2 = "tt.reduce"(%[[RANGE]], %[[SCAN]])
    // CHECK: arith.cmpi sgt
    // CHECK-NOT: "tle.exclusive_cumsum"
    // CHECK: tt.return %[[EXCLUSIVE]], %[[REDUCE]]#1
    %exclusive, %total = "tle.exclusive_cumsum"(%arg0) {axis = 0 : i32, reverse = false} : (tensor<16xi32, #blocked>) -> (tensor<16xi32, #blocked>, i32)
    tt.return %exclusive, %total : tensor<16xi32, #blocked>, i32
  }
}

// -----

#blocked = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [1], order = [0]}>

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 1 : i32, ttg.target = "cuda:90", "ttg.threads-per-warp" = 32 : i32} {
  // CHECK-LABEL: tt.func @lower_exclusive_cumsum_f32_reverse
  tt.func @lower_exclusive_cumsum_f32_reverse(%arg0: tensor<16xf32, #blocked>) -> (tensor<16xf32, #blocked>, f32) {
    // CHECK: %[[SCAN:.*]] = "tt.scan"(%arg0) <{axis = 0 : i32, reverse = true}>
    // CHECK: %[[EXCLUSIVE:.*]] = arith.subf %[[SCAN]], %arg0 : tensor<16xf32, #blocked>
    // CHECK: %[[RANGE:.*]] = tt.make_range
    // CHECK: %[[REDUCE:.*]]:2 = "tt.reduce"(%[[RANGE]], %[[SCAN]])
    // CHECK: arith.cmpi slt
    // CHECK-NOT: "tle.exclusive_cumsum"
    // CHECK: tt.return %[[EXCLUSIVE]], %[[REDUCE]]#1
    %exclusive, %total = "tle.exclusive_cumsum"(%arg0) {axis = 0 : i32, reverse = true} : (tensor<16xf32, #blocked>) -> (tensor<16xf32, #blocked>, f32)
    tt.return %exclusive, %total : tensor<16xf32, #blocked>, f32
  }
}
