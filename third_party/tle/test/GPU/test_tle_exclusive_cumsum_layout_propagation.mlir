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

// RUN: triton-opt %s -split-input-file -tritongpu-remove-layout-conversions | FileCheck %s

#blocked = #ttg.blocked<{sizePerThread = [2], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>
#blocked1 = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:90", "ttg.threads-per-warp" = 32 : i32} {
  // CHECK-LABEL: tt.func @remove_cumsum_layout_sandwich
  // CHECK: %[[EX:.*]], %[[TOT:.*]] = "tle.exclusive_cumsum"(%arg0) {{.*}} : (tensor<256xi32, #blocked>) -> (tensor<256xi32, #blocked>, i32)
  // CHECK-NOT: ttg.convert_layout %[[EX]]
  tt.func @remove_cumsum_layout_sandwich(%arg0: tensor<256xi32, #blocked>) -> (tensor<256xi32, #blocked>, i32) {
    %0 = ttg.convert_layout %arg0 : tensor<256xi32, #blocked> -> tensor<256xi32, #blocked1>
    %exclusive, %total = "tle.exclusive_cumsum"(%0) {axis = 0 : i32, reverse = false} : (tensor<256xi32, #blocked1>) -> (tensor<256xi32, #blocked1>, i32)
    %1 = ttg.convert_layout %exclusive : tensor<256xi32, #blocked1> -> tensor<256xi32, #blocked>
    tt.return %1, %total : tensor<256xi32, #blocked>, i32
  }
}

// -----

#blocked = #ttg.blocked<{sizePerThread = [2], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>
#blocked1 = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:90", "ttg.threads-per-warp" = 32 : i32} {
  // CHECK-LABEL: tt.func @keep_convert_for_non_convert_user
  // CHECK: %[[EX:.*]], %[[TOT:.*]] = tle.exclusive_cumsum %arg0 {axis = 0 : i32, reverse = false} : tensor<256xi32, #blocked> -> tensor<256xi32, #blocked>, i32
  // CHECK: %[[ADD:.*]] = arith.addi %[[EX]], %[[EX]] : tensor<256xi32, #blocked>
  // CHECK: %[[RET:.*]] = ttg.convert_layout %[[ADD]] : tensor<256xi32, #blocked> -> tensor<256xi32, #blocked1>
  tt.func @keep_convert_for_non_convert_user(%arg0: tensor<256xi32, #blocked>) -> (tensor<256xi32, #blocked1>, i32) {
    %0 = ttg.convert_layout %arg0 : tensor<256xi32, #blocked> -> tensor<256xi32, #blocked1>
    %exclusive, %total = "tle.exclusive_cumsum"(%0) {axis = 0 : i32, reverse = false} : (tensor<256xi32, #blocked1>) -> (tensor<256xi32, #blocked1>, i32)
    %1 = arith.addi %exclusive, %exclusive : tensor<256xi32, #blocked1>
    tt.return %1, %total : tensor<256xi32, #blocked1>, i32
  }
}
