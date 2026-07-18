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

// RUN: triton-opt %s --gluon-inline | FileCheck %s

#blocked = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>

module attributes {"ttg.target" = "cuda:90", "ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 32 : i32} {
  tt.func private @set_encoding(%arg0 : tensor<16xi32, #gluon.auto_encoding>) -> tensor<16xi32, #blocked> {
    %cvt = gluon.set_auto_layout %arg0 : tensor<16xi32, #gluon.auto_encoding> -> tensor<16xi32, #blocked>
    tt.return %cvt : tensor<16xi32, #blocked>
  }

  tt.func public @infer_make_range() -> tensor<16xi32, #blocked> {
    // CHECK-DAG: [[BLOCKED:#.*]] = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>
    // CHECK: [[CST:%.*]] = arith.constant dense<0> : tensor<16xi32, #gluon.auto_encoding>
    // CHECK: [[SET:%.*]] = gluon.set_auto_layout [[CST]] : tensor<16xi32, #gluon.auto_encoding> -> tensor<16xi32, [[BLOCKED]]>
    // CHECK: tt.return [[SET]] : tensor<16xi32, [[BLOCKED]]>
    %cst = arith.constant dense<0> : tensor<16xi32, #gluon.auto_encoding>
    %0 = tt.call @"set_encoding"(%cst) : (tensor<16xi32, #gluon.auto_encoding>) -> tensor<16xi32, #blocked>
    tt.return %0 : tensor<16xi32, #blocked>
  }
}
