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

// RUN: triton-opt %s -tritongpu-F32DotTC="emu-tf32=1" -canonicalize  | FileCheck %s --check-prefixes=CHECK

// CHECK:     %[[DOT1:.*]] = tt.dot %[[LHS_LOW:.*]], %[[RHS_HIGH:.*]], %cst, inputPrecision = tf32 : tensor<16x16xf32> * tensor<16x16xf32> -> tensor<16x16xf32>
// CHECK:     %[[DOT2:.*]] = tt.dot %[[LHS_HIGH:.*]], %[[RHS_LOW:.*]], %[[DOT1]], inputPrecision = tf32 : tensor<16x16xf32> * tensor<16x16xf32> -> tensor<16x16xf32>
// CHECK:     %[[CMP:.*]] = arith.cmpf uno, %[[DOT2]], %[[DOT2]] : tensor<16x16xf32>
// CHECK:     %[[MASKED:.*]] = arith.select %[[CMP]], %cst, %[[DOT2]] : tensor<16x16xi1>, tensor<16x16xf32>
// CHECK:     %[[RESULT:.*]] = tt.dot %[[LHS_HIGH]], %[[RHS_HIGH]], %[[MASKED]], inputPrecision = tf32 : tensor<16x16xf32> * tensor<16x16xf32> -> tensor<16x16xf32>

module {
  tt.func @dot_test(%arg0: tensor<16x16xf32>, %arg1: tensor<16x16xf32>, %arg2: tensor<16x16xf32>) -> tensor<16x16xf32> {
    %4 = tt.dot %arg0, %arg1, %arg2, inputPrecision = tf32x3 : tensor<16x16xf32> * tensor<16x16xf32> -> tensor<16x16xf32>
    tt.return %4 : tensor<16x16xf32>
  }
}
