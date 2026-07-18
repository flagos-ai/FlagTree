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

// RUN: triton-opt %s --allocate-shared-memory | FileCheck %s

#blocked = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [32, 1], warpsPerCTA = [2, 2], order = [1, 0]}>

// CHECK-LABEL: module
// CHECK-SAME: ttg.shared = 131072 : i32
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32} {

// CHECK-LABEL: @gather_op
// TODO(jeff): Optimize the lowering to reduce shared memory usage.
tt.func @gather_op(%arg0: tensor<1024x256xi32, #blocked>, %arg1: tensor<128x256xf32, #blocked>) {
  // CHECK-NEXT: allocation.offset = 0 : i32
  %0 = tt.gather %arg1[%arg0] {axis = 0 : i32} : (tensor<128x256xf32, #blocked>, tensor<1024x256xi32, #blocked>) -> tensor<1024x256xf32, #blocked>
  tt.return
}

}
