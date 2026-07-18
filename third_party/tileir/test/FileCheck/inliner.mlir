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

// RUN: triton-cuda-tile-opt %s --pass-pipeline="builtin.module(convert-triton-to-cuda-tile,cuda_tile.module(cuda_tile.experimental\$func(fuse-fma)),reconcile-unrealized-casts, inline)" | FileCheck %s

// CHECK-LABEL: kernel_12
module @kernel_12 {
  // CHECK-NOT: @kernel
  tt.func private @kernel(%arg0: i32) {
    tt.return
  }

  // CHECK-LABEL: @main
  tt.func @main(%arg0: i32) {
    // CHECK-NOT: call
    tt.call @kernel(%arg0) : (i32) -> ()
    tt.return
  }
}
