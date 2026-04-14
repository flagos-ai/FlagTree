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

// RUN: triton-opt %s -test-print-alignment -split-input-file 2>&1 | FileCheck %s

tt.func @ignore_rank_mismatched_axis_hints() {
  // CHECK: tt.make_range {{.*}} => contiguity = [128], divisibility = [1073741824], constancy = [1], constant_value = <none>
  %0 = tt.make_range {end = 128 : i32, start = 0 : i32} : tensor<128xi32>
  // CHECK: tt.expand_dims {{.*}} => contiguity = [1, 128], divisibility = [1, 1073741824], constancy = [1, 1], constant_value = <none>
  %1 = tt.expand_dims %0 {axis = 0 : i32} : tensor<128xi32> -> tensor<1x128xi32>
  // Scalar tt.* hints on rank-2 tensor results are malformed and must be
  // ignored. Otherwise AxisInfo rank may be shrunk and later vectorization can
  // query out-of-bounds dimensions.
  // CHECK: tt.broadcast {{.*}} => contiguity = [1, 128], divisibility = [1, 1073741824], constancy = [2, 1], constant_value = <none>
  %2 = tt.broadcast %1 {tt.contiguity = 8 : i32, tt.divisibility = 16 : i32, tt.constancy = 4 : i32} : tensor<1x128xi32> -> tensor<2x128xi32>
  tt.return
}
