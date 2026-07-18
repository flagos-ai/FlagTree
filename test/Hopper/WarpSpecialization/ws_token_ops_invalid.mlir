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

// RUN: triton-opt %s -split-input-file -verify-diagnostics

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:90", "ttg.threads-per-warp" = 32 : i32} {
  tt.func @reject_zero_num_buffers() {
    // expected-error @+1 {{requires positive numBuffers}}
    %token = nvws.create_token {loadType = 3 : i32, numBuffers = 0 : i32} : tensor<0x!nvws.token>
    tt.return
  }
}

// -----

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:90", "ttg.threads-per-warp" = 32 : i32} {
  tt.func @reject_zero_full_count() {
    // expected-error @+1 {{requires positive full_count}}
    %token = nvws.create_token {full_count = 0 : i32, loadType = 3 : i32, numBuffers = 2 : i32} : tensor<2x!nvws.token>
    tt.return
  }
}

// -----

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:90", "ttg.threads-per-warp" = 32 : i32} {
  tt.func @reject_negative_empty_count() {
    // expected-error @+1 {{requires positive empty_count}}
    %token = nvws.create_token {empty_count = -1 : i32, loadType = 3 : i32, numBuffers = 2 : i32} : tensor<2x!nvws.token>
    tt.return
  }
}

// -----

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:90", "ttg.threads-per-warp" = 32 : i32} {
  tt.func @reject_zero_release_count() {
    %idx = arith.constant 0 : i32
    %token = nvws.create_token {loadType = 3 : i32, numBuffers = 2 : i32} : tensor<2x!nvws.token>
    // expected-error @+1 {{requires positive release_count}}
    nvws.consumer_release %token, %idx {release_count = 0 : i32} : tensor<2x!nvws.token>, i32
    tt.return
  }
}
