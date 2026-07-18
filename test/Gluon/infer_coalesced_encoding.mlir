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

// RUN: triton-opt %s -split-input-file --gluon-infer-coalesced-encodings | FileCheck %s

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:90", "ttg.threads-per-warp" = 32 : i32} {
  tt.func public @infer_efficient(%in_ptr : !tt.ptr<f32>, %out_ptr : !tt.ptr<f32>) {
    // CHECK: [[BLOCKED:#.+]] = #ttg.blocked
    // CHECK: %[[IN_PTRS:.+]] = gluon.set_auto_layout {{.*}} : tensor<128x256x!tt.ptr<f32>, #gluon.auto_encoding> -> tensor<128x256x!tt.ptr<f32>, [[BLOCKED]]>
    // CHECK: %[[MASK_IN:.+]] = gluon.set_auto_layout {{.*}} : tensor<128x256xi1, #gluon.auto_encoding> -> tensor<128x256xi1, [[BLOCKED]]>
    // CHECK: %[[VALUE:.+]] = tt.load %[[IN_PTRS]], %[[MASK_IN]] : tensor<128x256x!tt.ptr<f32>, [[BLOCKED]]>
    %mask = arith.constant dense<0> : tensor<128x256xi1, #gluon.auto_encoding>
    %in_ptrs_1 = tt.splat %in_ptr : !tt.ptr<f32> -> tensor<128x256x!tt.ptr<f32>, #gluon.auto_encoding>
    %in_ptrs_2 = gluon.set_auto_layout %in_ptrs_1 : tensor<128x256x!tt.ptr<f32>, #gluon.auto_encoding> -> tensor<128x256x!tt.ptr<f32>, #gluon.coalesced_encoding>
    %mask_in = gluon.set_auto_layout %mask : tensor<128x256xi1, #gluon.auto_encoding> -> tensor<128x256xi1, #gluon.coalesced_encoding>
    %value = tt.load %in_ptrs_2, %mask_in : tensor<128x256x!tt.ptr<f32>, #gluon.coalesced_encoding>

    // CHECK: %[[SIN:.+]] = math.sin %[[VALUE]] : tensor<128x256xf32, [[BLOCKED]]>
    // CHECK: %[[MAX:.+]] = arith.maxnumf %[[SIN]], {{.*}} : tensor<128x256xf32, [[BLOCKED]]>
    %value_2 = math.sin %value : tensor<128x256xf32, #gluon.coalesced_encoding>
    %cst = arith.constant dense<0.000000e+00> : tensor<128x256xf32, #gluon.coalesced_encoding>
    %value_3 = arith.maxnumf %value_2, %cst : tensor<128x256xf32, #gluon.coalesced_encoding>

    // CHECK: %[[OUT_PTRS:.+]] = gluon.set_auto_layout {{.*}} : tensor<128x256x!tt.ptr<f32>, #gluon.auto_encoding> -> tensor<128x256x!tt.ptr<f32>, [[BLOCKED]]>
    // CHECK: %[[MASK_OUT:.+]] = gluon.set_auto_layout {{.*}} : tensor<128x256xi1, #gluon.auto_encoding> -> tensor<128x256xi1, [[BLOCKED]]>
    // CHECK: tt.store %[[OUT_PTRS]], %[[MAX]], %[[MASK_OUT]] : tensor<128x256x!tt.ptr<f32>, [[BLOCKED]]>
    %out_ptrs_1 = tt.splat %out_ptr : !tt.ptr<f32> -> tensor<128x256x!tt.ptr<f32>, #gluon.auto_encoding>
    %out_ptrs_2 = gluon.set_auto_layout %out_ptrs_1 : tensor<128x256x!tt.ptr<f32>, #gluon.auto_encoding> -> tensor<128x256x!tt.ptr<f32>, #gluon.coalesced_encoding>
    %mask_out = gluon.set_auto_layout %mask : tensor<128x256xi1, #gluon.auto_encoding> -> tensor<128x256xi1, #gluon.coalesced_encoding>
    tt.store %out_ptrs_2, %value_3, %mask_out : tensor<128x256x!tt.ptr<f32>, #gluon.coalesced_encoding>
    tt.return
  }
}



// -----
