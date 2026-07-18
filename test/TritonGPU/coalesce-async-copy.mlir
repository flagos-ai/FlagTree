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

// RUN: triton-opt %s -split-input-file -tritongpu-coalesce-async-copy | FileCheck %s

// CHECK: #[[NEW_BLOCKED:.*]] = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [16, 2], warpsPerCTA = [4, 1], order = [1, 0]}>
// CHECK: %{{.*}} = ttg.convert_layout %{{.*}} : {{.*}} -> tensor<64x16x!tt.ptr<i8>, #[[NEW_BLOCKED]]>
// CHECK: %{{.*}} = ttg.convert_layout %{{.*}} : {{.*}} -> tensor<64x16xi1, #[[NEW_BLOCKED]]>
// CHECK: %{{.*}} = ttg.convert_layout %{{.*}} : {{.*}} -> tensor<64x16xi8, #[[NEW_BLOCKED]]>
// CHECK: %{{.*}} = ttg.async_copy_global_to_local %{{.*}}: tensor<64x16x!tt.ptr<i8>, #[[NEW_BLOCKED]]>
#blocked = #ttg.blocked<{sizePerThread = [1, 16], threadsPerWarp = [4, 8], warpsPerCTA = [4, 1], order = [1, 0]}>
#shared = #ttg.swizzled_shared<{vec = 8, perPhase = 1, maxPhase = 8, order = [1, 0]}>
#smem = #ttg.shared_memory

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32} {
tt.func @async_copy_i8(%input: tensor<64x16x!tt.ptr<i8>, #blocked>,
    %view: !ttg.memdesc<64x16xi8, #shared, #smem, mutable>,
    %mask: tensor<64x16xi1, #blocked>,
    %other: tensor<64x16xi8, #blocked>) {
  %token = ttg.async_copy_global_to_local %input, %view mask %mask other %other: tensor<64x16x!tt.ptr<i8>, #blocked> -> <64x16xi8, #shared, #smem, mutable>
  tt.return
}
}

// -----

// CHECK: #[[NEW_BLOCKED:.*]] = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [16, 2], warpsPerCTA = [4, 1], order = [1, 0]}>
// CHECK: %{{.*}} = ttg.convert_layout %{{.*}} : {{.*}} -> tensor<64x16x!tt.ptr<i8>, #[[NEW_BLOCKED]]>
// CHECK: %{{.*}} = ttg.async_copy_global_to_local %{{.*}}: tensor<64x16x!tt.ptr<i8>, #[[NEW_BLOCKED]]>
#blocked = #ttg.blocked<{sizePerThread = [1, 16], threadsPerWarp = [4, 8], warpsPerCTA = [4, 1], order = [1, 0]}>
#shared = #ttg.swizzled_shared<{vec = 8, perPhase = 1, maxPhase = 8, order = [1, 0]}>
#smem = #ttg.shared_memory

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32} {
tt.func @async_copy_i8_no_mask_or_other(%input: tensor<64x16x!tt.ptr<i8>, #blocked>,
    %view: !ttg.memdesc<64x16xi8, #shared, #smem, mutable>) {
  %token = ttg.async_copy_global_to_local %input, %view : tensor<64x16x!tt.ptr<i8>, #blocked> -> <64x16xi8, #shared, #smem, mutable>
  tt.return
}
}
