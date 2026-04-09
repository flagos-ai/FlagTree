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

// RUN: triton-opt %s -split-input-file -triton-tle-select-encodings | FileCheck %s

#blocked = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [32, 1], warpsPerCTA = [1, 1], order = [1, 0]}>
#blocked4 = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [1, 1], order = [1, 0]}>
#shared = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>
#smem = #ttg.shared_memory

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 1 : i32, ttg.target = "cuda:90", "ttg.threads-per-warp" = 32 : i32} {
  // CHECK-LABEL: tt.func @atomic_votes_drive_encoding
  tt.func @atomic_votes_drive_encoding() {
    %idx = arith.constant dense<0> : tensor<32x4xi32, #blocked4>
    %ones = arith.constant dense<1> : tensor<32x4xi32, #blocked>
    %mask = arith.constant dense<true> : tensor<32x4xi1, #blocked>
    %smem = ttg.local_alloc : () -> !ttg.memdesc<1xi32, #shared, #smem, mutable>
    %ptrs = "tle.local_pointers"(%smem, %idx) : (!ttg.memdesc<1xi32, #shared, #smem, mutable>, tensor<32x4xi32, #blocked4>) -> tensor<32x4x!tt.ptr<i32, 3>, #blocked4>
    %ptrs_blocked = ttg.convert_layout %ptrs : tensor<32x4x!tt.ptr<i32, 3>, #blocked4> -> tensor<32x4x!tt.ptr<i32, 3>, #blocked>
    %old = tt.atomic_rmw add, relaxed, cta, %ptrs_blocked, %ones, %mask : (tensor<32x4x!tt.ptr<i32, 3>, #blocked>, tensor<32x4xi32, #blocked>, tensor<32x4xi1, #blocked>) -> tensor<32x4xi32, #blocked>
    tt.return
  }
  // CHECK: %[[A_IDX:.*]] = arith.constant dense<0> : tensor<32x4xi32, #[[A_IDX_ENC:[A-Za-z0-9_]+]]>
  // CHECK: %[[A_ONES:.*]] = arith.constant dense<1> : tensor<32x4xi32, #[[A_DATA_ENC:[A-Za-z0-9_]+]]>
  // CHECK: %[[A_MASK:.*]] = arith.constant dense<true> : tensor<32x4xi1, #[[A_DATA_ENC]]>
  // CHECK: %[[A_IDX_CAST:.*]] = ttg.convert_layout %[[A_IDX]] : tensor<32x4xi32, #[[A_IDX_ENC]]> -> tensor<32x4xi32, #[[A_DATA_ENC]]>
  // CHECK: %[[A_PTRS:.*]] = "tle.local_pointers"(%{{.*}}, %[[A_IDX_CAST]]) {{.*}} : (!ttg.memdesc<1xi32, #shared, #smem, mutable>, tensor<32x4xi32, #[[A_DATA_ENC]]>) -> tensor<32x4x!tt.ptr<i32, 3>, #[[A_DATA_ENC]]>
  // CHECK: tt.atomic_rmw add, relaxed, cta, %{{.*}}, %[[A_ONES]], %[[A_MASK]] : (tensor<32x4x!tt.ptr<i32, 3>, #[[A_DATA_ENC]]>, tensor<32x4xi32, #[[A_DATA_ENC]]>, tensor<32x4xi1, #[[A_DATA_ENC]]>) -> tensor<32x4xi32, #[[A_DATA_ENC]]>
}

// -----

#blocked = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [32, 1], warpsPerCTA = [1, 1], order = [1, 0]}>
#blocked4 = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [1, 1], order = [1, 0]}>
#shared = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>
#smem = #ttg.shared_memory

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 1 : i32, ttg.target = "cuda:90", "ttg.threads-per-warp" = 32 : i32} {
  // CHECK-LABEL: tt.func @index_convert_reused
  tt.func @index_convert_reused() {
    %idx = arith.constant dense<0> : tensor<32x4xi32, #blocked4>
    %vals = arith.constant dense<1> : tensor<32x4xi32, #blocked>
    %mask = arith.constant dense<true> : tensor<32x4xi1, #blocked>
    %smem0 = ttg.local_alloc : () -> !ttg.memdesc<1xi32, #shared, #smem, mutable>
    %smem1 = ttg.local_alloc : () -> !ttg.memdesc<1xi32, #shared, #smem, mutable>
    %ptr0 = "tle.local_pointers"(%smem0, %idx) : (!ttg.memdesc<1xi32, #shared, #smem, mutable>, tensor<32x4xi32, #blocked4>) -> tensor<32x4x!tt.ptr<i32, 3>, #blocked4>
    %ptr1 = "tle.local_pointers"(%smem1, %idx) : (!ttg.memdesc<1xi32, #shared, #smem, mutable>, tensor<32x4xi32, #blocked4>) -> tensor<32x4x!tt.ptr<i32, 3>, #blocked4>
    %ptr0_blocked = ttg.convert_layout %ptr0 : tensor<32x4x!tt.ptr<i32, 3>, #blocked4> -> tensor<32x4x!tt.ptr<i32, 3>, #blocked>
    %ptr1_blocked = ttg.convert_layout %ptr1 : tensor<32x4x!tt.ptr<i32, 3>, #blocked4> -> tensor<32x4x!tt.ptr<i32, 3>, #blocked>
    tt.store %ptr0_blocked, %vals, %mask : tensor<32x4x!tt.ptr<i32, 3>, #blocked>
    tt.store %ptr1_blocked, %vals, %mask : tensor<32x4x!tt.ptr<i32, 3>, #blocked>
    tt.return
  }
  // CHECK: %[[B_IDX:.*]] = arith.constant dense<0> : tensor<32x4xi32, #[[B_IDX_ENC:[A-Za-z0-9_]+]]>
  // CHECK: %[[B_VALS:.*]] = arith.constant dense<1> : tensor<32x4xi32, #[[B_DATA_ENC:[A-Za-z0-9_]+]]>
  // CHECK: %[[B_MASK:.*]] = arith.constant dense<true> : tensor<32x4xi1, #[[B_DATA_ENC]]>
  // CHECK: %[[B_IDX_CAST:.*]] = ttg.convert_layout %[[B_IDX]] : tensor<32x4xi32, #[[B_IDX_ENC]]> -> tensor<32x4xi32, #[[B_DATA_ENC]]>
  // CHECK: %[[B_PTR0:.*]] = "tle.local_pointers"(%{{.*}}, %[[B_IDX_CAST]]) {{.*}} : (!ttg.memdesc<1xi32, #shared, #smem, mutable>, tensor<32x4xi32, #[[B_DATA_ENC]]>) -> tensor<32x4x!tt.ptr<i32, 3>, #[[B_DATA_ENC]]>
  // CHECK: %[[B_PTR1:.*]] = "tle.local_pointers"(%{{.*}}, %[[B_IDX_CAST]]) {{.*}} : (!ttg.memdesc<1xi32, #shared, #smem, mutable>, tensor<32x4xi32, #[[B_DATA_ENC]]>) -> tensor<32x4x!tt.ptr<i32, 3>, #[[B_DATA_ENC]]>
  // CHECK-NOT: ttg.convert_layout %[[B_IDX]]
  // CHECK: tt.store %{{.*}}, %[[B_VALS]], %[[B_MASK]] : tensor<32x4x!tt.ptr<i32, 3>, #[[B_DATA_ENC]]>
  // CHECK: tt.store %{{.*}}, %[[B_VALS]], %[[B_MASK]] : tensor<32x4x!tt.ptr<i32, 3>, #[[B_DATA_ENC]]>
}

// -----

#blocked = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [32, 1], warpsPerCTA = [1, 1], order = [1, 0]}>
#blocked4 = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [1, 1], order = [1, 0]}>
#shared = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>
#smem = #ttg.shared_memory

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 1 : i32, ttg.target = "cuda:90", "ttg.threads-per-warp" = 32 : i32} {
  // CHECK-LABEL: tt.func @scalar_local_ptr_prefers_atomic_mask_encoding
  tt.func @scalar_local_ptr_prefers_atomic_mask_encoding() {
    %idx = arith.constant dense<0> : tensor<32x4xi32, #blocked>
    %ones = arith.constant dense<1> : tensor<32x4xi32, #blocked>
    %mask_seed = arith.constant dense<0> : tensor<32x4xi32, #blocked4>
    %mask = arith.cmpi eq, %mask_seed, %mask_seed : tensor<32x4xi32, #blocked4>
    %smem = ttg.local_alloc : () -> !ttg.memdesc<1xi32, #shared, #smem, mutable>
    %ptr = "tle.local_pointers"(%smem, %idx) : (!ttg.memdesc<1xi32, #shared, #smem, mutable>, tensor<32x4xi32, #blocked>) -> tensor<32x4x!tt.ptr<i32, 3>, #blocked>
    %ptr_b4 = ttg.convert_layout %ptr : tensor<32x4x!tt.ptr<i32, 3>, #blocked> -> tensor<32x4x!tt.ptr<i32, 3>, #blocked4>
    %ones_b4 = ttg.convert_layout %ones : tensor<32x4xi32, #blocked> -> tensor<32x4xi32, #blocked4>
    %old = tt.atomic_rmw add, relaxed, cta, %ptr_b4, %ones_b4, %mask : (tensor<32x4x!tt.ptr<i32, 3>, #blocked4>, tensor<32x4xi32, #blocked4>, tensor<32x4xi1, #blocked4>) -> tensor<32x4xi32, #blocked4>
    tt.return
  }
  // CHECK: %[[IDX:.*]] = arith.constant dense<0> : tensor<32x4xi32, #[[IDX_ENC:[A-Za-z0-9_]+]]>
  // CHECK: %[[MASK_SEED:.*]] = arith.constant dense<0> : tensor<32x4xi32, #[[MASK_ENC:[A-Za-z0-9_]+]]>
  // CHECK: %[[MASK:.*]] = arith.cmpi eq, %[[MASK_SEED]], %[[MASK_SEED]] : tensor<32x4xi32, #[[MASK_ENC]]>
  // CHECK: %[[IDX_CAST:.*]] = ttg.convert_layout %[[IDX]] : tensor<32x4xi32, #[[IDX_ENC]]> -> tensor<32x4xi32, #[[MASK_ENC]]>
  // CHECK: %[[PTR:.*]] = "tle.local_pointers"(%{{.*}}, %[[IDX_CAST]]) {{.*}} : (!ttg.memdesc<1xi32, #shared, #smem, mutable>, tensor<32x4xi32, #[[MASK_ENC]]>) -> tensor<32x4x!tt.ptr<i32, 3>, #[[MASK_ENC]]>
  // CHECK: tt.atomic_rmw add, relaxed, cta, %{{.*}}, %{{.*}}, %[[MASK]] : (tensor<32x4x!tt.ptr<i32, 3>, #[[MASK_ENC]]>, tensor<32x4xi32, #[[MASK_ENC]]>, tensor<32x4xi1, #[[MASK_ENC]]>) -> tensor<32x4xi32, #[[MASK_ENC]]>
}

// -----

#blocked = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [32, 1], warpsPerCTA = [1, 1], order = [1, 0]}>
#blocked4 = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [1, 1], order = [1, 0]}>
#shared = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>
#smem = #ttg.shared_memory

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 1 : i32, ttg.target = "cuda:90", "ttg.threads-per-warp" = 32 : i32} {
  // CHECK-LABEL: tt.func @fold_pointer_convert_for_atomic_users
  tt.func @fold_pointer_convert_for_atomic_users() {
    %idx = arith.constant dense<0> : tensor<32x4xi32, #blocked>
    %vals = arith.constant dense<1> : tensor<32x4xi32, #blocked4>
    %mask = arith.constant dense<true> : tensor<32x4xi1, #blocked4>
    %smem = ttg.local_alloc : () -> !ttg.memdesc<1xi32, #shared, #smem, mutable>
    %ptr = "tle.local_pointers"(%smem, %idx) : (!ttg.memdesc<1xi32, #shared, #smem, mutable>, tensor<32x4xi32, #blocked>) -> tensor<32x4x!tt.ptr<i32, 3>, #blocked>
    %ptr_b4 = ttg.convert_layout %ptr : tensor<32x4x!tt.ptr<i32, 3>, #blocked> -> tensor<32x4x!tt.ptr<i32, 3>, #blocked4>
    %old = tt.atomic_rmw add, relaxed, cta, %ptr_b4, %vals, %mask : (tensor<32x4x!tt.ptr<i32, 3>, #blocked4>, tensor<32x4xi32, #blocked4>, tensor<32x4xi1, #blocked4>) -> tensor<32x4xi32, #blocked4>
    tt.return
  }
  // CHECK: %[[PTR:.*]] = "tle.local_pointers"(%{{.*}}, %{{.*}}) {{.*}} : (!ttg.memdesc<1xi32, #shared, #smem, mutable>, tensor<32x4xi32, #[[P_ENC:[A-Za-z0-9_]+]]>) -> tensor<32x4x!tt.ptr<i32, 3>, #[[P_ENC]]>
  // CHECK-NOT: ttg.convert_layout %[[PTR]] : tensor<32x4x!tt.ptr<i32, 3>, #[[P_ENC]]>
  // CHECK: tt.atomic_rmw add, relaxed, cta, %[[PTR]], %{{.*}}, %{{.*}} : (tensor<32x4x!tt.ptr<i32, 3>, #[[P_ENC]]>, tensor<32x4xi32, #[[P_ENC]]>, tensor<32x4xi1, #[[P_ENC]]>) -> tensor<32x4xi32, #[[P_ENC]]>
}
