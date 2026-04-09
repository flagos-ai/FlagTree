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

// RUN: triton-opt %s -split-input-file -triton-tle-insert-local-pointer-barriers | FileCheck %s

#blocked = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [1], order = [0]}>
#shared = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>
#smem = #ttg.shared_memory

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 1 : i32, ttg.target = "cuda:90", "ttg.threads-per-warp" = 32 : i32} {
  // CHECK-LABEL: tt.func @uniform_reduce_or_if_barrier
  tt.func @uniform_reduce_or_if_barrier() {
    %idx = tt.make_range {end = 32 : i32, start = 0 : i32} : tensor<32xi32, #blocked>
    %zeros = arith.constant dense<0> : tensor<32xi32, #blocked>
    %ones = arith.constant dense<1> : tensor<32xi32, #blocked>
    %true_mask = arith.constant dense<true> : tensor<32xi1, #blocked>
    %smem0 = ttg.local_alloc : () -> !ttg.memdesc<32xi32, #shared, #smem, mutable>
    %smem1 = ttg.local_alloc : () -> !ttg.memdesc<32xi32, #shared, #smem, mutable>
    %ptr0 = "tle.local_pointers"(%smem0, %idx) {tle.barrier_group = 0 : i64} : (!ttg.memdesc<32xi32, #shared, #smem, mutable>, tensor<32xi32, #blocked>) -> tensor<32x!tt.ptr<i32, 3>, #blocked>
    %ptr1 = "tle.local_pointers"(%smem1, %idx) {tle.barrier_group = 1 : i64} : (!ttg.memdesc<32xi32, #shared, #smem, mutable>, tensor<32xi32, #blocked>) -> tensor<32x!tt.ptr<i32, 3>, #blocked>
    tt.store %ptr0, %zeros : tensor<32x!tt.ptr<i32, 3>, #blocked>
    // CHECK: %[[FOUND:.*]] = "tt.reduce"(%{{.*}})
    // CHECK-NOT: gpu.barrier
    // CHECK: scf.if %[[FOUND]] {
    %found = "tt.reduce"(%true_mask) <{axis = 0 : i32}> ({
    ^bb0(%lhs: i1, %rhs: i1):
      %or = arith.ori %lhs, %rhs : i1
      tt.reduce.return %or : i1
    }) : (tensor<32xi1, #blocked>) -> i1
    scf.if %found {
      tt.store %ptr1, %ones : tensor<32x!tt.ptr<i32, 3>, #blocked>
    } else {
      // CHECK: } else {
      // CHECK-NEXT: gpu.barrier
      // CHECK-NEXT: %[[LOAD:.*]] = tt.load %{{.*}} : tensor<32x!tt.ptr<i32, 3>, #blocked>
      %load = tt.load %ptr0 : tensor<32x!tt.ptr<i32, 3>, #blocked>
      tt.store %ptr1, %load : tensor<32x!tt.ptr<i32, 3>, #blocked>
    }
    tt.return
  }
}

// -----

#blocked = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [1], order = [0]}>
#shared = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>
#smem = #ttg.shared_memory

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 1 : i32, ttg.target = "cuda:90", "ttg.threads-per-warp" = 32 : i32} {
  // CHECK-LABEL: tt.func @splat_store_scalar_load_barrier
  tt.func @splat_store_scalar_load_barrier() {
    %c0 = arith.constant 0 : i32
    %idx = tt.make_range {end = 32 : i32, start = 0 : i32} : tensor<32xi32, #blocked>
    %vals = arith.constant dense<7> : tensor<32xi32, #blocked>
    %mask = arith.cmpi eq, %idx, %idx : tensor<32xi32, #blocked>
    %smem = ttg.local_alloc : () -> !ttg.memdesc<1xi32, #shared, #smem, mutable>
    %ptr = "tle.local_pointers"(%smem, %c0) {tle.barrier_group = 5 : i64} : (!ttg.memdesc<1xi32, #shared, #smem, mutable>, i32) -> !tt.ptr<i32, 3>
    %ptrs = tt.splat %ptr : !tt.ptr<i32, 3> -> tensor<32x!tt.ptr<i32, 3>, #blocked>
    // CHECK: tt.store %{{.*}}, %{{.*}}, %{{.*}} : tensor<32x!tt.ptr<i32, 3>, #blocked>
    // CHECK-NEXT: gpu.barrier
    // CHECK-NEXT: %[[L:.*]] = tt.load %{{.*}} : !tt.ptr<i32, 3>
    tt.store %ptrs, %vals, %mask : tensor<32x!tt.ptr<i32, 3>, #blocked>
    %l = tt.load %ptr : !tt.ptr<i32, 3>
    tt.store %ptr, %l : !tt.ptr<i32, 3>
    tt.return
  }
}

// -----

#blocked = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [1], order = [0]}>
#shared = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>
#smem = #ttg.shared_memory

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 1 : i32, ttg.target = "cuda:90", "ttg.threads-per-warp" = 32 : i32} {
  // CHECK-LABEL: tt.func @addptr_store_scalar_load_barrier
  tt.func @addptr_store_scalar_load_barrier() {
    %c0 = arith.constant 0 : i32
    %c1 = arith.constant 1 : i32
    %c7 = arith.constant 7 : i32
    %smem = ttg.local_alloc : () -> !ttg.memdesc<4xi32, #shared, #smem, mutable>
    %ptr = "tle.local_pointers"(%smem, %c0) {tle.barrier_group = 6 : i64} : (!ttg.memdesc<4xi32, #shared, #smem, mutable>, i32) -> !tt.ptr<i32, 3>
    %ptr_next = tt.addptr %ptr, %c1 : !tt.ptr<i32, 3>, i32
    // CHECK: tt.store %{{.*}}, %{{.*}} : !tt.ptr<i32, 3>
    // CHECK-NEXT: gpu.barrier
    // CHECK-NEXT: %[[L:.*]] = tt.load %{{.*}} : !tt.ptr<i32, 3>
    tt.store %ptr_next, %c7 : !tt.ptr<i32, 3>
    %l = tt.load %ptr : !tt.ptr<i32, 3>
    tt.store %ptr, %l : !tt.ptr<i32, 3>
    tt.return
  }
}

// -----

#blocked = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [1], order = [0]}>
#shared = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>
#smem = #ttg.shared_memory

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 1 : i32, ttg.target = "cuda:90", "ttg.threads-per-warp" = 32 : i32} {
  // CHECK-LABEL: tt.func @single_barrier_for_consecutive_group_loads
  tt.func @single_barrier_for_consecutive_group_loads() {
    %c0 = arith.constant 0 : i32
    %c1 = arith.constant 1 : i32
    %smem0 = ttg.local_alloc : () -> !ttg.memdesc<1xi32, #shared, #smem, mutable>
    %smem1 = ttg.local_alloc : () -> !ttg.memdesc<1xi32, #shared, #smem, mutable>
    %ptr0 = "tle.local_pointers"(%smem0, %c0) {tle.barrier_group = 10 : i64} : (!ttg.memdesc<1xi32, #shared, #smem, mutable>, i32) -> !tt.ptr<i32, 3>
    %ptr1 = "tle.local_pointers"(%smem1, %c0) {tle.barrier_group = 11 : i64} : (!ttg.memdesc<1xi32, #shared, #smem, mutable>, i32) -> !tt.ptr<i32, 3>
    tt.store %ptr0, %c0 : !tt.ptr<i32, 3>
    tt.store %ptr1, %c1 : !tt.ptr<i32, 3>
    // CHECK: tt.store %{{.*}}, %{{.*}} : !tt.ptr<i32, 3>
    // CHECK: tt.store %{{.*}}, %{{.*}} : !tt.ptr<i32, 3>
    // CHECK-NEXT: gpu.barrier
    // CHECK-NEXT: %[[L0:.*]] = tt.load %{{.*}} : !tt.ptr<i32, 3>
    // CHECK-NEXT: %[[L1:.*]] = tt.load %{{.*}} : !tt.ptr<i32, 3>
    // CHECK-NOT: gpu.barrier
    %l0 = tt.load %ptr0 : !tt.ptr<i32, 3>
    %l1 = tt.load %ptr1 : !tt.ptr<i32, 3>
    %sum = arith.addi %l0, %l1 : i32
    tt.store %ptr0, %sum : !tt.ptr<i32, 3>
    tt.return
  }
}

// -----

#blocked = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [1], order = [0]}>
#shared = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>
#smem = #ttg.shared_memory

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 1 : i32, ttg.target = "cuda:90", "ttg.threads-per-warp" = 32 : i32} {
  // CHECK-LABEL: tt.func private @callee_pointer_arg_barrier
  tt.func private @callee_pointer_arg_barrier(%ptr: !tt.ptr<i32, 3>) {
    %cond = arith.constant true
    %v0 = arith.constant 0 : i32
    %v1 = arith.constant 1 : i32
    scf.if %cond {
      tt.store %ptr, %v1 : !tt.ptr<i32, 3>
    }
    // CHECK: scf.if %{{.*}} {
    // CHECK: tt.store %{{.*}}, %{{.*}} : !tt.ptr<i32, 3>
    // CHECK: }
    // CHECK-NEXT: gpu.barrier
    // CHECK-NEXT: %[[L:.*]] = tt.load %{{.*}} : !tt.ptr<i32, 3>
    %l = tt.load %ptr : !tt.ptr<i32, 3>
    %sum = arith.addi %l, %v0 : i32
    tt.store %ptr, %sum : !tt.ptr<i32, 3>
    tt.return
  }

  // CHECK-LABEL: tt.func @caller_passes_local_pointer
  tt.func @caller_passes_local_pointer() {
    %c0 = arith.constant 0 : i32
    %smem = ttg.local_alloc : () -> !ttg.memdesc<1xi32, #shared, #smem, mutable>
    %ptr = "tle.local_pointers"(%smem, %c0) {tle.barrier_group = 9 : i64} : (!ttg.memdesc<1xi32, #shared, #smem, mutable>, i32) -> !tt.ptr<i32, 3>
    tt.call @callee_pointer_arg_barrier(%ptr) : (!tt.ptr<i32, 3>) -> ()
    tt.return
  }
}

// -----

#blocked = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [1], order = [0]}>
#shared = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>
#smem = #ttg.shared_memory

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 1 : i32, ttg.target = "cuda:90", "ttg.threads-per-warp" = 32 : i32} {
  // CHECK-LABEL: tt.func @nested_store_outer_load_barrier
  tt.func @nested_store_outer_load_barrier() {
    %idx = tt.make_range {end = 32 : i32, start = 0 : i32} : tensor<32xi32, #blocked>
    %vals = arith.constant dense<7> : tensor<32xi32, #blocked>
    %cond = arith.constant true
    %smem = ttg.local_alloc : () -> !ttg.memdesc<32xi32, #shared, #smem, mutable>
    %ptr = "tle.local_pointers"(%smem, %idx) {tle.barrier_group = 3 : i64} : (!ttg.memdesc<32xi32, #shared, #smem, mutable>, tensor<32xi32, #blocked>) -> tensor<32x!tt.ptr<i32, 3>, #blocked>
    scf.if %cond {
      tt.store %ptr, %vals : tensor<32x!tt.ptr<i32, 3>, #blocked>
    }
    // CHECK: scf.if %{{.*}} {
    // CHECK: tt.store %{{.*}}, %{{.*}} : tensor<32x!tt.ptr<i32, 3>, #blocked>
    // CHECK: }
    // CHECK-NEXT: gpu.barrier
    // CHECK-NEXT: %[[LOAD:.*]] = tt.load %{{.*}} : tensor<32x!tt.ptr<i32, 3>, #blocked>
    %load = tt.load %ptr : tensor<32x!tt.ptr<i32, 3>, #blocked>
    tt.store %ptr, %load : tensor<32x!tt.ptr<i32, 3>, #blocked>
    tt.return
  }
}
