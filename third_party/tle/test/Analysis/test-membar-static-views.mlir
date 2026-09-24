// Copyright 2025-     FlagOS Contributors
//
// Permission is hereby granted, free of charge, to any person obtaining
// a copy of this software and associated documentation files
// (the "Software"), to deal in the Software without restriction,
// including without limitation the rights to use, copy, modify, merge,
// publish, distribute, sublicense, and/or sell copies of the Software,
// and to permit persons to whom the Software is furnished to do so,
// subject to the following conditions:
//
// The above copyright notice and this permission notice shall be
// included in all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
// EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
// MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
// IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY
// CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,
// TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE
// SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

// RUN: triton-opt %s -split-input-file --allocate-shared-memory -test-print-membar | FileCheck %s

// Static sub-view tracking for multi-buffered shared allocations: disjoint
// slots of one allocation must not rendezvous, including when a slot is
// carried through loop iteration arguments.

#blocked = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [1, 1], order = [1, 0]}>
#shared = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [2, 1, 0]}>
#view = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [1, 0]}>
#smem = #ttg.shared_memory

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 1 : i32, ttg.target = "cuda:90", "ttg.threads-per-warp" = 32 : i32} {
  // CHECK-LABEL: @disjoint_slots
  // CHECK: ttg.local_store
  // CHECK-NEXT: ttg.local_load
  tt.func public @disjoint_slots(%v: tensor<16x16xf32, #blocked>) -> tensor<16x16xf32, #blocked> {
    %c0 = arith.constant 0 : i32
    %c1 = arith.constant 1 : i32
    %buf = ttg.local_alloc : () -> !ttg.memdesc<2x16x16xf32, #shared, #smem, mutable>
    %s0 = ttg.memdesc_index %buf[%c0] : !ttg.memdesc<2x16x16xf32, #shared, #smem, mutable> -> !ttg.memdesc<16x16xf32, #view, #smem, mutable>
    %s1 = ttg.memdesc_index %buf[%c1] : !ttg.memdesc<2x16x16xf32, #shared, #smem, mutable> -> !ttg.memdesc<16x16xf32, #view, #smem, mutable>
    ttg.local_store %v, %s1 : tensor<16x16xf32, #blocked> -> !ttg.memdesc<16x16xf32, #view, #smem, mutable>
    %x = ttg.local_load %s0 : !ttg.memdesc<16x16xf32, #view, #smem, mutable> -> tensor<16x16xf32, #blocked>
    tt.return %x : tensor<16x16xf32, #blocked>
  }
}

// -----

#blocked = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [1, 1], order = [1, 0]}>
#shared = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [2, 1, 0]}>
#view = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [1, 0]}>
#smem = #ttg.shared_memory

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 1 : i32, ttg.target = "cuda:90", "ttg.threads-per-warp" = 32 : i32} {
  // The same slot keeps its read-after-write rendezvous.
  // CHECK-LABEL: @same_slot
  // CHECK: ttg.local_store
  // CHECK-NEXT: ttg.local_barrier
  // CHECK-NEXT: ttg.local_load
  tt.func public @same_slot(%v: tensor<16x16xf32, #blocked>) -> tensor<16x16xf32, #blocked> {
    %c0 = arith.constant 0 : i32
    %buf = ttg.local_alloc : () -> !ttg.memdesc<2x16x16xf32, #shared, #smem, mutable>
    %s0 = ttg.memdesc_index %buf[%c0] : !ttg.memdesc<2x16x16xf32, #shared, #smem, mutable> -> !ttg.memdesc<16x16xf32, #view, #smem, mutable>
    ttg.local_store %v, %s0 : tensor<16x16xf32, #blocked> -> !ttg.memdesc<16x16xf32, #view, #smem, mutable>
    %x = ttg.local_load %s0 : !ttg.memdesc<16x16xf32, #view, #smem, mutable> -> tensor<16x16xf32, #blocked>
    tt.return %x : tensor<16x16xf32, #blocked>
  }
}

// -----

#blocked = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [1, 1], order = [1, 0]}>
#shared = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [2, 1, 0]}>
#view = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [1, 0]}>
#smem = #ttg.shared_memory

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 1 : i32, ttg.target = "cuda:90", "ttg.threads-per-warp" = 32 : i32} {
  // A slot carried through iteration arguments keeps its interval, so writing
  // it does not rendezvous with a read of the other slot.
  // CHECK-LABEL: @carried_disjoint_slot
  // CHECK: scf.for
  // CHECK-NEXT: ttg.local_barrier
  // CHECK-NEXT: ttg.local_store
  // CHECK-NEXT: ttg.local_load
  tt.func public @carried_disjoint_slot(%v: tensor<16x16xf32, #blocked>, %n: i32) -> tensor<16x16xf32, #blocked> {
    %c0 = arith.constant 0 : i32
    %c1 = arith.constant 1 : i32
    %init = arith.constant dense<0.000000e+00> : tensor<16x16xf32, #blocked>
    %buf = ttg.local_alloc : () -> !ttg.memdesc<2x16x16xf32, #shared, #smem, mutable>
    %s0 = ttg.memdesc_index %buf[%c0] : !ttg.memdesc<2x16x16xf32, #shared, #smem, mutable> -> !ttg.memdesc<16x16xf32, #view, #smem, mutable>
    %s1 = ttg.memdesc_index %buf[%c1] : !ttg.memdesc<2x16x16xf32, #shared, #smem, mutable> -> !ttg.memdesc<16x16xf32, #view, #smem, mutable>
    %r:2 = scf.for %i = %c0 to %n step %c1 iter_args(%acc = %init, %cur = %s1) -> (tensor<16x16xf32, #blocked>, !ttg.memdesc<16x16xf32, #view, #smem, mutable>) : i32 {
      ttg.local_store %v, %cur : tensor<16x16xf32, #blocked> -> !ttg.memdesc<16x16xf32, #view, #smem, mutable>
      %x = ttg.local_load %s0 : !ttg.memdesc<16x16xf32, #view, #smem, mutable> -> tensor<16x16xf32, #blocked>
      %y = arith.addf %acc, %x : tensor<16x16xf32, #blocked>
      scf.yield %y, %cur : tensor<16x16xf32, #blocked>, !ttg.memdesc<16x16xf32, #view, #smem, mutable>
    }
    tt.return %r#0 : tensor<16x16xf32, #blocked>
  }
}

// -----

#blocked = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [1, 1], order = [1, 0]}>
#shared = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [2, 1, 0]}>
#view = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [1, 0]}>
#smem = #ttg.shared_memory

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 1 : i32, ttg.target = "cuda:90", "ttg.threads-per-warp" = 32 : i32} {
  // Carrying the slot that is also read keeps the rendezvous.
  // CHECK-LABEL: @carried_same_slot
  // CHECK: scf.for
  // CHECK-NEXT: ttg.local_barrier
  // CHECK-NEXT: ttg.local_store
  // CHECK-NEXT: ttg.local_barrier
  // CHECK-NEXT: ttg.local_load
  tt.func public @carried_same_slot(%v: tensor<16x16xf32, #blocked>, %n: i32) -> tensor<16x16xf32, #blocked> {
    %c0 = arith.constant 0 : i32
    %c1 = arith.constant 1 : i32
    %init = arith.constant dense<0.000000e+00> : tensor<16x16xf32, #blocked>
    %buf = ttg.local_alloc : () -> !ttg.memdesc<2x16x16xf32, #shared, #smem, mutable>
    %s0 = ttg.memdesc_index %buf[%c0] : !ttg.memdesc<2x16x16xf32, #shared, #smem, mutable> -> !ttg.memdesc<16x16xf32, #view, #smem, mutable>
    %r:2 = scf.for %i = %c0 to %n step %c1 iter_args(%acc = %init, %cur = %s0) -> (tensor<16x16xf32, #blocked>, !ttg.memdesc<16x16xf32, #view, #smem, mutable>) : i32 {
      ttg.local_store %v, %cur : tensor<16x16xf32, #blocked> -> !ttg.memdesc<16x16xf32, #view, #smem, mutable>
      %x = ttg.local_load %s0 : !ttg.memdesc<16x16xf32, #view, #smem, mutable> -> tensor<16x16xf32, #blocked>
      %y = arith.addf %acc, %x : tensor<16x16xf32, #blocked>
      scf.yield %y, %cur : tensor<16x16xf32, #blocked>, !ttg.memdesc<16x16xf32, #view, #smem, mutable>
    }
    tt.return %r#0 : tensor<16x16xf32, #blocked>
  }
}

// -----

#blocked = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [1, 1], order = [1, 0]}>
#shared = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [2, 1, 0]}>
#view = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [1, 0]}>
#smem = #ttg.shared_memory

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 1 : i32, ttg.target = "cuda:90", "ttg.threads-per-warp" = 32 : i32} {
  // The value yielded back is a view of the carried slot rather than the slot
  // itself. It resolves through the view the loop was entered with, so the
  // interval survives and the read of the other slot still does not rendezvous.
  // CHECK-LABEL: @carried_disjoint_derived_view
  // CHECK: scf.for
  // CHECK-NEXT: ttg.memdesc_subslice
  // CHECK-NEXT: ttg.local_barrier
  // CHECK-NEXT: ttg.local_store
  // CHECK-NEXT: ttg.local_load
  tt.func public @carried_disjoint_derived_view(%v: tensor<16x16xf32, #blocked>, %n: i32) -> tensor<16x16xf32, #blocked> {
    %c0 = arith.constant 0 : i32
    %c1 = arith.constant 1 : i32
    %init = arith.constant dense<0.000000e+00> : tensor<16x16xf32, #blocked>
    %buf = ttg.local_alloc : () -> !ttg.memdesc<2x16x16xf32, #shared, #smem, mutable>
    %s0 = ttg.memdesc_index %buf[%c0] : !ttg.memdesc<2x16x16xf32, #shared, #smem, mutable> -> !ttg.memdesc<16x16xf32, #view, #smem, mutable>
    %s1 = ttg.memdesc_index %buf[%c1] : !ttg.memdesc<2x16x16xf32, #shared, #smem, mutable> -> !ttg.memdesc<16x16xf32, #view, #smem, mutable>
    %r:2 = scf.for %i = %c0 to %n step %c1 iter_args(%acc = %init, %cur = %s1) -> (tensor<16x16xf32, #blocked>, !ttg.memdesc<16x16xf32, #view, #smem, mutable>) : i32 {
      %sub = ttg.memdesc_subslice %cur [0, 0] : !ttg.memdesc<16x16xf32, #view, #smem, mutable> -> !ttg.memdesc<16x16xf32, #view, #smem, mutable>
      ttg.local_store %v, %sub : tensor<16x16xf32, #blocked> -> !ttg.memdesc<16x16xf32, #view, #smem, mutable>
      %x = ttg.local_load %s0 : !ttg.memdesc<16x16xf32, #view, #smem, mutable> -> tensor<16x16xf32, #blocked>
      %y = arith.addf %acc, %x : tensor<16x16xf32, #blocked>
      scf.yield %y, %sub : tensor<16x16xf32, #blocked>, !ttg.memdesc<16x16xf32, #view, #smem, mutable>
    }
    tt.return %r#0 : tensor<16x16xf32, #blocked>
  }
}
