// Copyright 2025- FlagOS Contributors
// SPDX-License-Identifier: MIT
// RUN: triton-opt %s -triton-tle-lower-barriers -split-input-file | FileCheck %s
//
// A helper's shared barrier storage can overlap a later helper's TMA weight
// tile. PTX requires mbarrier.inval before the storage has any unrelated use.
// In the GLM TP4 decode reproducer, all 64 origin CTAs read nonfinite BF16
// halfwords at [76800, 76832), exactly the preceding helper's four mbarriers.
// The observed 0xfffe/0x7fff words match its hardware initialization state.

#shared = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [1, 0]}>
#slot = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>
#smem = #ttg.shared_memory

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32} {
  // CHECK-LABEL: tt.func @owned_helper_barriers
  tt.func @owned_helper_barriers() {
    %c0 = arith.constant 0 : i32
    %bars = tle.barrier.alloc {arrive_count = 1 : i32, init_polarity = 0 : i32, num_barriers = 2 : i32} : !ttg.memdesc<2x1xi64, #shared, #smem, mutable>
    %s0 = ttg.memdesc_index %bars[%c0] : !ttg.memdesc<2x1xi64, #shared, #smem, mutable> -> !ttg.memdesc<1xi64, #slot, #smem, mutable>
    tle.barrier.arrive %s0, %c0 {arrive_count = 1 : i32, backend = "mbarrier", named_id = 0 : i32, named_num_threads = 0 : i32} : !ttg.memdesc<1xi64, #slot, #smem, mutable>
    tle.barrier.wait %s0, %c0 {backend = "mbarrier", named_id = 0 : i32, named_num_threads = 0 : i32} : !ttg.memdesc<1xi64, #slot, #smem, mutable>
    // CHECK: ttng.init_barrier
    // CHECK: ttng.init_barrier
    // CHECK: gpu.barrier
    // CHECK: ttng.arrive_barrier
    // CHECK: ttng.wait_barrier
    // CHECK: gpu.barrier
    // CHECK: ttng.inval_barrier
    // CHECK: ttng.inval_barrier
    // CHECK: gpu.barrier
    // CHECK-NEXT: tt.return
    tt.return
  }
}

// -----

#shared = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [1, 0]}>
#slot = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>
#smem = #ttg.shared_memory

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32} {
  // CHECK-LABEL: tt.func @loop_owned_barrier
  tt.func @loop_owned_barrier(%end: i32) {
    %c0 = arith.constant 0 : i32
    %c1 = arith.constant 1 : i32
    scf.for %i = %c0 to %end step %c1 : i32 {
      %bar = tle.barrier.alloc {arrive_count = 1 : i32, init_polarity = 0 : i32, num_barriers = 1 : i32} : !ttg.memdesc<1x1xi64, #shared, #smem, mutable>
      %s = ttg.memdesc_index %bar[%c0] : !ttg.memdesc<1x1xi64, #shared, #smem, mutable> -> !ttg.memdesc<1xi64, #slot, #smem, mutable>
      tle.barrier.arrive %s, %c0 {arrive_count = 1 : i32, backend = "mbarrier", named_id = 0 : i32, named_num_threads = 0 : i32} : !ttg.memdesc<1xi64, #slot, #smem, mutable>
      tle.barrier.wait %s, %c0 {backend = "mbarrier", named_id = 0 : i32, named_num_threads = 0 : i32} : !ttg.memdesc<1xi64, #slot, #smem, mutable>
      // CHECK: scf.for
      // CHECK: ttng.init_barrier
      // CHECK: gpu.barrier
      // CHECK: ttng.arrive_barrier
      // CHECK: ttng.wait_barrier
      // CHECK: gpu.barrier
      // CHECK: ttng.inval_barrier
      // CHECK: gpu.barrier
      // The printer omits a zero-operand scf.yield.
      // CHECK-NEXT: }
      // CHECK-NEXT: tt.return
      scf.yield
    }
    tt.return
  }
}

// -----

#shared = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [1, 0]}>
#slot = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>
#smem = #ttg.shared_memory

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32} {
  // Loop-carried aliases of an outer allocation remain owned by the outer scope.
  // CHECK-LABEL: tt.func @outer_loop_owned_barrier
  tt.func @outer_loop_owned_barrier(%end: i32) {
    %c0 = arith.constant 0 : i32
    %c1 = arith.constant 1 : i32
    %c2 = arith.constant 2 : i32
    %bars = tle.barrier.alloc {arrive_count = 1 : i32, init_polarity = 0 : i32, num_barriers = 2 : i32} : !ttg.memdesc<2x1xi64, #shared, #smem, mutable>
    // CHECK: ttng.init_barrier
    // CHECK: ttng.init_barrier
    // CHECK: gpu.barrier
    // CHECK: scf.for
    %result = scf.for %i = %c0 to %end step %c1 iter_args(%iter = %bars) -> (!ttg.memdesc<2x1xi64, #shared, #smem, mutable>) : i32 {
      %phase = arith.remui %i, %c2 : i32
      %s = ttg.memdesc_index %iter[%c0] : !ttg.memdesc<2x1xi64, #shared, #smem, mutable> -> !ttg.memdesc<1xi64, #slot, #smem, mutable>
      tle.barrier.arrive %s, %phase {arrive_count = 1 : i32, backend = "mbarrier", named_id = 0 : i32, named_num_threads = 0 : i32} : !ttg.memdesc<1xi64, #slot, #smem, mutable>
      tle.barrier.wait %s, %phase {backend = "mbarrier", named_id = 0 : i32, named_num_threads = 0 : i32} : !ttg.memdesc<1xi64, #slot, #smem, mutable>
      // CHECK: ttng.wait_barrier
      // CHECK-NOT: ttng.inval_barrier
      // CHECK: scf.yield
      scf.yield %iter : !ttg.memdesc<2x1xi64, #shared, #smem, mutable>
    }
    // CHECK: gpu.barrier
    // CHECK: ttng.inval_barrier
    // CHECK: ttng.inval_barrier
    // CHECK: gpu.barrier
    // CHECK-NEXT: tt.return
    tt.return
  }
}

// -----

#shared = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [1, 0]}>
#smem = #ttg.shared_memory

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32} {
  // CHECK-LABEL: tt.func @unused_barriers
  tt.func @unused_barriers() {
    %unused = tle.barrier.alloc {arrive_count = 1 : i32, init_polarity = 0 : i32, num_barriers = 2 : i32} : !ttg.memdesc<2x1xi64, #shared, #smem, mutable>
    // CHECK-NOT: ttng.init_barrier
    // CHECK-NOT: ttng.inval_barrier
    // CHECK-NOT: gpu.barrier
    // CHECK: tt.return
    tt.return
  }
}

// -----

#shared = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [1, 0]}>
#slot = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>
#smem = #ttg.shared_memory

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32} {
  // A borrowed barrier belongs to its caller and must remain initialized.
  // CHECK-LABEL: tt.func @borrowed_barrier
  tt.func @borrowed_barrier(%s: !ttg.memdesc<1xi64, #slot, #smem, mutable>) {
    %c0 = arith.constant 0 : i32
    tle.barrier.arrive %s, %c0 {arrive_count = 1 : i32, backend = "mbarrier", named_id = 0 : i32, named_num_threads = 0 : i32} : !ttg.memdesc<1xi64, #slot, #smem, mutable>
    tle.barrier.wait %s, %c0 {backend = "mbarrier", named_id = 0 : i32, named_num_threads = 0 : i32} : !ttg.memdesc<1xi64, #slot, #smem, mutable>
    // CHECK: ttng.wait_barrier
    // CHECK-NOT: ttng.inval_barrier
    // CHECK-NEXT: tt.return
    tt.return
  }
}
