// RUN: triton-opt %s -triton-tle-lower-barriers -split-input-file -verify-each | FileCheck %s
// Copyright 2026 FlagOS Contributors
// SPDX-License-Identifier: MIT
#shared = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [1, 0]}>
#slot = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>
#smem = #ttg.shared_memory
// CHECK-LABEL: tt.func @six_joint_full_empty
// CHECK-NOT: gpu.barrier
// CHECK: ttng.init_barrier
// CHECK-NOT: gpu.barrier
// CHECK: ttng.init_barrier
// CHECK-NOT: gpu.barrier
// CHECK: ttng.init_barrier
// CHECK-NOT: gpu.barrier
// CHECK: ttng.init_barrier
// CHECK-NOT: gpu.barrier
// CHECK: ttng.init_barrier
// CHECK-NOT: gpu.barrier
// CHECK: ttng.init_barrier
// CHECK: gpu.barrier
// CHECK-NOT: gpu.barrier
// CHECK: ttng.wait_barrier
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 8 : i32} {
  tt.func @six_joint_full_empty(%cond: i1) {
    %c0 = arith.constant 0 : i32
    %b0 = tle.barrier.alloc {arrive_count = 2 : i32, init_polarity = 0 : i32, num_barriers = 1 : i32} : !ttg.memdesc<1x1xi64, #shared, #smem, mutable>
    %b1 = tle.barrier.alloc {arrive_count = 2 : i32, init_polarity = 0 : i32, num_barriers = 1 : i32} : !ttg.memdesc<1x1xi64, #shared, #smem, mutable>
    %b2 = tle.barrier.alloc {arrive_count = 2 : i32, init_polarity = 0 : i32, num_barriers = 1 : i32} : !ttg.memdesc<1x1xi64, #shared, #smem, mutable>
    %b3 = tle.barrier.alloc {arrive_count = 1 : i32, init_polarity = 1 : i32, num_barriers = 1 : i32} : !ttg.memdesc<1x1xi64, #shared, #smem, mutable>
    %b4 = tle.barrier.alloc {arrive_count = 1 : i32, init_polarity = 1 : i32, num_barriers = 1 : i32} : !ttg.memdesc<1x1xi64, #shared, #smem, mutable>
    %b5 = tle.barrier.alloc {arrive_count = 1 : i32, init_polarity = 1 : i32, num_barriers = 1 : i32} : !ttg.memdesc<1x1xi64, #shared, #smem, mutable>
    %b0s = ttg.memdesc_index %b0[%c0] : !ttg.memdesc<1x1xi64, #shared, #smem, mutable> -> !ttg.memdesc<1xi64, #slot, #smem, mutable>
    tle.barrier.wait %b0s, %c0 {backend = "mbarrier", named_id = 1 : i32, named_num_threads = 128 : i32} : !ttg.memdesc<1xi64, #slot, #smem, mutable>
    %b1s = ttg.memdesc_index %b1[%c0] : !ttg.memdesc<1x1xi64, #shared, #smem, mutable> -> !ttg.memdesc<1xi64, #slot, #smem, mutable>
    tle.barrier.wait %b1s, %c0 {backend = "mbarrier", named_id = 1 : i32, named_num_threads = 128 : i32} : !ttg.memdesc<1xi64, #slot, #smem, mutable>
    %b2s = ttg.memdesc_index %b2[%c0] : !ttg.memdesc<1x1xi64, #shared, #smem, mutable> -> !ttg.memdesc<1xi64, #slot, #smem, mutable>
    tle.barrier.wait %b2s, %c0 {backend = "mbarrier", named_id = 1 : i32, named_num_threads = 128 : i32} : !ttg.memdesc<1xi64, #slot, #smem, mutable>
    %b3s = ttg.memdesc_index %b3[%c0] : !ttg.memdesc<1x1xi64, #shared, #smem, mutable> -> !ttg.memdesc<1xi64, #slot, #smem, mutable>
    tle.barrier.wait %b3s, %c0 {backend = "mbarrier", named_id = 1 : i32, named_num_threads = 128 : i32} : !ttg.memdesc<1xi64, #slot, #smem, mutable>
    %b4s = ttg.memdesc_index %b4[%c0] : !ttg.memdesc<1x1xi64, #shared, #smem, mutable> -> !ttg.memdesc<1xi64, #slot, #smem, mutable>
    tle.barrier.wait %b4s, %c0 {backend = "mbarrier", named_id = 1 : i32, named_num_threads = 128 : i32} : !ttg.memdesc<1xi64, #slot, #smem, mutable>
    %b5s = ttg.memdesc_index %b5[%c0] : !ttg.memdesc<1x1xi64, #shared, #smem, mutable> -> !ttg.memdesc<1xi64, #slot, #smem, mutable>
    tle.barrier.wait %b5s, %c0 {backend = "mbarrier", named_id = 1 : i32, named_num_threads = 128 : i32} : !ttg.memdesc<1xi64, #slot, #smem, mutable>
    tt.return
  }
}

// -----
#shared = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [1, 0]}>
#slot = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>
#smem = #ttg.shared_memory
// CHECK-LABEL: tt.func @array_control
// CHECK-NOT: gpu.barrier
// CHECK: ttng.init_barrier
// CHECK-NOT: gpu.barrier
// CHECK: ttng.init_barrier
// CHECK: gpu.barrier
// CHECK-NOT: gpu.barrier
// CHECK: ttng.wait_barrier
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 8 : i32} {
  tt.func @array_control(%cond: i1) {
    %c0 = arith.constant 0 : i32
    %arr = tle.barrier.alloc {arrive_count = 2 : i32, init_polarity = 0 : i32, num_barriers = 2 : i32} : !ttg.memdesc<2x1xi64, #shared, #smem, mutable>
    %arrs = ttg.memdesc_index %arr[%c0] : !ttg.memdesc<2x1xi64, #shared, #smem, mutable> -> !ttg.memdesc<1xi64, #slot, #smem, mutable>
    tle.barrier.wait %arrs, %c0 {backend = "mbarrier", named_id = 1 : i32, named_num_threads = 128 : i32} : !ttg.memdesc<1xi64, #slot, #smem, mutable>
    tt.return
  }
}

// -----
// Copyright 2026 FlagOS Contributors
// SPDX-License-Identifier: MIT
#shared = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [1, 0]}>
#slot = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>
#smem = #ttg.shared_memory
// CHECK-LABEL: tt.func @early_use
// CHECK-COUNT-4: gpu.barrier
// CHECK-NOT: gpu.barrier
// CHECK: tt.return
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 8 : i32} {
  tt.func @early_use(%cond: i1) {
    %c0 = arith.constant 0 : i32
    %a = tle.barrier.alloc {arrive_count = 1 : i32, init_polarity = 0 : i32, num_barriers = 1 : i32} : !ttg.memdesc<1x1xi64, #shared, #smem, mutable>
    %as = ttg.memdesc_index %a[%c0] : !ttg.memdesc<1x1xi64, #shared, #smem, mutable> -> !ttg.memdesc<1xi64, #slot, #smem, mutable>
    tle.barrier.wait %as, %c0 {backend = "mbarrier", named_id = 1 : i32, named_num_threads = 128 : i32} : !ttg.memdesc<1xi64, #slot, #smem, mutable>
    %b = tle.barrier.alloc {arrive_count = 1 : i32, init_polarity = 0 : i32, num_barriers = 1 : i32} : !ttg.memdesc<1x1xi64, #shared, #smem, mutable>
    %bs = ttg.memdesc_index %b[%c0] : !ttg.memdesc<1x1xi64, #shared, #smem, mutable> -> !ttg.memdesc<1xi64, #slot, #smem, mutable>
    tle.barrier.wait %bs, %c0 {backend = "mbarrier", named_id = 1 : i32, named_num_threads = 128 : i32} : !ttg.memdesc<1xi64, #slot, #smem, mutable>
    tt.return
  }
}

// -----
#shared = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [1, 0]}>
#slot = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>
#smem = #ttg.shared_memory
// CHECK-LABEL: tt.func @effect_gap
// CHECK-COUNT-5: gpu.barrier
// CHECK-NOT: gpu.barrier
// CHECK: tt.return
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 8 : i32} {
  tt.func @effect_gap(%cond: i1) {
    %c0 = arith.constant 0 : i32
    %a = tle.barrier.alloc {arrive_count = 1 : i32, init_polarity = 0 : i32, num_barriers = 1 : i32} : !ttg.memdesc<1x1xi64, #shared, #smem, mutable>
    gpu.barrier
    %b = tle.barrier.alloc {arrive_count = 1 : i32, init_polarity = 0 : i32, num_barriers = 1 : i32} : !ttg.memdesc<1x1xi64, #shared, #smem, mutable>
    %as = ttg.memdesc_index %a[%c0] : !ttg.memdesc<1x1xi64, #shared, #smem, mutable> -> !ttg.memdesc<1xi64, #slot, #smem, mutable>
    tle.barrier.wait %as, %c0 {backend = "mbarrier", named_id = 1 : i32, named_num_threads = 128 : i32} : !ttg.memdesc<1xi64, #slot, #smem, mutable>
    %bs = ttg.memdesc_index %b[%c0] : !ttg.memdesc<1x1xi64, #shared, #smem, mutable> -> !ttg.memdesc<1xi64, #slot, #smem, mutable>
    tle.barrier.wait %bs, %c0 {backend = "mbarrier", named_id = 1 : i32, named_num_threads = 128 : i32} : !ttg.memdesc<1xi64, #slot, #smem, mutable>
    tt.return
  }
}

// -----
#shared = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [1, 0]}>
#slot = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>
#smem = #ttg.shared_memory
// CHECK-LABEL: tt.func @pure_operation_gap
// CHECK-COUNT-4: gpu.barrier
// CHECK-NOT: gpu.barrier
// CHECK: tt.return
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 8 : i32} {
  tt.func @pure_operation_gap(%cond: i1) {
    %c0 = arith.constant 0 : i32
    %a = tle.barrier.alloc {arrive_count = 1 : i32, init_polarity = 0 : i32, num_barriers = 1 : i32} : !ttg.memdesc<1x1xi64, #shared, #smem, mutable>
    %v = arith.addi %c0, %c0 : i32
    %b = tle.barrier.alloc {arrive_count = 1 : i32, init_polarity = 0 : i32, num_barriers = 1 : i32} : !ttg.memdesc<1x1xi64, #shared, #smem, mutable>
    %as = ttg.memdesc_index %a[%c0] : !ttg.memdesc<1x1xi64, #shared, #smem, mutable> -> !ttg.memdesc<1xi64, #slot, #smem, mutable>
    tle.barrier.wait %as, %c0 {backend = "mbarrier", named_id = 1 : i32, named_num_threads = 128 : i32} : !ttg.memdesc<1xi64, #slot, #smem, mutable>
    %bs = ttg.memdesc_index %b[%c0] : !ttg.memdesc<1x1xi64, #shared, #smem, mutable> -> !ttg.memdesc<1xi64, #slot, #smem, mutable>
    tle.barrier.wait %bs, %c0 {backend = "mbarrier", named_id = 1 : i32, named_num_threads = 128 : i32} : !ttg.memdesc<1xi64, #slot, #smem, mutable>
    tt.return
  }
}

// -----
#shared = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [1, 0]}>
#slot = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>
#smem = #ttg.shared_memory
// CHECK-LABEL: tt.func @unused_allocation_gap
// CHECK-COUNT-4: gpu.barrier
// CHECK-NOT: gpu.barrier
// CHECK: tt.return
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 8 : i32} {
  tt.func @unused_allocation_gap(%cond: i1) {
    %c0 = arith.constant 0 : i32
    %a = tle.barrier.alloc {arrive_count = 1 : i32, init_polarity = 0 : i32, num_barriers = 1 : i32} : !ttg.memdesc<1x1xi64, #shared, #smem, mutable>
    %dead = tle.barrier.alloc {arrive_count = 1 : i32, init_polarity = 0 : i32, num_barriers = 1 : i32} : !ttg.memdesc<1x1xi64, #shared, #smem, mutable>
    %b = tle.barrier.alloc {arrive_count = 1 : i32, init_polarity = 0 : i32, num_barriers = 1 : i32} : !ttg.memdesc<1x1xi64, #shared, #smem, mutable>
    %as = ttg.memdesc_index %a[%c0] : !ttg.memdesc<1x1xi64, #shared, #smem, mutable> -> !ttg.memdesc<1xi64, #slot, #smem, mutable>
    tle.barrier.wait %as, %c0 {backend = "mbarrier", named_id = 1 : i32, named_num_threads = 128 : i32} : !ttg.memdesc<1xi64, #slot, #smem, mutable>
    %bs = ttg.memdesc_index %b[%c0] : !ttg.memdesc<1x1xi64, #shared, #smem, mutable> -> !ttg.memdesc<1xi64, #slot, #smem, mutable>
    tle.barrier.wait %bs, %c0 {backend = "mbarrier", named_id = 1 : i32, named_num_threads = 128 : i32} : !ttg.memdesc<1xi64, #slot, #smem, mutable>
    tt.return
  }
}

// -----
#shared = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [1, 0]}>
#slot = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>
#smem = #ttg.shared_memory
// CHECK-LABEL: tt.func @unused_view_gap
// CHECK-COUNT-4: gpu.barrier
// CHECK-NOT: gpu.barrier
// CHECK: tt.return
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 8 : i32} {
  tt.func @unused_view_gap(%cond: i1) {
    %c0 = arith.constant 0 : i32
    %a = tle.barrier.alloc {arrive_count = 1 : i32, init_polarity = 0 : i32, num_barriers = 1 : i32} : !ttg.memdesc<1x1xi64, #shared, #smem, mutable>
    %dead = tle.barrier.alloc {arrive_count = 1 : i32, init_polarity = 0 : i32, num_barriers = 1 : i32} : !ttg.memdesc<1x1xi64, #shared, #smem, mutable>
    %b = tle.barrier.alloc {arrive_count = 1 : i32, init_polarity = 0 : i32, num_barriers = 1 : i32} : !ttg.memdesc<1x1xi64, #shared, #smem, mutable>
    %deadview = ttg.memdesc_index %dead[%c0] : !ttg.memdesc<1x1xi64, #shared, #smem, mutable> -> !ttg.memdesc<1xi64, #slot, #smem, mutable>
    %as = ttg.memdesc_index %a[%c0] : !ttg.memdesc<1x1xi64, #shared, #smem, mutable> -> !ttg.memdesc<1xi64, #slot, #smem, mutable>
    tle.barrier.wait %as, %c0 {backend = "mbarrier", named_id = 1 : i32, named_num_threads = 128 : i32} : !ttg.memdesc<1xi64, #slot, #smem, mutable>
    %bs = ttg.memdesc_index %b[%c0] : !ttg.memdesc<1x1xi64, #shared, #smem, mutable> -> !ttg.memdesc<1xi64, #slot, #smem, mutable>
    tle.barrier.wait %bs, %c0 {backend = "mbarrier", named_id = 1 : i32, named_num_threads = 128 : i32} : !ttg.memdesc<1xi64, #slot, #smem, mutable>
    tt.return
  }
}

// -----
#shared = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [1, 0]}>
#slot = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>
#smem = #ttg.shared_memory
// CHECK-LABEL: tt.func @named_allocation_gap
// CHECK-COUNT-4: gpu.barrier
// CHECK-NOT: gpu.barrier
// CHECK: tt.return
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 8 : i32} {
  tt.func @named_allocation_gap(%cond: i1) {
    %c0 = arith.constant 0 : i32
    %a = tle.barrier.alloc {arrive_count = 1 : i32, init_polarity = 0 : i32, num_barriers = 1 : i32} : !ttg.memdesc<1x1xi64, #shared, #smem, mutable>
    %n = tle.barrier.alloc {arrive_count = 1 : i32, init_polarity = 0 : i32, num_barriers = 1 : i32} : !ttg.memdesc<1x1xi64, #shared, #smem, mutable>
    %b = tle.barrier.alloc {arrive_count = 1 : i32, init_polarity = 0 : i32, num_barriers = 1 : i32} : !ttg.memdesc<1x1xi64, #shared, #smem, mutable>
    %as = ttg.memdesc_index %a[%c0] : !ttg.memdesc<1x1xi64, #shared, #smem, mutable> -> !ttg.memdesc<1xi64, #slot, #smem, mutable>
    tle.barrier.wait %as, %c0 {backend = "mbarrier", named_id = 1 : i32, named_num_threads = 128 : i32} : !ttg.memdesc<1xi64, #slot, #smem, mutable>
    %ns = ttg.memdesc_index %n[%c0] : !ttg.memdesc<1x1xi64, #shared, #smem, mutable> -> !ttg.memdesc<1xi64, #slot, #smem, mutable>
    tle.barrier.wait %ns {backend = "named", named_id = 1 : i32, named_num_threads = 128 : i32} : !ttg.memdesc<1xi64, #slot, #smem, mutable>
    %bs = ttg.memdesc_index %b[%c0] : !ttg.memdesc<1x1xi64, #shared, #smem, mutable> -> !ttg.memdesc<1xi64, #slot, #smem, mutable>
    tle.barrier.wait %bs, %c0 {backend = "mbarrier", named_id = 1 : i32, named_num_threads = 128 : i32} : !ttg.memdesc<1xi64, #slot, #smem, mutable>
    tt.return
  }
}

// -----
#shared = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [1, 0]}>
#slot = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>
#smem = #ttg.shared_memory
// CHECK-LABEL: tt.func @mixed_backend_gap
// CHECK-COUNT-5: gpu.barrier
// CHECK-NOT: gpu.barrier
// CHECK: tt.return
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 8 : i32} {
  tt.func @mixed_backend_gap(%cond: i1) {
    %c0 = arith.constant 0 : i32
    %a = tle.barrier.alloc {arrive_count = 1 : i32, init_polarity = 0 : i32, num_barriers = 1 : i32} : !ttg.memdesc<1x1xi64, #shared, #smem, mutable>
    %n = tle.barrier.alloc {arrive_count = 1 : i32, init_polarity = 0 : i32, num_barriers = 1 : i32} : !ttg.memdesc<1x1xi64, #shared, #smem, mutable>
    %b = tle.barrier.alloc {arrive_count = 1 : i32, init_polarity = 0 : i32, num_barriers = 1 : i32} : !ttg.memdesc<1x1xi64, #shared, #smem, mutable>
    %as = ttg.memdesc_index %a[%c0] : !ttg.memdesc<1x1xi64, #shared, #smem, mutable> -> !ttg.memdesc<1xi64, #slot, #smem, mutable>
    tle.barrier.wait %as, %c0 {backend = "mbarrier", named_id = 1 : i32, named_num_threads = 128 : i32} : !ttg.memdesc<1xi64, #slot, #smem, mutable>
    %ns = ttg.memdesc_index %n[%c0] : !ttg.memdesc<1x1xi64, #shared, #smem, mutable> -> !ttg.memdesc<1xi64, #slot, #smem, mutable>
    tle.barrier.wait %ns {backend = "named", named_id = 1 : i32, named_num_threads = 128 : i32} : !ttg.memdesc<1xi64, #slot, #smem, mutable>
    tle.barrier.wait %ns, %c0 {backend = "mbarrier", named_id = 0 : i32, named_num_threads = 0 : i32} : !ttg.memdesc<1xi64, #slot, #smem, mutable>
    %bs = ttg.memdesc_index %b[%c0] : !ttg.memdesc<1x1xi64, #shared, #smem, mutable> -> !ttg.memdesc<1xi64, #slot, #smem, mutable>
    tle.barrier.wait %bs, %c0 {backend = "mbarrier", named_id = 1 : i32, named_num_threads = 128 : i32} : !ttg.memdesc<1xi64, #slot, #smem, mutable>
    tt.return
  }
}

// -----
#shared = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [1, 0]}>
#slot = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>
#smem = #ttg.shared_memory
// CHECK-LABEL: tt.func @native_only_use_gap
// CHECK-COUNT-5: gpu.barrier
// CHECK-NOT: gpu.barrier
// CHECK: tt.return
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 8 : i32} {
  tt.func @native_only_use_gap(%cond: i1) {
    %c0 = arith.constant 0 : i32
    %a = tle.barrier.alloc {arrive_count = 1 : i32, init_polarity = 0 : i32, num_barriers = 1 : i32} : !ttg.memdesc<1x1xi64, #shared, #smem, mutable>
    %n = tle.barrier.alloc {arrive_count = 1 : i32, init_polarity = 0 : i32, num_barriers = 1 : i32} : !ttg.memdesc<1x1xi64, #shared, #smem, mutable>
    %b = tle.barrier.alloc {arrive_count = 1 : i32, init_polarity = 0 : i32, num_barriers = 1 : i32} : !ttg.memdesc<1x1xi64, #shared, #smem, mutable>
    %as = ttg.memdesc_index %a[%c0] : !ttg.memdesc<1x1xi64, #shared, #smem, mutable> -> !ttg.memdesc<1xi64, #slot, #smem, mutable>
    tle.barrier.wait %as, %c0 {backend = "mbarrier", named_id = 1 : i32, named_num_threads = 128 : i32} : !ttg.memdesc<1xi64, #slot, #smem, mutable>
    %ns = ttg.memdesc_index %n[%c0] : !ttg.memdesc<1x1xi64, #shared, #smem, mutable> -> !ttg.memdesc<1xi64, #slot, #smem, mutable>
    ttng.wait_barrier %ns, %c0 : !ttg.memdesc<1xi64, #slot, #smem, mutable>
    %bs = ttg.memdesc_index %b[%c0] : !ttg.memdesc<1x1xi64, #shared, #smem, mutable> -> !ttg.memdesc<1xi64, #slot, #smem, mutable>
    tle.barrier.wait %bs, %c0 {backend = "mbarrier", named_id = 1 : i32, named_num_threads = 128 : i32} : !ttg.memdesc<1xi64, #slot, #smem, mutable>
    tt.return
  }
}

// -----
#shared = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [1, 0]}>
#slot = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>
#smem = #ttg.shared_memory
// CHECK-LABEL: tt.func @singleton_control
// CHECK-COUNT-3: gpu.barrier
// CHECK-NOT: gpu.barrier
// CHECK: tt.return
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 8 : i32} {
  tt.func @singleton_control(%cond: i1) {
    %c0 = arith.constant 0 : i32
    %a = tle.barrier.alloc {arrive_count = 1 : i32, init_polarity = 0 : i32, num_barriers = 1 : i32} : !ttg.memdesc<1x1xi64, #shared, #smem, mutable>
    %as = ttg.memdesc_index %a[%c0] : !ttg.memdesc<1x1xi64, #shared, #smem, mutable> -> !ttg.memdesc<1xi64, #slot, #smem, mutable>
    tle.barrier.wait %as, %c0 {backend = "mbarrier", named_id = 1 : i32, named_num_threads = 128 : i32} : !ttg.memdesc<1xi64, #slot, #smem, mutable>
    tt.return
  }
}

// -----
#shared = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [1, 0]}>
#slot = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>
#smem = #ttg.shared_memory
// CHECK-LABEL: tt.func @call_gap
// CHECK-COUNT-4: gpu.barrier
// CHECK-NOT: gpu.barrier
// CHECK: tt.return
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 8 : i32} {
  tt.func private @sink()
  tt.func @call_gap(%cond: i1) {
    %c0 = arith.constant 0 : i32
    %a = tle.barrier.alloc {arrive_count = 1 : i32, init_polarity = 0 : i32, num_barriers = 1 : i32} : !ttg.memdesc<1x1xi64, #shared, #smem, mutable>
    tt.call @sink() : () -> ()
    %b = tle.barrier.alloc {arrive_count = 1 : i32, init_polarity = 0 : i32, num_barriers = 1 : i32} : !ttg.memdesc<1x1xi64, #shared, #smem, mutable>
    %as = ttg.memdesc_index %a[%c0] : !ttg.memdesc<1x1xi64, #shared, #smem, mutable> -> !ttg.memdesc<1xi64, #slot, #smem, mutable>
    tle.barrier.wait %as, %c0 {backend = "mbarrier", named_id = 1 : i32, named_num_threads = 128 : i32} : !ttg.memdesc<1xi64, #slot, #smem, mutable>
    %bs = ttg.memdesc_index %b[%c0] : !ttg.memdesc<1x1xi64, #shared, #smem, mutable> -> !ttg.memdesc<1xi64, #slot, #smem, mutable>
    tle.barrier.wait %bs, %c0 {backend = "mbarrier", named_id = 1 : i32, named_num_threads = 128 : i32} : !ttg.memdesc<1xi64, #slot, #smem, mutable>
    tt.return
  }
}

// -----
#shared = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [1, 0]}>
#slot = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>
#smem = #ttg.shared_memory
// CHECK-LABEL: tt.func @store_gap
// CHECK-COUNT-4: gpu.barrier
// CHECK-NOT: gpu.barrier
// CHECK: tt.return
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 8 : i32} {
  tt.func @store_gap(%cond: i1, %ptr: !tt.ptr<i32>) {
    %c0 = arith.constant 0 : i32
    %a = tle.barrier.alloc {arrive_count = 1 : i32, init_polarity = 0 : i32, num_barriers = 1 : i32} : !ttg.memdesc<1x1xi64, #shared, #smem, mutable>
    tt.store %ptr, %c0 : !tt.ptr<i32>
    %b = tle.barrier.alloc {arrive_count = 1 : i32, init_polarity = 0 : i32, num_barriers = 1 : i32} : !ttg.memdesc<1x1xi64, #shared, #smem, mutable>
    %as = ttg.memdesc_index %a[%c0] : !ttg.memdesc<1x1xi64, #shared, #smem, mutable> -> !ttg.memdesc<1xi64, #slot, #smem, mutable>
    tle.barrier.wait %as, %c0 {backend = "mbarrier", named_id = 1 : i32, named_num_threads = 128 : i32} : !ttg.memdesc<1xi64, #slot, #smem, mutable>
    %bs = ttg.memdesc_index %b[%c0] : !ttg.memdesc<1x1xi64, #shared, #smem, mutable> -> !ttg.memdesc<1xi64, #slot, #smem, mutable>
    tle.barrier.wait %bs, %c0 {backend = "mbarrier", named_id = 1 : i32, named_num_threads = 128 : i32} : !ttg.memdesc<1xi64, #slot, #smem, mutable>
    tt.return
  }
}

// -----
#shared = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [1, 0]}>
#slot = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>
#smem = #ttg.shared_memory
// CHECK-LABEL: tt.func @control_gap
// CHECK-COUNT-4: gpu.barrier
// CHECK-NOT: gpu.barrier
// CHECK: tt.return
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 8 : i32} {
  tt.func @control_gap(%cond: i1) {
    %c0 = arith.constant 0 : i32
    %a = tle.barrier.alloc {arrive_count = 1 : i32, init_polarity = 0 : i32, num_barriers = 1 : i32} : !ttg.memdesc<1x1xi64, #shared, #smem, mutable>
    scf.if %cond {
      scf.yield
    }
    %b = tle.barrier.alloc {arrive_count = 1 : i32, init_polarity = 0 : i32, num_barriers = 1 : i32} : !ttg.memdesc<1x1xi64, #shared, #smem, mutable>
    %as = ttg.memdesc_index %a[%c0] : !ttg.memdesc<1x1xi64, #shared, #smem, mutable> -> !ttg.memdesc<1xi64, #slot, #smem, mutable>
    tle.barrier.wait %as, %c0 {backend = "mbarrier", named_id = 1 : i32, named_num_threads = 128 : i32} : !ttg.memdesc<1xi64, #slot, #smem, mutable>
    %bs = ttg.memdesc_index %b[%c0] : !ttg.memdesc<1x1xi64, #shared, #smem, mutable> -> !ttg.memdesc<1xi64, #slot, #smem, mutable>
    tle.barrier.wait %bs, %c0 {backend = "mbarrier", named_id = 1 : i32, named_num_threads = 128 : i32} : !ttg.memdesc<1xi64, #slot, #smem, mutable>
    tt.return
  }
}

// -----
#shared = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [1, 0]}>
#slot = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>
#smem = #ttg.shared_memory
// CHECK-LABEL: tt.func @nested_loop_domain
// CHECK-COUNT-4: gpu.barrier
// CHECK-NOT: gpu.barrier
// CHECK: tt.return
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 8 : i32} {
  tt.func @nested_loop_domain(%cond: i1) {
    %c0 = arith.constant 0 : i32
    %c1 = arith.constant 1 : i32
    scf.for %i = %c0 to %c1 step %c1 : i32 {
      %a = tle.barrier.alloc {arrive_count = 1 : i32, init_polarity = 0 : i32, num_barriers = 1 : i32} : !ttg.memdesc<1x1xi64, #shared, #smem, mutable>
      %b = tle.barrier.alloc {arrive_count = 1 : i32, init_polarity = 0 : i32, num_barriers = 1 : i32} : !ttg.memdesc<1x1xi64, #shared, #smem, mutable>
      %as = ttg.memdesc_index %a[%c0] : !ttg.memdesc<1x1xi64, #shared, #smem, mutable> -> !ttg.memdesc<1xi64, #slot, #smem, mutable>
      tle.barrier.wait %as, %c0 {backend = "mbarrier", named_id = 1 : i32, named_num_threads = 128 : i32} : !ttg.memdesc<1xi64, #slot, #smem, mutable>
      %bs = ttg.memdesc_index %b[%c0] : !ttg.memdesc<1x1xi64, #shared, #smem, mutable> -> !ttg.memdesc<1xi64, #slot, #smem, mutable>
      tle.barrier.wait %bs, %c0 {backend = "mbarrier", named_id = 1 : i32, named_num_threads = 128 : i32} : !ttg.memdesc<1xi64, #slot, #smem, mutable>
      scf.yield
    }
    tt.return
  }
}

// -----
#shared = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [1, 0]}>
#slot = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>
#smem = #ttg.shared_memory
// CHECK-LABEL: tt.func @ws_default_domain
// CHECK-COUNT-4: gpu.barrier
// CHECK-NOT: gpu.barrier
// CHECK: tt.return
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 8 : i32} {
  tt.func @ws_default_domain(%cond: i1) {
    %c0 = arith.constant 0 : i32
    ttg.warp_specialize() default {
      %a = tle.barrier.alloc {arrive_count = 1 : i32, init_polarity = 0 : i32, num_barriers = 1 : i32} : !ttg.memdesc<1x1xi64, #shared, #smem, mutable>
      %b = tle.barrier.alloc {arrive_count = 1 : i32, init_polarity = 0 : i32, num_barriers = 1 : i32} : !ttg.memdesc<1x1xi64, #shared, #smem, mutable>
      %as = ttg.memdesc_index %a[%c0] : !ttg.memdesc<1x1xi64, #shared, #smem, mutable> -> !ttg.memdesc<1xi64, #slot, #smem, mutable>
      tle.barrier.wait %as, %c0 {backend = "mbarrier", named_id = 1 : i32, named_num_threads = 128 : i32} : !ttg.memdesc<1xi64, #slot, #smem, mutable>
      %bs = ttg.memdesc_index %b[%c0] : !ttg.memdesc<1x1xi64, #shared, #smem, mutable> -> !ttg.memdesc<1xi64, #slot, #smem, mutable>
      tle.barrier.wait %bs, %c0 {backend = "mbarrier", named_id = 1 : i32, named_num_threads = 128 : i32} : !ttg.memdesc<1xi64, #slot, #smem, mutable>
      ttg.warp_yield
    } partition0() num_warps(1) {
      ttg.warp_return
    } : () -> ()
    tt.return
  }
}

// -----
#shared = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [1, 0]}>
#slot = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>
#smem = #ttg.shared_memory
// CHECK-LABEL: tt.func @ws_worker_domain
// CHECK-COUNT-4: gpu.barrier
// CHECK-NOT: gpu.barrier
// CHECK: tt.return
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 8 : i32} {
  tt.func @ws_worker_domain(%cond: i1) {
    %c0 = arith.constant 0 : i32
    ttg.warp_specialize() default {
      ttg.warp_yield
    } partition0() num_warps(1) {
      %c0_worker = arith.constant 0 : i32
      %a = tle.barrier.alloc {arrive_count = 1 : i32, init_polarity = 0 : i32, num_barriers = 1 : i32} : !ttg.memdesc<1x1xi64, #shared, #smem, mutable>
      %b = tle.barrier.alloc {arrive_count = 1 : i32, init_polarity = 0 : i32, num_barriers = 1 : i32} : !ttg.memdesc<1x1xi64, #shared, #smem, mutable>
      %as = ttg.memdesc_index %a[%c0_worker] : !ttg.memdesc<1x1xi64, #shared, #smem, mutable> -> !ttg.memdesc<1xi64, #slot, #smem, mutable>
      tle.barrier.wait %as, %c0_worker {backend = "mbarrier", named_id = 1 : i32, named_num_threads = 128 : i32} : !ttg.memdesc<1xi64, #slot, #smem, mutable>
      %bs = ttg.memdesc_index %b[%c0_worker] : !ttg.memdesc<1x1xi64, #shared, #smem, mutable> -> !ttg.memdesc<1xi64, #slot, #smem, mutable>
      tle.barrier.wait %bs, %c0_worker {backend = "mbarrier", named_id = 1 : i32, named_num_threads = 128 : i32} : !ttg.memdesc<1xi64, #slot, #smem, mutable>
      ttg.warp_return
    } : () -> ()
    tt.return
  }
}
