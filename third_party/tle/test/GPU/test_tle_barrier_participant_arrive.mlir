// Copyright 2026 FlagOS Contributors
// SPDX-License-Identifier: MIT
// RUN: triton-opt %s --triton-tle-lower-barriers -split-input-file -verify-each -verify-diagnostics | FileCheck %s --check-prefix=POS
#shared = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [1, 0]}>
#slot = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>
#smem = #ttg.shared_memory
// POS-LABEL: tt.func @participant_256
// POS: ttng.init_barrier {{.*}}, 256
// POS: ttng.arrive_barrier {{.*}}, 256
// POS-SAME: participant_arrive = true
// POS-SAME: release_fence = true
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 8 : i32, "ttg.threads-per-warp" = 32 : i32} {
  tt.func @participant_256() {
    %c0 = arith.constant 0 : i32
    %b = tle.barrier.alloc {arrive_count = 256 : i32, init_polarity = 1 : i32, num_barriers = 1 : i32} : !ttg.memdesc<1x1xi64, #shared, #smem, mutable>
    %s = ttg.memdesc_index %b[%c0] : !ttg.memdesc<1x1xi64, #shared, #smem, mutable> -> !ttg.memdesc<1xi64, #slot, #smem, mutable>
    tle.barrier.arrive %s, %c0 {arrive_count = 1 : i32, backend = "mbarrier", named_id = 0 : i32, named_num_threads = 0 : i32, participant_count = 256 : i32} : !ttg.memdesc<1xi64, #slot, #smem, mutable>
    tt.return
  }
}

// -----
#shared = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [1, 0]}>
#slot = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>
#smem = #ttg.shared_memory
// POS-LABEL: tt.func @ws_default_exact256
// POS: ttng.arrive_barrier {{.*}}, 256
// POS-SAME: participant_arrive = true
// POS-SAME: release_fence = true
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 8 : i32, "ttg.threads-per-warp" = 32 : i32} {
  tt.func @ws_default_exact256() {
    %c0 = arith.constant 0 : i32
    %b = tle.barrier.alloc {arrive_count = 256 : i32, init_polarity = 1 : i32, num_barriers = 1 : i32} : !ttg.memdesc<1x1xi64, #shared, #smem, mutable>
    %s = ttg.memdesc_index %b[%c0] : !ttg.memdesc<1x1xi64, #shared, #smem, mutable> -> !ttg.memdesc<1xi64, #slot, #smem, mutable>
    ttg.warp_specialize(%b) default {
      %d = ttg.memdesc_index %b[%c0] : !ttg.memdesc<1x1xi64, #shared, #smem, mutable> -> !ttg.memdesc<1xi64, #slot, #smem, mutable>
      tle.barrier.arrive %d, %c0 {arrive_count = 1 : i32, backend = "mbarrier", named_id = 0 : i32, named_num_threads = 256 : i32, participant_count = 256 : i32} : !ttg.memdesc<1xi64, #slot, #smem, mutable>
      ttg.warp_yield
    } partition0(%capture: !ttg.memdesc<1x1xi64, #shared, #smem, mutable>) num_warps(1) {
      ttg.warp_return
    } : (!ttg.memdesc<1x1xi64, #shared, #smem, mutable>) -> ()
    tt.return
  }
}

// -----
#shared = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [1, 0]}>
#slot = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>
#smem = #ttg.shared_memory
// POS-LABEL: tt.func @ws_worker_exact32
// POS: ttng.arrive_barrier {{.*}}, 32
// POS-SAME: participant_arrive = true
// POS-SAME: release_fence = true
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 8 : i32, "ttg.threads-per-warp" = 32 : i32} {
  tt.func @ws_worker_exact32() {
    %c0 = arith.constant 0 : i32
    %b = tle.barrier.alloc {arrive_count = 32 : i32, init_polarity = 1 : i32, num_barriers = 1 : i32} : !ttg.memdesc<1x1xi64, #shared, #smem, mutable>
    %s = ttg.memdesc_index %b[%c0] : !ttg.memdesc<1x1xi64, #shared, #smem, mutable> -> !ttg.memdesc<1xi64, #slot, #smem, mutable>
    ttg.warp_specialize(%b) default {
      ttg.warp_yield
    } partition0(%capture: !ttg.memdesc<1x1xi64, #shared, #smem, mutable>) num_warps(1) {
      %c0w = arith.constant 0 : i32
      %w = ttg.memdesc_index %capture[%c0w] : !ttg.memdesc<1x1xi64, #shared, #smem, mutable> -> !ttg.memdesc<1xi64, #slot, #smem, mutable>
    tle.barrier.arrive %w, %c0w {arrive_count = 1 : i32, backend = "mbarrier", named_id = 0 : i32, named_num_threads = 256 : i32, participant_count = 32 : i32} : !ttg.memdesc<1xi64, #slot, #smem, mutable>
      ttg.warp_return
    } : (!ttg.memdesc<1x1xi64, #shared, #smem, mutable>) -> ()
    tt.return
  }
}

// -----
#shared = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [1, 0]}>
#slot = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>
#smem = #ttg.shared_memory
// POS-LABEL: tt.func @uniform_loop_slot
// POS: ttng.arrive_barrier {{.*}}, 256
// POS-SAME: participant_arrive = true
// POS-SAME: release_fence = true
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 8 : i32, "ttg.threads-per-warp" = 32 : i32} {
  tt.func @uniform_loop_slot() {
    %c0 = arith.constant 0 : i32
    %c1 = arith.constant 1 : i32
    %c2 = arith.constant 2 : i32
    %c3 = arith.constant 3 : i32
    %c16 = arith.constant 16 : i32
    %b = tle.barrier.alloc {arrive_count = 256 : i32, init_polarity = 1 : i32, num_barriers = 3 : i32} : !ttg.memdesc<3x1xi64, #shared, #smem, mutable>
    scf.for %ko = %c0 to %c16 step %c1 : i32 {
      %slot = arith.remsi %ko, %c3 : i32
      %g = arith.divsi %ko, %c3 : i32
      %phase = arith.remsi %g, %c2 : i32
      %s = ttg.memdesc_index %b[%slot] : !ttg.memdesc<3x1xi64, #shared, #smem, mutable> -> !ttg.memdesc<1xi64, #slot, #smem, mutable>
      tle.barrier.arrive %s, %phase {arrive_count = 1 : i32, backend = "mbarrier", named_id = 0 : i32, named_num_threads = 0 : i32, participant_count = 256 : i32} : !ttg.memdesc<1xi64, #slot, #smem, mutable>
    }
    tt.return
  }
}

// -----
#shared = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [1, 0]}>
#slot = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>
#smem = #ttg.shared_memory
// POS-LABEL: tt.func @legacy_elected256
// POS: ttng.arrive_barrier {{.*}}, 256
// POS-NOT: participant_arrive = true
// POS-NOT: release_fence = true
// POS: tt.return
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 8 : i32, "ttg.threads-per-warp" = 32 : i32} {
  tt.func @legacy_elected256() {
    %c0 = arith.constant 0 : i32
    %b = tle.barrier.alloc {arrive_count = 256 : i32, init_polarity = 1 : i32, num_barriers = 1 : i32} : !ttg.memdesc<1x1xi64, #shared, #smem, mutable>
    %s = ttg.memdesc_index %b[%c0] : !ttg.memdesc<1x1xi64, #shared, #smem, mutable> -> !ttg.memdesc<1xi64, #slot, #smem, mutable>
    tle.barrier.arrive %s, %c0 {arrive_count = 256 : i32, backend = "mbarrier", named_id = 0 : i32, named_num_threads = 256 : i32} : !ttg.memdesc<1xi64, #slot, #smem, mutable>
    tt.return
  }
}

// -----
#shared = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [1, 0]}>
#slot = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>
#smem = #ttg.shared_memory
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 8 : i32, "ttg.threads-per-warp" = 32 : i32} {
  tt.func @zero_count() {
    %c0 = arith.constant 0 : i32
    %b = tle.barrier.alloc {arrive_count = 256 : i32, init_polarity = 1 : i32, num_barriers = 1 : i32} : !ttg.memdesc<1x1xi64, #shared, #smem, mutable>
    %s = ttg.memdesc_index %b[%c0] : !ttg.memdesc<1x1xi64, #shared, #smem, mutable> -> !ttg.memdesc<1xi64, #slot, #smem, mutable>
    // expected-error @+1 {{participant_count must be positive}}
    tle.barrier.arrive %s, %c0 {arrive_count = 1 : i32, backend = "mbarrier", named_id = 0 : i32, named_num_threads = 256 : i32, participant_count = 0 : i32} : !ttg.memdesc<1xi64, #slot, #smem, mutable>
    tt.return
  }
}

// -----
#shared = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [1, 0]}>
#slot = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>
#smem = #ttg.shared_memory
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 8 : i32, "ttg.threads-per-warp" = 32 : i32} {
  tt.func @named() {
    %c0 = arith.constant 0 : i32
    %b = tle.barrier.alloc {arrive_count = 256 : i32, init_polarity = 1 : i32, num_barriers = 1 : i32} : !ttg.memdesc<1x1xi64, #shared, #smem, mutable>
    %s = ttg.memdesc_index %b[%c0] : !ttg.memdesc<1x1xi64, #shared, #smem, mutable> -> !ttg.memdesc<1xi64, #slot, #smem, mutable>
    // expected-error @+1 {{participant arrival requires explicit mbarrier phase}}
    tle.barrier.arrive %s {arrive_count = 1 : i32, backend = "named", named_id = 1 : i32, named_num_threads = 256 : i32, participant_count = 256 : i32} : !ttg.memdesc<1xi64, #slot, #smem, mutable>
    tt.return
  }
}

// -----
#shared = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [1, 0]}>
#slot = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>
#smem = #ttg.shared_memory
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 8 : i32, "ttg.threads-per-warp" = 32 : i32} {
  tt.func @missing_phase() {
    %c0 = arith.constant 0 : i32
    %b = tle.barrier.alloc {arrive_count = 256 : i32, init_polarity = 1 : i32, num_barriers = 1 : i32} : !ttg.memdesc<1x1xi64, #shared, #smem, mutable>
    %s = ttg.memdesc_index %b[%c0] : !ttg.memdesc<1x1xi64, #shared, #smem, mutable> -> !ttg.memdesc<1xi64, #slot, #smem, mutable>
    // expected-error @+1 {{participant arrival requires explicit mbarrier phase}}
    tle.barrier.arrive %s {arrive_count = 1 : i32, backend = "mbarrier", named_id = 0 : i32, named_num_threads = 256 : i32, participant_count = 256 : i32} : !ttg.memdesc<1xi64, #slot, #smem, mutable>
    tt.return
  }
}

// -----
#shared = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [1, 0]}>
#slot = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>
#smem = #ttg.shared_memory
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 8 : i32, "ttg.threads-per-warp" = 32 : i32} {
  tt.func @elected_multicredit() {
    %c0 = arith.constant 0 : i32
    %b = tle.barrier.alloc {arrive_count = 256 : i32, init_polarity = 1 : i32, num_barriers = 1 : i32} : !ttg.memdesc<1x1xi64, #shared, #smem, mutable>
    %s = ttg.memdesc_index %b[%c0] : !ttg.memdesc<1x1xi64, #shared, #smem, mutable> -> !ttg.memdesc<1xi64, #slot, #smem, mutable>
    // expected-error @+1 {{participant arrival requires explicit mbarrier phase}}
    tle.barrier.arrive %s, %c0 {arrive_count = 256 : i32, backend = "mbarrier", named_id = 0 : i32, named_num_threads = 256 : i32, participant_count = 256 : i32} : !ttg.memdesc<1xi64, #slot, #smem, mutable>
    tt.return
  }
}

// -----
#shared = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [1, 0]}>
#slot = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>
#smem = #ttg.shared_memory
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 8 : i32, "ttg.threads-per-warp" = 32 : i32} {
  tt.func @pending_mismatch() {
    %c0 = arith.constant 0 : i32
    %b = tle.barrier.alloc {arrive_count = 128 : i32, init_polarity = 1 : i32, num_barriers = 1 : i32} : !ttg.memdesc<1x1xi64, #shared, #smem, mutable>
    %s = ttg.memdesc_index %b[%c0] : !ttg.memdesc<1x1xi64, #shared, #smem, mutable> -> !ttg.memdesc<1xi64, #slot, #smem, mutable>
    // expected-error @+1 {{known non-TX non-named allocation}}
    tle.barrier.arrive %s, %c0 {arrive_count = 1 : i32, backend = "mbarrier", named_id = 0 : i32, named_num_threads = 256 : i32, participant_count = 256 : i32} : !ttg.memdesc<1xi64, #slot, #smem, mutable>
    tt.return
  }
}

// -----
#shared = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [1, 0]}>
#slot = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>
#smem = #ttg.shared_memory
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 8 : i32, "ttg.threads-per-warp" = 32 : i32} {
  tt.func @TX() {
    %c0 = arith.constant 0 : i32
    %b = tle.barrier.alloc {arrive_count = 256 : i32, init_polarity = 1 : i32, num_barriers = 1 : i32, expect_bytes = 128 : i32} : !ttg.memdesc<1x1xi64, #shared, #smem, mutable>
    %s = ttg.memdesc_index %b[%c0] : !ttg.memdesc<1x1xi64, #shared, #smem, mutable> -> !ttg.memdesc<1xi64, #slot, #smem, mutable>
    // expected-error @+1 {{known non-TX non-named allocation}}
    tle.barrier.arrive %s, %c0 {arrive_count = 1 : i32, backend = "mbarrier", named_id = 0 : i32, named_num_threads = 256 : i32, participant_count = 256 : i32} : !ttg.memdesc<1xi64, #slot, #smem, mutable>
    tt.return
  }
}

// -----
#shared = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [1, 0]}>
#slot = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>
#smem = #ttg.shared_memory
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 8 : i32, "ttg.threads-per-warp" = 32 : i32} {
  tt.func @physical384() {
    %c0 = arith.constant 0 : i32
    %b = tle.barrier.alloc {arrive_count = 384 : i32, init_polarity = 1 : i32, num_barriers = 1 : i32} : !ttg.memdesc<1x1xi64, #shared, #smem, mutable>
    %s = ttg.memdesc_index %b[%c0] : !ttg.memdesc<1x1xi64, #shared, #smem, mutable> -> !ttg.memdesc<1xi64, #slot, #smem, mutable>
    // expected-error @+1 {{exact logical execution domain}}
    tle.barrier.arrive %s, %c0 {arrive_count = 1 : i32, backend = "mbarrier", named_id = 0 : i32, named_num_threads = 256 : i32, participant_count = 384 : i32} : !ttg.memdesc<1xi64, #slot, #smem, mutable>
    tt.return
  }
}

// -----
#shared = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [1, 0]}>
#slot = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>
#smem = #ttg.shared_memory
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 8 : i32, "ttg.threads-per-warp" = 32 : i32} {
  tt.func @partial128() {
    %c0 = arith.constant 0 : i32
    %b = tle.barrier.alloc {arrive_count = 128 : i32, init_polarity = 1 : i32, num_barriers = 1 : i32} : !ttg.memdesc<1x1xi64, #shared, #smem, mutable>
    %s = ttg.memdesc_index %b[%c0] : !ttg.memdesc<1x1xi64, #shared, #smem, mutable> -> !ttg.memdesc<1xi64, #slot, #smem, mutable>
    // expected-error @+1 {{exact logical execution domain}}
    tle.barrier.arrive %s, %c0 {arrive_count = 1 : i32, backend = "mbarrier", named_id = 0 : i32, named_num_threads = 256 : i32, participant_count = 128 : i32} : !ttg.memdesc<1xi64, #slot, #smem, mutable>
    tt.return
  }
}

// -----
#shared = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [1, 0]}>
#slot = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>
#smem = #ttg.shared_memory
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 8 : i32, "ttg.threads-per-warp" = 32 : i32} {
  tt.func @public_unknown(%s: !ttg.memdesc<1xi64, #slot, #smem, mutable>) {
    %c0 = arith.constant 0 : i32
    // expected-error @+1 {{known non-TX non-named allocation}}
    tle.barrier.arrive %s, %c0 {arrive_count = 1 : i32, backend = "mbarrier", named_id = 0 : i32, named_num_threads = 256 : i32, participant_count = 256 : i32} : !ttg.memdesc<1xi64, #slot, #smem, mutable>
    tt.return
  }
}

// -----
#shared = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [1, 0]}>
#slot = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>
#smem = #ttg.shared_memory
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 8 : i32, "ttg.threads-per-warp" = 32 : i32} {
  tt.func @unknown_condition(%cond: i1) {
    %c0 = arith.constant 0 : i32
    %b = tle.barrier.alloc {arrive_count = 256 : i32, init_polarity = 1 : i32, num_barriers = 1 : i32} : !ttg.memdesc<1x1xi64, #shared, #smem, mutable>
    %s = ttg.memdesc_index %b[%c0] : !ttg.memdesc<1x1xi64, #shared, #smem, mutable> -> !ttg.memdesc<1xi64, #slot, #smem, mutable>
    scf.if %cond {
      // expected-error @+1 {{exact logical execution domain}}
    tle.barrier.arrive %s, %c0 {arrive_count = 1 : i32, backend = "mbarrier", named_id = 0 : i32, named_num_threads = 256 : i32, participant_count = 256 : i32} : !ttg.memdesc<1xi64, #slot, #smem, mutable>
    }
    tt.return
  }
}

// -----
#shared = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [1, 0]}>
#slot = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>
#smem = #ttg.shared_memory
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 8 : i32, "ttg.threads-per-warp" = 32 : i32} {
  tt.func @per_lane(%cond: i1) {
    %c0 = arith.constant 0 : i32
    %b = tle.barrier.alloc {arrive_count = 256 : i32, init_polarity = 1 : i32, num_barriers = 1 : i32} : !ttg.memdesc<1x1xi64, #shared, #smem, mutable>
    %s = ttg.memdesc_index %b[%c0] : !ttg.memdesc<1x1xi64, #shared, #smem, mutable> -> !ttg.memdesc<1xi64, #slot, #smem, mutable>
    %tid = tt.elementwise_inline_asm "mov.u32 $0, %tid.x;" {constraints = "=r", packed_element = 1 : i32, pure = true} -> i32
    %lane = arith.cmpi eq, %tid, %c0 : i32
    scf.if %lane {
      // expected-error @+1 {{exact logical execution domain}}
    tle.barrier.arrive %s, %c0 {arrive_count = 1 : i32, backend = "mbarrier", named_id = 0 : i32, named_num_threads = 256 : i32, participant_count = 256 : i32} : !ttg.memdesc<1xi64, #slot, #smem, mutable>
    }
    tt.return
  }
}

// -----
#shared = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [1, 0]}>
#slot = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>
#smem = #ttg.shared_memory
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 8 : i32, "ttg.threads-per-warp" = 32 : i32} {
  tt.func @duplicate() {
    %c0 = arith.constant 0 : i32
    %b = tle.barrier.alloc {arrive_count = 256 : i32, init_polarity = 1 : i32, num_barriers = 1 : i32} : !ttg.memdesc<1x1xi64, #shared, #smem, mutable>
    %s = ttg.memdesc_index %b[%c0] : !ttg.memdesc<1x1xi64, #shared, #smem, mutable> -> !ttg.memdesc<1xi64, #slot, #smem, mutable>
    tle.barrier.arrive %s, %c0 {arrive_count = 1 : i32, backend = "mbarrier", named_id = 0 : i32, named_num_threads = 256 : i32, participant_count = 256 : i32} : !ttg.memdesc<1xi64, #slot, #smem, mutable>
    // expected-error @+1 {{duplicate participant arrival}}
    tle.barrier.arrive %s, %c0 {arrive_count = 1 : i32, backend = "mbarrier", named_id = 0 : i32, named_num_threads = 256 : i32, participant_count = 256 : i32} : !ttg.memdesc<1xi64, #slot, #smem, mutable>
    tt.return
  }
}

// -----
#shared = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [1, 0]}>
#slot = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>
#smem = #ttg.shared_memory
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 8 : i32, "ttg.threads-per-warp" = 32 : i32} {
  tt.func @named_mixed() {
    %c0 = arith.constant 0 : i32
    %b = tle.barrier.alloc {arrive_count = 256 : i32, init_polarity = 1 : i32, num_barriers = 1 : i32} : !ttg.memdesc<1x1xi64, #shared, #smem, mutable>
    %s = ttg.memdesc_index %b[%c0] : !ttg.memdesc<1x1xi64, #shared, #smem, mutable> -> !ttg.memdesc<1xi64, #slot, #smem, mutable>
    tle.barrier.wait %s {backend = "named", named_id = 1 : i32, named_num_threads = 256 : i32} : !ttg.memdesc<1xi64, #slot, #smem, mutable>
    // expected-error @+1 {{known non-TX non-named allocation}}
    tle.barrier.arrive %s, %c0 {arrive_count = 1 : i32, backend = "mbarrier", named_id = 0 : i32, named_num_threads = 256 : i32, participant_count = 256 : i32} : !ttg.memdesc<1xi64, #slot, #smem, mutable>
    tt.return
  }
}

// -----
#shared = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [1, 0]}>
#slot = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>
#smem = #ttg.shared_memory
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 8 : i32, "ttg.threads-per-warp" = 32 : i32} {
  tt.func @unknown_nested_region() {
    %c0 = arith.constant 0 : i32
    %b = tle.barrier.alloc {arrive_count = 256 : i32, init_polarity = 1 : i32, num_barriers = 1 : i32} : !ttg.memdesc<1x1xi64, #shared, #smem, mutable>
    %s = ttg.memdesc_index %b[%c0] : !ttg.memdesc<1x1xi64, #shared, #smem, mutable> -> !ttg.memdesc<1xi64, #slot, #smem, mutable>
    scf.execute_region {
      // expected-error @+1 {{exact logical execution domain}}
    tle.barrier.arrive %s, %c0 {arrive_count = 1 : i32, backend = "mbarrier", named_id = 0 : i32, named_num_threads = 256 : i32, participant_count = 256 : i32} : !ttg.memdesc<1xi64, #slot, #smem, mutable>
      scf.yield
    }
    tt.return
  }
}

// -----
#shared = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [1, 0]}>
#slot = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>
#smem = #ttg.shared_memory
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 8 : i32, "ttg.threads-per-warp" = 32 : i32} {
  tt.func @wrong_worker_count() {
    %c0 = arith.constant 0 : i32
    %b = tle.barrier.alloc {arrive_count = 256 : i32, init_polarity = 1 : i32, num_barriers = 1 : i32} : !ttg.memdesc<1x1xi64, #shared, #smem, mutable>
    %s = ttg.memdesc_index %b[%c0] : !ttg.memdesc<1x1xi64, #shared, #smem, mutable> -> !ttg.memdesc<1xi64, #slot, #smem, mutable>
    ttg.warp_specialize(%b) default {
      ttg.warp_yield
    } partition0(%capture: !ttg.memdesc<1x1xi64, #shared, #smem, mutable>) num_warps(1) {
      %c0w = arith.constant 0 : i32
      %w = ttg.memdesc_index %capture[%c0w] : !ttg.memdesc<1x1xi64, #shared, #smem, mutable> -> !ttg.memdesc<1xi64, #slot, #smem, mutable>
      // expected-error @+1 {{exact logical execution domain}}
    tle.barrier.arrive %w, %c0w {arrive_count = 1 : i32, backend = "mbarrier", named_id = 0 : i32, named_num_threads = 256 : i32, participant_count = 256 : i32} : !ttg.memdesc<1xi64, #slot, #smem, mutable>
      ttg.warp_return
    } : (!ttg.memdesc<1x1xi64, #shared, #smem, mutable>) -> ()
    tt.return
  }
}
