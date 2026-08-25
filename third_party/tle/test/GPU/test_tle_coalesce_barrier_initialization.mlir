// RUN: triton-opt %s -split-input-file --triton-tle-coalesce-barrier-initialization='min-barrier-count=2' | FileCheck %s

#shared = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>
#smem = #ttg.shared_memory

module attributes {"ttg.num-warps" = 1 : i32, "ttg.threads-per-warp" = 2 : i32} {
  // CHECK-LABEL: tt.func @coalesce_static_barriers
  // CHECK: tle.init_barrier_group
  // CHECK-SAME: counts = array<i32: 1, 256>
  // CHECK-SAME: offsets = array<i32: 0, 16>
  // CHECK-SAME: worker_count = 2 : i32
  // CHECK-NOT: ttng.init_barrier
  tt.func @coalesce_static_barriers() {
    %barrier0 = ttg.local_alloc {allocation.offset = 0 : i32} : () -> !ttg.memdesc<1xi64, #shared, #smem, mutable>
    ttng.init_barrier %barrier0, 1 : !ttg.memdesc<1xi64, #shared, #smem, mutable>
    %c1 = arith.constant 1 : i32
    %barriers1 = ttg.local_alloc {allocation.offset = 8 : i32} : () -> !ttg.memdesc<2x1xi64, #shared, #smem, mutable>
    %barrier1 = ttg.memdesc_index %barriers1[%c1] : !ttg.memdesc<2x1xi64, #shared, #smem, mutable> -> !ttg.memdesc<1xi64, #shared, #smem, mutable>
    ttng.init_barrier %barrier1, 256 : !ttg.memdesc<1xi64, #shared, #smem, mutable>
    tt.return
  }
}

// -----

#shared = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>
#smem = #ttg.shared_memory

module attributes {"ttg.num-warps" = 1 : i32, "ttg.threads-per-warp" = 2 : i32} {
  // CHECK-LABEL: tt.func @keep_small_barrier_set
  // CHECK: ttng.init_barrier
  // CHECK-NOT: tle.init_barrier_group
  tt.func @keep_small_barrier_set() {
    %barrier = ttg.local_alloc {allocation.offset = 0 : i32} : () -> !ttg.memdesc<1xi64, #shared, #smem, mutable>
    ttng.init_barrier %barrier, 1 : !ttg.memdesc<1xi64, #shared, #smem, mutable>
    tt.return
  }
}
