// RUN: triton-opt %s -triton-tle-lower-barriers -triton-tle-allocate-named-barriers -split-input-file | FileCheck %s

#slot = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>
#smem = #ttg.shared_memory

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32} {
  // CHECK-LABEL: tt.func @lower_and_allocate_tle_virtual_named_barriers
  tt.func @lower_and_allocate_tle_virtual_named_barriers(%slot: !ttg.memdesc<1xi64, #slot, #smem, mutable>) {
    tle.barrier.wait %slot {backend = "named", named_id = 16 : i32, named_num_threads = 256 : i32} : !ttg.memdesc<1xi64, #slot, #smem, mutable>
    tle.barrier.arrive %slot {arrive_count = 1 : i32, backend = "named", named_id = 17 : i32, named_num_threads = 256 : i32} : !ttg.memdesc<1xi64, #slot, #smem, mutable>
    tt.return
  }
}

// CHECK-DAG: %[[ID1:.+]] = arith.constant 1 : i32
// CHECK-DAG: %[[THREADS:.+]] = arith.constant 256 : i32
// CHECK: ttng.wait_barrier_named %[[ID1]], %[[THREADS]]
// CHECK: %[[ID2:.+]] = arith.constant 2 : i32
// CHECK: ttng.arrive_barrier_named %[[ID2]], {{.*}}
// CHECK-NOT: tle.barrier
