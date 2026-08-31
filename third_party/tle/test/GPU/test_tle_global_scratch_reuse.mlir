// RUN: triton-opt %s --tritongpu-global-scratch-memory-allocation | FileCheck %s

// CHECK: module attributes
// CHECK-SAME: ttg.global_scratch_memory_alignment = 4 : i32
// CHECK-SAME: ttg.global_scratch_memory_size = 68 : i32
// CHECK-SAME: ttg.global_scratch_reset_per_launch = 0 : i32

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:90", "ttg.threads-per-warp" = 32 : i32} {
  // CHECK-LABEL: tt.func private @collective_helper
  // CHECK-SAME: tle.grid_barrier_scratch_only
  // CHECK-SAME: ttg.global_scratch_memory_size = 68 : i32
  tt.func private @collective_helper() {
    // CHECK: tle.distributed_barrier {group_kind = "grid", ttg.global_scratch_memory_offset = 0 : i32}
    tle.distributed_barrier {group_kind = "grid"}
    // CHECK: tle.distributed_barrier
    // CHECK-SAME: group_kind = "grid_axis_group"
    // CHECK-SAME: ttg.global_scratch_memory_offset = 4 : i32
    tle.distributed_barrier {group_axes = array<i32: 0>, group_domain_shape = array<i32: 8, 16>, group_kind = "grid_axis_group", group_rank = 1 : i32, group_shape = array<i32: 8>}
    // CHECK: tle.distributed_barrier {group_kind = "grid", ttg.global_scratch_memory_offset = 0 : i32}
    tle.distributed_barrier {group_kind = "grid"}
    tt.return
  }

  // CHECK-LABEL: tt.func public @reuse_collective_scratch
  // CHECK-SAME: tle.grid_barrier_scratch_only
  // CHECK-SAME: ttg.global_scratch_memory_size = 68 : i32
  tt.func public @reuse_collective_scratch() {
    // CHECK: tle.distributed_barrier {group_kind = "grid", ttg.global_scratch_memory_offset = 0 : i32}
    tle.distributed_barrier {group_kind = "grid"}
    // CHECK: tt.call @collective_helper() {ttg.global_scratch_memory_offset = 0 : i32}
    tt.call @collective_helper() : () -> ()
    // CHECK: tle.distributed_barrier {group_kind = "grid", ttg.global_scratch_memory_offset = 0 : i32}
    tle.distributed_barrier {group_kind = "grid"}
    // CHECK: tt.call @collective_helper() {ttg.global_scratch_memory_offset = 0 : i32}
    tt.call @collective_helper() : () -> ()
    tt.return
  }
}
