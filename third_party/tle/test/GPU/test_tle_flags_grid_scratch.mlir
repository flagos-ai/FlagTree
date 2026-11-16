// RUN: triton-opt %s --tritongpu-global-scratch-memory-allocation | FileCheck %s
// CHECK: tle.flags_launch_contract =
// CHECK-SAME: ttg.global_scratch_memory_size = 5376 : i32
module attributes {"ttg.num-warps" = 8 : i32, "ttg.target" = "cuda:90"} {
  // CHECK-LABEL: tt.func private @leaf
  // CHECK-SAME: ttg.global_scratch_memory_alignment = 128 : i32
  // CHECK-SAME: ttg.global_scratch_memory_size = 512 : i32
  tt.func private @leaf() attributes {noinline = true} {
    "tle.distributed_barrier"() <{group_kind = "grid"}> {tle.grid_barrier_protocol = "flags32", tle.flags_domain = array<i32: 8, 16>, tle.flags_launch_grid = array<i32: 16, 8, 1>} : () -> ()
    tt.return
  }
  // CHECK-LABEL: tt.func private @role
  // CHECK-SAME: ttg.global_scratch_memory_size = 2688 : i32
  tt.func private @role() attributes {noinline = true} {
    "tle.distributed_barrier"() <{group_kind = "grid"}> : () -> ()
    // CHECK: tt.call @leaf() {ttg.global_scratch_memory_offset = 128 : i32}
    tt.call @leaf() : () -> ()
    // CHECK: ttg.global_scratch_memory_offset = 640 : i32
    "tle.distributed_barrier"() <{group_kind = "grid_axis", group_rank = 1 : i32, group_shape = array<i32: 8>, group_axes = array<i32: 0>, group_domain_shape = array<i32: 8, 16>}> {tle.grid_barrier_protocol = "flags32", tle.flags_domain = array<i32: 8, 16>, tle.flags_launch_grid = array<i32: 16, 8, 1>} : () -> ()
    tt.return
  }
  tt.func public @kernel() {
    tt.call @role() : () -> ()
    // CHECK: tt.call @role() {ttg.global_scratch_memory_offset = 2688 : i32}
    tt.call @role() : () -> ()
    tt.return
  }
}
