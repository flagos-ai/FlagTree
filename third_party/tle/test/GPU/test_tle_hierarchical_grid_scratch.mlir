// RUN: triton-opt %s --tritongpu-global-scratch-memory-allocation | FileCheck %s
// Fixed8 leaf slots plus root:1152 bytes/group, all128-byte aligned.
// CHECK: ttg.global_scratch_memory_size = 39424 : i32
module attributes {"ttg.num-warps" = 8 : i32, "ttg.target" = "cuda:90"} {
  // CHECK-LABEL: tt.func private @leaf
  // CHECK-SAME: ttg.global_scratch_memory_alignment = 128 : i32
  // CHECK-SAME: ttg.global_scratch_memory_size = 1152 : i32
  tt.func private @leaf() attributes {noinline = true} {
    "tle.distributed_barrier"() <{group_kind = "grid"}> {tle.grid_barrier_protocol = "hierarchical8"} : () -> ()
    tt.return
  }
  // CHECK-LABEL: tt.func private @role
  // CHECK-SAME: ttg.global_scratch_memory_size = 19712 : i32
  tt.func private @role() attributes {noinline = true} {
    // CHECK: ttg.global_scratch_memory_offset = 0 : i32
    "tle.distributed_barrier"() <{group_kind = "grid"}> : () -> ()
    // CHECK: tt.call @leaf() {ttg.global_scratch_memory_offset = 128 : i32}
    tt.call @leaf() : () -> ()
    // CHECK: ttg.global_scratch_memory_offset = 1280 : i32
    "tle.distributed_barrier"() <{group_kind = "grid_axis", group_rank = 1 : i32, group_shape = array<i32: 8>, group_axes = array<i32: 0>, group_domain_shape = array<i32: 8, 16>}> {tle.grid_barrier_protocol = "hierarchical8"} : () -> ()
    tt.return
  }
  tt.func public @kernel() {
    tt.call @role() : () -> ()
    // CHECK: tt.call @role() {ttg.global_scratch_memory_offset = 19712 : i32}
    tt.call @role() : () -> ()
    tt.return
  }
}
