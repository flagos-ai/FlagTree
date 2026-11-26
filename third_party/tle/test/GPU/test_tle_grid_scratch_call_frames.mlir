// RUN: triton-opt %s --tritongpu-global-scratch-memory-allocation | FileCheck %s

// Grid counters are allocated in function frames before LLVM call lowering,
// including partial axis groups; each static call owns its own frame.
// CHECK: ttg.global_scratch_memory_size = 264 : i32
module attributes {"ttg.num-warps" = 8 : i32} {
  // CHECK-LABEL: tt.func private @role
  // CHECK-SAME: ttg.global_scratch_memory_size = 132 : i32
  tt.func private @role() attributes {noinline = true} {
    // CHECK: ttg.global_scratch_memory_offset = 0 : i32
    "tle.distributed_barrier"() <{group_kind = "grid"}> : () -> ()
    // Domain 8x16 partitioned into groups of 4 along axis 1 has 32 counters.
    // CHECK: ttg.global_scratch_memory_offset = 4 : i32
    "tle.distributed_barrier"() <{group_kind = "grid_axis", group_rank = 1 : i32, group_shape = array<i32: 4>, group_axes = array<i32: 1>, group_domain_shape = array<i32: 8, 16>}> : () -> ()
    tt.return
  }
  // CHECK-LABEL: tt.func public @kernel
  tt.func public @kernel() {
    // CHECK: tt.call @role() {ttg.global_scratch_memory_offset = 0 : i32}
    tt.call @role() : () -> ()
    // CHECK: tt.call @role() {ttg.global_scratch_memory_offset = 132 : i32}
    tt.call @role() : () -> ()
    tt.return
  }
}
