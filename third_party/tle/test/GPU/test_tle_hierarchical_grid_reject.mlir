// RUN: triton-opt %s --split-input-file --tritongpu-global-scratch-memory-allocation --verify-diagnostics
module attributes {"ttg.num-warps" = 8 : i32} {
  tt.func public @domain_overflow() {
    // expected-error @+1 {{hierarchical8 domain size exceeds signed i32}}
    "tle.distributed_barrier"() <{group_kind = "grid_axis", group_rank = 1 : i32, group_shape = array<i32: 1>, group_axes = array<i32: 0>, group_domain_shape = array<i32: 65536, 65536>}> {tle.grid_barrier_protocol = "hierarchical8"} : () -> ()
    tt.return
  }
}
// -----
module attributes {"ttg.num-warps" = 8 : i32} {
  tt.func public @group_bytes_overflow() {
    // expected-error @+1 {{hierarchical8 scratch size exceeds signed i32}}
    "tle.distributed_barrier"() <{group_kind = "grid_axis", group_rank = 1 : i32, group_shape = array<i32: 1>, group_axes = array<i32: 0>, group_domain_shape = array<i32: 2000000>}> {tle.grid_barrier_protocol = "hierarchical8"} : () -> ()
    tt.return
  }
}
// -----
module attributes {"ttg.num-warps" = 8 : i32} {
  tt.func private @large_leaf() attributes {noinline = true} {
    "tle.distributed_barrier"() <{group_kind = "grid_axis", group_rank = 1 : i32, group_shape = array<i32: 1>, group_axes = array<i32: 0>, group_domain_shape = array<i32: 1000000>}> {tle.grid_barrier_protocol = "hierarchical8"} : () -> ()
    tt.return
  }
  tt.func public @nested_overflow() {
    tt.call @large_leaf() : () -> ()
    // expected-error @+1 {{hierarchical grid scratch accumulated frame exceeds signed i32}}
    tt.call @large_leaf() : () -> ()
    tt.return
  }
}
// -----
module attributes {"ttg.num-warps" = 8 : i32} {
  // expected-error @+1 {{hierarchical grid scratch has inconsistent preexisting frame metadata}}
  tt.func public @stale_frame() attributes {ttg.global_scratch_memory_size = 4 : i32, ttg.global_scratch_memory_alignment = 4 : i32} {
    "tle.distributed_barrier"() <{group_kind = "grid"}> {tle.grid_barrier_protocol = "hierarchical8"} : () -> ()
    tt.return
  }
}
// -----
module attributes {"ttg.num-warps" = 8 : i32} {
  // expected-error @+1 {{hierarchical grid scratch does not support recursive calls}}
  tt.func public @recursive() {
    "tle.distributed_barrier"() <{group_kind = "grid"}> {tle.grid_barrier_protocol = "hierarchical8"} : () -> ()
    tt.call @recursive() : () -> ()
    tt.return
  }
}
// -----
module attributes {"ttg.num-warps" = 8 : i32, "ttg.target" = "hip:gfx942"} {
  tt.func public @unsupported_backend() {
    // expected-error @+1 {{hierarchical8 requires the NVIDIA backend}}
    "tle.distributed_barrier"() <{group_kind = "grid"}> {tle.grid_barrier_protocol = "hierarchical8"} : () -> ()
    tt.return
  }
}
// -----
module attributes {"ttg.num-warps" = 8 : i32} {
  tt.func public @unknown_protocol() {
    // expected-error @+1 {{hierarchical8 requires grid/grid_axis without communicator or census}}
    "tle.distributed_barrier"() <{group_kind = "grid"}> {tle.grid_barrier_protocol = "other"} : () -> ()
    tt.return
  }
}
