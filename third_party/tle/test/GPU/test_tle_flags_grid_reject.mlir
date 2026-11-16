// RUN: triton-opt %s --split-input-file --tritongpu-global-scratch-memory-allocation --verify-diagnostics
module {
  tt.func public @missing_domain() {
    // expected-error @+1 {{flags32 requires static domain and three-dimensional launch grid}}
    "tle.distributed_barrier"() <{group_kind = "grid"}> {tle.grid_barrier_protocol = "flags32"} : () -> ()
    tt.return
  }
}
// -----
module {
  tt.func public @wrong_population() {
    // expected-error @+1 {{flags32 physical launch and logical domain populations differ}}
    "tle.distributed_barrier"() <{group_kind = "grid"}> {tle.grid_barrier_protocol = "flags32", tle.flags_domain = array<i32: 128>, tle.flags_launch_grid = array<i32: 16, 4, 1>} : () -> ()
    tt.return
  }
}
// -----
module {
  tt.func public @size_overflow() {
    // expected-error @+1 {{flags32 scratch layout exceeds signed i32 byte offsets}}
    "tle.distributed_barrier"() <{group_kind = "grid"}> {tle.grid_barrier_protocol = "flags32", tle.flags_domain = array<i32: 1073741824>, tle.flags_launch_grid = array<i32: 1073741824, 1, 1>} : () -> ()
    tt.return
  }
}
// -----
module {
  tt.func private @one() attributes {noinline = true} {
    "tle.distributed_barrier"() <{group_kind = "grid"}> {tle.grid_barrier_protocol = "flags32", tle.flags_domain = array<i32: 8, 16>, tle.flags_launch_grid = array<i32: 16, 8, 1>} : () -> ()
    tt.return
  }
  tt.func public @conflicting_launch() {
    tt.call @one() : () -> ()
    // expected-error @+1 {{flags32 sites/callees have conflicting launch domains}}
    "tle.distributed_barrier"() <{group_kind = "grid"}> {tle.grid_barrier_protocol = "flags32", tle.flags_domain = array<i32: 8, 16>, tle.flags_launch_grid = array<i32: 128, 1, 1>} : () -> ()
    tt.return
  }
}
// -----
module {
  tt.func private @large() attributes {noinline = true} {
    "tle.distributed_barrier"() <{group_kind = "grid"}> {tle.grid_barrier_protocol = "flags32", tle.flags_domain = array<i32: 268435456>, tle.flags_launch_grid = array<i32: 268435456, 1, 1>} : () -> ()
    tt.return
  }
  tt.func public @nested_overflow() {
    tt.call @large() : () -> ()
    // expected-error @+1 {{hierarchical grid scratch accumulated frame exceeds signed i32}}
    tt.call @large() : () -> ()
    tt.return
  }
}
