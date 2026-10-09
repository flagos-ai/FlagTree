// RUN: triton-opt %s -split-input-file -tritongpu-global-scratch-memory-allocation -convert-triton-amdgpu-to-llvm=arch=gfx1100 | FileCheck %s

// FlagMega-on-Radeon P1-C: tle.distributed_barrier (group_kind "grid") lowers to
// a sense-reversing global-memory barrier — a global atomicrmw add with agent
// scope to arrive, an acquire atomicrmw to spin, and a CTA barrier bracket. No
// tle op survives.

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "hip:gfx1100", "ttg.threads-per-warp" = 32 : i32} {
  // CHECK-LABEL: llvm.func @grid_barrier
  // arrive (release) + spin (acquire), both agent-scoped global atomics:
  // CHECK: llvm.atomicrmw add {{.*}} syncscope("agent") release
  // CHECK: llvm.atomicrmw add {{.*}} syncscope("agent") acquire
  // CHECK-NOT: tle.distributed_barrier
  tt.func public @grid_barrier() {
    "tle.distributed_barrier"() <{group_kind = "grid"}> : () -> ()
    tt.return
  }
}

// -----

// grid_axis: a per-group counter (group of 4 along axis 1 of an 8x16 domain).

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "hip:gfx1100", "ttg.threads-per-warp" = 32 : i32} {
  // CHECK-LABEL: llvm.func @grid_axis_barrier
  // CHECK: llvm.atomicrmw add {{.*}} syncscope("agent")
  // CHECK-NOT: tle.distributed_barrier
  tt.func public @grid_axis_barrier() {
    "tle.distributed_barrier"() <{group_kind = "grid_axis", group_rank = 1 : i32, group_shape = array<i32: 4>, group_axes = array<i32: 1>, group_domain_shape = array<i32: 8, 16>}> : () -> ()
    tt.return
  }
}
