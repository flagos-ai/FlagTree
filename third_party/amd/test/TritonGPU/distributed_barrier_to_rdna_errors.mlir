// RUN: triton-opt %s -split-input-file -verify-diagnostics -convert-triton-amdgpu-to-llvm=arch=gfx1100

// FlagMega-on-Radeon P1-C: cluster/submesh group kinds use NVIDIA hardware
// clusters, which RDNA lacks — diagnosed, not miscompiled. The op then fails to
// legalize (the partial conversion also reports that).

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "hip:gfx1100", "ttg.threads-per-warp" = 32 : i32} {
  tt.func public @cluster_unsupported() {
    // expected-error @below {{is unsupported on the RDNA backend}}
    // expected-error @below {{failed to legalize operation 'tle.distributed_barrier'}}
    "tle.distributed_barrier"() <{group_kind = "cluster"}> : () -> ()
    tt.return
  }
}
