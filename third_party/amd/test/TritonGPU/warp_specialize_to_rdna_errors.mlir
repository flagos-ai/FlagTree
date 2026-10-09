// RUN: triton-opt %s -split-input-file -verify-diagnostics -tritonamdgpu-convert-warp-specialize

// FlagMega-on-Radeon P1-B: unsupported warp_specialize shapes are diagnosed,
// not silently miscompiled.

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "hip:gfx1100", "ttg.threads-per-warp" = 32 : i32} {
  tt.func public @ws_yields_value(%arg0: i32) -> i32 {
    // expected-error @below {{warp_specialize that returns values is unsupported}}
    %r = ttg.warp_specialize(%arg0) attributes {warpGroupStartIds = array<i32: 4>}
    default {
      ttg.warp_yield %arg0 : i32
    }
    partition0(%a: i32) num_warps(4) {
      ttg.warp_return
    } : (i32) -> i32
    tt.return %r : i32
  }
}

// -----

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "hip:gfx1100", "ttg.threads-per-warp" = 32 : i32} {
  tt.func public @ws_barrier_in_partition(%arg0: i32) {
    // expected-error @below {{would deadlock on RDNA}}
    ttg.warp_specialize(%arg0) attributes {warpGroupStartIds = array<i32: 4>}
    default {
      ttg.warp_yield
    }
    partition0(%a: i32) num_warps(4) {
      gpu.barrier
      ttg.warp_return
    } : (i32) -> ()
    tt.return
  }
}
