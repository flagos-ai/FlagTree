// RUN: triton-opt --split-input-file %s --verify-diagnostics


module attributes {"ttg.num-warps" = 4 : i32} {
  tt.func @reuse_warps_wrong_attribute_type() {
    // expected-error @+1 {{reuseDefaultWarps}}
    ttg.warp_specialize() attributes {reuseDefaultWarps = 1 : i32}
    default { ttg.warp_yield }
    partition0() num_warps(4) { ttg.warp_return }
    : () -> ()
    tt.return
  }
}

// -----

module attributes {"ttg.num-warps" = 4 : i32} {
  tt.func @reuse_warps_incomplete_coverage() {
    // expected-error @+1 {{in-place partitions cover 2 warps, expected 4}}
    ttg.warp_specialize() attributes {reuseDefaultWarps = true}
    default { ttg.warp_yield }
    partition0() num_warps(2) { ttg.warp_return }
    : () -> ()
    tt.return
  }
}

// -----

module attributes {"ttg.num-warps" = 4 : i32} {
  tt.func @reuse_warps_nonempty_default(%ptr: !tt.ptr<i32>, %value: i32) {
    // expected-error @+1 {{reuseDefaultWarps requires a default region containing only ttg.warp_yield}}
    ttg.warp_specialize() attributes {reuseDefaultWarps = true}
    default {
      tt.store %ptr, %value : !tt.ptr<i32>
      ttg.warp_yield
    }
    partition0() num_warps(4) { ttg.warp_return }
    : () -> ()
    tt.return
  }
}

// -----
