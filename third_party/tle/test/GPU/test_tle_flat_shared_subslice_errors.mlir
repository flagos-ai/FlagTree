// Copyright 2026 FlagOS Contributors
// SPDX-License-Identifier: MIT
// RUN: triton-opt %s --verify-diagnostics
#flat = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>
#smem = #ttg.shared_memory
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32} {
  tt.func @outside_flat_slice() {
    %s = ttg.local_alloc : () -> !ttg.memdesc<192xf32, #flat, #smem, mutable>
    // expected-error @+1 {{affine flat subslice is outside the source extent}}
    %v = ttg.memdesc_subslice %s[160] : !ttg.memdesc<192xf32, #flat, #smem, mutable> -> !ttg.memdesc<96xf32, #flat, #smem, mutable, 192>
    tt.return
  }
}
