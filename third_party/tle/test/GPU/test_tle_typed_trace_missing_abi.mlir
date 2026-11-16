// Copyright 2026 FlagOS Contributors
// SPDX-License-Identifier: MIT
// RUN: triton-opt %s -convert-warp-specialize-to-llvm -verify-diagnostics

module attributes {"ttg.num-warps" = 4 : i32, "ttg.num-ctas" = 1 : i32} {
  llvm.func @missing_profile_abi(%layer: i32, %iter: i32, %slot: i32, %phase: i32) {
    // expected-error @+1 {{typed trace requires an enabled, configured WS profile-scratch ABI}}
    "tle.trace_event"(%layer, %iter, %slot, %phase) {kind = 256 : i32, site = 1 : i32} : (i32, i32, i32, i32) -> ()
    llvm.return
  }
}
