// Copyright 2026 FlagOS Contributors
// SPDX-License-Identifier: MIT
// RUN: triton-opt %s -convert-warp-specialize-to-llvm -split-input-file -verify-diagnostics

module attributes {"ttg.num-warps" = 8 : i32, "ttg.total-num-warps" = 12 : i32, "ttg.num-ctas" = 1 : i32, "tle.ws_trace.capacity" = 64 : i32, "tle.ws_trace.typed_ctas" = "invalid"} {
llvm.mlir.global external @global_smem() {addr_space = 3 : i32, alignment = 16 : i64} : !llvm.array<0 x i8>
// expected-error @+1 {{tle.ws_trace.typed_ctas must be a dense i32 array}}
llvm.func @wrong_type(%scratch: !llvm.ptr<1>, %profile: !llvm.ptr<1>) attributes {allocation.offset = 0 : i32} {
  ttg.warp_specialize() attributes {allocation.offset = 16 : i32, warpGroupStartIds = array<i32: 8>}
  default { ttg.warp_yield }
  partition0() num_warps(1) { ttg.warp_return } : () -> ()
  llvm.return
}
}

// -----

module attributes {"ttg.num-warps" = 8 : i32, "ttg.total-num-warps" = 12 : i32, "ttg.num-ctas" = 1 : i32, "tle.ws_trace.capacity" = 64 : i32, "tle.ws_trace.typed_ctas" = array<i32: 1, 1>} {
llvm.mlir.global external @global_smem() {addr_space = 3 : i32, alignment = 16 : i64} : !llvm.array<0 x i8>
// expected-error @+1 {{tle.ws_trace.typed_ctas must be nonempty, strictly increasing and nonnegative}}
llvm.func @duplicate(%scratch: !llvm.ptr<1>, %profile: !llvm.ptr<1>) attributes {allocation.offset = 0 : i32} {
  ttg.warp_specialize() attributes {allocation.offset = 16 : i32, warpGroupStartIds = array<i32: 8>}
  default { ttg.warp_yield }
  partition0() num_warps(1) { ttg.warp_return } : () -> ()
  llvm.return
}
}

// -----

module {
llvm.func @reserved_kind(%value: i32) {
  // expected-error @+1 {{typed trace event kind must be a supported pipe or module event}}
  tle.trace_event %value, %value, %value, %value {kind = 262 : i32, site = 1 : i32}
  llvm.return
}
}
