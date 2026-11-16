// Copyright 2026 FlagOS Contributors
// SPDX-License-Identifier: MIT
// RUN: triton-opt %s -convert-warp-specialize-to-llvm -verify-diagnostics

module attributes {"ttg.num-warps" = 8 : i32, "ttg.total-num-warps" = 12 : i32, "ttg.num-ctas" = 1 : i32, "tle.ws_trace.capacity" = "invalid"} {
llvm.mlir.global external @global_smem() {addr_space = 3 : i32, alignment = 16 : i64} : !llvm.array<0 x i8>

// expected-error @+1 {{tle.ws_trace.capacity must be an integer attribute}}
llvm.func @wrong_capacity_type(%global_scratch: !llvm.ptr<1>, %profile_scratch: !llvm.ptr<1>) attributes {allocation.offset = 0 : i32} {
  ttg.warp_specialize() attributes {allocation.offset = 16 : i32, warpGroupStartIds = array<i32: 8>}
  default {
    ttg.warp_yield
  }
  partition0() num_warps(1) {
    ttg.warp_return
  } : () -> ()
  llvm.return
}
}
