// Copyright 2026 FlagOS Contributors
// SPDX-License-Identifier: MIT
// RUN: triton-opt %s -convert-warp-specialize-to-llvm -split-input-file -verify-diagnostics
// Negative native guards; the positive frontend bridge is covered by Python.

module attributes {"ttg.num-warps" = 8 : i32, "ttg.total-num-warps" = 12 : i32, "ttg.num-ctas" = 1 : i32, "tle.ws_trace.capacity" = 64 : i32} {
llvm.mlir.global external @global_smem() {addr_space = 3 : i32, alignment = 16 : i64} : !llvm.array<0 x i8>
// expected-error @+1 {{typed trace requires defined nonvariadic functions}}
llvm.func internal @variadic_leaf(%unrelated: !llvm.ptr<1>, ...) {
  %value = llvm.mlir.constant(17 : i32) : i32
  tle.trace_event %value, %value, %value, %value {kind = 272 : i32, site = 11 : i32}
  llvm.return
}
llvm.func @entry(%scratch: !llvm.ptr<1>, %profile: !llvm.ptr<1>) attributes {allocation.offset = 0 : i32} {
  ttg.warp_specialize() attributes {allocation.offset = 16 : i32, warpGroupStartIds = array<i32: 8>}
  default { ttg.warp_yield }
  partition0() num_warps(1) { ttg.warp_return } : () -> ()
  llvm.return
}
}

// -----

module attributes {"ttg.num-warps" = 8 : i32, "ttg.total-num-warps" = 12 : i32, "ttg.num-ctas" = 1 : i32, "tle.ws_trace.capacity" = 64 : i32} {
llvm.mlir.global external @global_smem() {addr_space = 3 : i32, alignment = 16 : i64} : !llvm.array<0 x i8>
// expected-error @+1 {{typed trace does not support address-taken or indirect device functions}}
llvm.func internal @address_taken_leaf(%unrelated: !llvm.ptr<1>) {
  %value = llvm.mlir.constant(17 : i32) : i32
  tle.trace_event %value, %value, %value, %value {kind = 272 : i32, site = 11 : i32}
  llvm.return
}
llvm.func @entry(%scratch: !llvm.ptr<1>, %profile: !llvm.ptr<1>) attributes {allocation.offset = 0 : i32} {
  %address = llvm.mlir.addressof @address_taken_leaf : !llvm.ptr
  ttg.warp_specialize() attributes {allocation.offset = 16 : i32, warpGroupStartIds = array<i32: 8>}
  default { ttg.warp_yield }
  partition0() num_warps(1) { ttg.warp_return } : () -> ()
  llvm.return
}
}
