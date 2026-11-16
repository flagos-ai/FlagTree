// Copyright 2026 FlagOS Contributors
// SPDX-License-Identifier: MIT
// RUN: triton-opt %s -convert-warp-specialize-to-llvm -split-input-file -verify-diagnostics > %t
// RUN: %PYTHON %S/Inputs/check_trace_cta_ranges.py %s %t

// Check actual lowering, exact unsigned-u64 truth and single-range cost.
module attributes {"ttg.num-warps" = 8 : i32, "ttg.total-num-warps" = 12 : i32, "ttg.num-ctas" = 1 : i32, "tle.ws_trace.capacity" = 64 : i32, "tle.ws_trace.typed_ctas" = array<i32: 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115, 116, 117, 118, 119, 120, 121, 122, 123, 124, 125, 126, 127>} {
llvm.mlir.global external @global_smem() {addr_space = 3 : i32, alignment = 16 : i64} : !llvm.array<0 x i8>
llvm.func internal @select_full128(%layer: i32, %output: !llvm.ptr<1>) {
  %unused = llvm.mlir.constant(-1 : i32) : i32
  tle.trace_event %layer, %unused, %unused, %unused {kind = 272 : i32, site = 11 : i32}
  llvm.store %layer, %output : i32, !llvm.ptr<1>
  tle.trace_event %layer, %unused, %unused, %unused {kind = 273 : i32, site = 11 : i32}
  llvm.return
}
llvm.func @entry_full128(%output: !llvm.ptr<1>, %scratch: !llvm.ptr<1>, %profile: !llvm.ptr<1>) attributes {allocation.offset = 0 : i32} {
  ttg.warp_specialize(%output) attributes {allocation.offset = 16 : i32, warpGroupStartIds = array<i32: 8>}
  default {
    %layer = llvm.mlir.constant(17 : i32) : i32
    llvm.call @select_full128(%layer, %output) : (i32, !llvm.ptr<1>) -> ()
    ttg.warp_yield
  }
  partition0(%arg: !llvm.ptr<1>) num_warps(1) { ttg.warp_return } : (!llvm.ptr<1>) -> ()
  llvm.return
}
}

// -----

module attributes {"ttg.num-warps" = 8 : i32, "ttg.total-num-warps" = 12 : i32, "ttg.num-ctas" = 1 : i32, "tle.ws_trace.capacity" = 64 : i32, "tle.ws_trace.typed_ctas" = array<i32: 0, 1, 2, 3>} {
llvm.mlir.global external @global_smem() {addr_space = 3 : i32, alignment = 16 : i64} : !llvm.array<0 x i8>
llvm.func internal @select_zero_short(%layer: i32, %output: !llvm.ptr<1>) {
  %unused = llvm.mlir.constant(-1 : i32) : i32
  tle.trace_event %layer, %unused, %unused, %unused {kind = 272 : i32, site = 11 : i32}
  llvm.store %layer, %output : i32, !llvm.ptr<1>
  tle.trace_event %layer, %unused, %unused, %unused {kind = 273 : i32, site = 11 : i32}
  llvm.return
}
llvm.func @entry_zero_short(%output: !llvm.ptr<1>, %scratch: !llvm.ptr<1>, %profile: !llvm.ptr<1>) attributes {allocation.offset = 0 : i32} {
  ttg.warp_specialize(%output) attributes {allocation.offset = 16 : i32, warpGroupStartIds = array<i32: 8>}
  default {
    %layer = llvm.mlir.constant(17 : i32) : i32
    llvm.call @select_zero_short(%layer, %output) : (i32, !llvm.ptr<1>) -> ()
    ttg.warp_yield
  }
  partition0(%arg: !llvm.ptr<1>) num_warps(1) { ttg.warp_return } : (!llvm.ptr<1>) -> ()
  llvm.return
}
}

// -----

module attributes {"ttg.num-warps" = 8 : i32, "ttg.total-num-warps" = 12 : i32, "ttg.num-ctas" = 1 : i32, "tle.ws_trace.capacity" = 64 : i32, "tle.ws_trace.typed_ctas" = array<i32: 7, 8, 9, 10, 11, 12, 13>} {
llvm.mlir.global external @global_smem() {addr_space = 3 : i32, alignment = 16 : i64} : !llvm.array<0 x i8>
llvm.func internal @select_nonzero(%layer: i32, %output: !llvm.ptr<1>) {
  %unused = llvm.mlir.constant(-1 : i32) : i32
  tle.trace_event %layer, %unused, %unused, %unused {kind = 272 : i32, site = 11 : i32}
  llvm.store %layer, %output : i32, !llvm.ptr<1>
  tle.trace_event %layer, %unused, %unused, %unused {kind = 273 : i32, site = 11 : i32}
  llvm.return
}
llvm.func @entry_nonzero(%output: !llvm.ptr<1>, %scratch: !llvm.ptr<1>, %profile: !llvm.ptr<1>) attributes {allocation.offset = 0 : i32} {
  ttg.warp_specialize(%output) attributes {allocation.offset = 16 : i32, warpGroupStartIds = array<i32: 8>}
  default {
    %layer = llvm.mlir.constant(17 : i32) : i32
    llvm.call @select_nonzero(%layer, %output) : (i32, !llvm.ptr<1>) -> ()
    ttg.warp_yield
  }
  partition0(%arg: !llvm.ptr<1>) num_warps(1) { ttg.warp_return } : (!llvm.ptr<1>) -> ()
  llvm.return
}
}

// -----

module attributes {"ttg.num-warps" = 8 : i32, "ttg.total-num-warps" = 12 : i32, "ttg.num-ctas" = 1 : i32, "tle.ws_trace.capacity" = 64 : i32, "tle.ws_trace.typed_ctas" = array<i32: 0>} {
llvm.mlir.global external @global_smem() {addr_space = 3 : i32, alignment = 16 : i64} : !llvm.array<0 x i8>
llvm.func internal @select_singleton0(%layer: i32, %output: !llvm.ptr<1>) {
  %unused = llvm.mlir.constant(-1 : i32) : i32
  tle.trace_event %layer, %unused, %unused, %unused {kind = 272 : i32, site = 11 : i32}
  llvm.store %layer, %output : i32, !llvm.ptr<1>
  tle.trace_event %layer, %unused, %unused, %unused {kind = 273 : i32, site = 11 : i32}
  llvm.return
}
llvm.func @entry_singleton0(%output: !llvm.ptr<1>, %scratch: !llvm.ptr<1>, %profile: !llvm.ptr<1>) attributes {allocation.offset = 0 : i32} {
  ttg.warp_specialize(%output) attributes {allocation.offset = 16 : i32, warpGroupStartIds = array<i32: 8>}
  default {
    %layer = llvm.mlir.constant(17 : i32) : i32
    llvm.call @select_singleton0(%layer, %output) : (i32, !llvm.ptr<1>) -> ()
    ttg.warp_yield
  }
  partition0(%arg: !llvm.ptr<1>) num_warps(1) { ttg.warp_return } : (!llvm.ptr<1>) -> ()
  llvm.return
}
}

// -----

module attributes {"ttg.num-warps" = 8 : i32, "ttg.total-num-warps" = 12 : i32, "ttg.num-ctas" = 1 : i32, "tle.ws_trace.capacity" = 64 : i32, "tle.ws_trace.typed_ctas" = array<i32: 2147483647>} {
llvm.mlir.global external @global_smem() {addr_space = 3 : i32, alignment = 16 : i64} : !llvm.array<0 x i8>
llvm.func internal @select_singleton_max(%layer: i32, %output: !llvm.ptr<1>) {
  %unused = llvm.mlir.constant(-1 : i32) : i32
  tle.trace_event %layer, %unused, %unused, %unused {kind = 272 : i32, site = 11 : i32}
  llvm.store %layer, %output : i32, !llvm.ptr<1>
  tle.trace_event %layer, %unused, %unused, %unused {kind = 273 : i32, site = 11 : i32}
  llvm.return
}
llvm.func @entry_singleton_max(%output: !llvm.ptr<1>, %scratch: !llvm.ptr<1>, %profile: !llvm.ptr<1>) attributes {allocation.offset = 0 : i32} {
  ttg.warp_specialize(%output) attributes {allocation.offset = 16 : i32, warpGroupStartIds = array<i32: 8>}
  default {
    %layer = llvm.mlir.constant(17 : i32) : i32
    llvm.call @select_singleton_max(%layer, %output) : (i32, !llvm.ptr<1>) -> ()
    ttg.warp_yield
  }
  partition0(%arg: !llvm.ptr<1>) num_warps(1) { ttg.warp_return } : (!llvm.ptr<1>) -> ()
  llvm.return
}
}

// -----

module attributes {"ttg.num-warps" = 8 : i32, "ttg.total-num-warps" = 12 : i32, "ttg.num-ctas" = 1 : i32, "tle.ws_trace.capacity" = 64 : i32, "tle.ws_trace.typed_ctas" = array<i32: 0, 15, 64, 127>} {
llvm.mlir.global external @global_smem() {addr_space = 3 : i32, alignment = 16 : i64} : !llvm.array<0 x i8>
llvm.func internal @select_sparse(%layer: i32, %output: !llvm.ptr<1>) {
  %unused = llvm.mlir.constant(-1 : i32) : i32
  tle.trace_event %layer, %unused, %unused, %unused {kind = 272 : i32, site = 11 : i32}
  llvm.store %layer, %output : i32, !llvm.ptr<1>
  tle.trace_event %layer, %unused, %unused, %unused {kind = 273 : i32, site = 11 : i32}
  llvm.return
}
llvm.func @entry_sparse(%output: !llvm.ptr<1>, %scratch: !llvm.ptr<1>, %profile: !llvm.ptr<1>) attributes {allocation.offset = 0 : i32} {
  ttg.warp_specialize(%output) attributes {allocation.offset = 16 : i32, warpGroupStartIds = array<i32: 8>}
  default {
    %layer = llvm.mlir.constant(17 : i32) : i32
    llvm.call @select_sparse(%layer, %output) : (i32, !llvm.ptr<1>) -> ()
    ttg.warp_yield
  }
  partition0(%arg: !llvm.ptr<1>) num_warps(1) { ttg.warp_return } : (!llvm.ptr<1>) -> ()
  llvm.return
}
}

// -----

module attributes {"ttg.num-warps" = 8 : i32, "ttg.total-num-warps" = 12 : i32, "ttg.num-ctas" = 1 : i32, "tle.ws_trace.capacity" = 64 : i32, "tle.ws_trace.typed_ctas" = array<i32: 0, 1, 2, 7, 10, 11, 12, 21, 2147483646, 2147483647>} {
llvm.mlir.global external @global_smem() {addr_space = 3 : i32, alignment = 16 : i64} : !llvm.array<0 x i8>
llvm.func internal @select_mixed(%layer: i32, %output: !llvm.ptr<1>) {
  %unused = llvm.mlir.constant(-1 : i32) : i32
  tle.trace_event %layer, %unused, %unused, %unused {kind = 272 : i32, site = 11 : i32}
  llvm.store %layer, %output : i32, !llvm.ptr<1>
  tle.trace_event %layer, %unused, %unused, %unused {kind = 273 : i32, site = 11 : i32}
  llvm.return
}
llvm.func @entry_mixed(%output: !llvm.ptr<1>, %scratch: !llvm.ptr<1>, %profile: !llvm.ptr<1>) attributes {allocation.offset = 0 : i32} {
  ttg.warp_specialize(%output) attributes {allocation.offset = 16 : i32, warpGroupStartIds = array<i32: 8>}
  default {
    %layer = llvm.mlir.constant(17 : i32) : i32
    llvm.call @select_mixed(%layer, %output) : (i32, !llvm.ptr<1>) -> ()
    ttg.warp_yield
  }
  partition0(%arg: !llvm.ptr<1>) num_warps(1) { ttg.warp_return } : (!llvm.ptr<1>) -> ()
  llvm.return
}
}

// -----

module attributes {"ttg.num-warps" = 8 : i32, "ttg.total-num-warps" = 12 : i32, "ttg.num-ctas" = 1 : i32, "tle.ws_trace.capacity" = 64 : i32, "tle.ws_trace.typed_ctas" = array<i32: 2147483645, 2147483646, 2147483647>} {
llvm.mlir.global external @global_smem() {addr_space = 3 : i32, alignment = 16 : i64} : !llvm.array<0 x i8>
llvm.func internal @select_max_run(%layer: i32, %output: !llvm.ptr<1>) {
  %unused = llvm.mlir.constant(-1 : i32) : i32
  tle.trace_event %layer, %unused, %unused, %unused {kind = 272 : i32, site = 11 : i32}
  llvm.store %layer, %output : i32, !llvm.ptr<1>
  tle.trace_event %layer, %unused, %unused, %unused {kind = 273 : i32, site = 11 : i32}
  llvm.return
}
llvm.func @entry_max_run(%output: !llvm.ptr<1>, %scratch: !llvm.ptr<1>, %profile: !llvm.ptr<1>) attributes {allocation.offset = 0 : i32} {
  ttg.warp_specialize(%output) attributes {allocation.offset = 16 : i32, warpGroupStartIds = array<i32: 8>}
  default {
    %layer = llvm.mlir.constant(17 : i32) : i32
    llvm.call @select_max_run(%layer, %output) : (i32, !llvm.ptr<1>) -> ()
    ttg.warp_yield
  }
  partition0(%arg: !llvm.ptr<1>) num_warps(1) { ttg.warp_return } : (!llvm.ptr<1>) -> ()
  llvm.return
}
}

// -----

module attributes {"ttg.num-warps" = 8 : i32, "ttg.total-num-warps" = 12 : i32, "ttg.num-ctas" = 1 : i32, "tle.ws_trace.capacity" = 64 : i32, "tle.ws_trace.typed_ctas" = array<i32: 1023, 1024, 1025, 1026>} {
llvm.mlir.global external @global_smem() {addr_space = 3 : i32, alignment = 16 : i64} : !llvm.array<0 x i8>
llvm.func internal @select_far_run(%layer: i32, %output: !llvm.ptr<1>) {
  %unused = llvm.mlir.constant(-1 : i32) : i32
  tle.trace_event %layer, %unused, %unused, %unused {kind = 272 : i32, site = 11 : i32}
  llvm.store %layer, %output : i32, !llvm.ptr<1>
  tle.trace_event %layer, %unused, %unused, %unused {kind = 273 : i32, site = 11 : i32}
  llvm.return
}
llvm.func @entry_far_run(%output: !llvm.ptr<1>, %scratch: !llvm.ptr<1>, %profile: !llvm.ptr<1>) attributes {allocation.offset = 0 : i32} {
  ttg.warp_specialize(%output) attributes {allocation.offset = 16 : i32, warpGroupStartIds = array<i32: 8>}
  default {
    %layer = llvm.mlir.constant(17 : i32) : i32
    llvm.call @select_far_run(%layer, %output) : (i32, !llvm.ptr<1>) -> ()
    ttg.warp_yield
  }
  partition0(%arg: !llvm.ptr<1>) num_warps(1) { ttg.warp_return } : (!llvm.ptr<1>) -> ()
  llvm.return
}
}

// -----

module attributes {"ttg.num-warps" = 8 : i32, "ttg.total-num-warps" = 12 : i32, "ttg.num-ctas" = 1 : i32, "tle.ws_trace.capacity" = 64 : i32, "tle.ws_trace.typed_ctas" = array<i32: 2, 3, 4, 9, 10, 127, 128>} {
llvm.mlir.global external @global_smem() {addr_space = 3 : i32, alignment = 16 : i64} : !llvm.array<0 x i8>
llvm.func internal @select_multi_run(%layer: i32, %output: !llvm.ptr<1>) {
  %unused = llvm.mlir.constant(-1 : i32) : i32
  tle.trace_event %layer, %unused, %unused, %unused {kind = 272 : i32, site = 11 : i32}
  llvm.store %layer, %output : i32, !llvm.ptr<1>
  tle.trace_event %layer, %unused, %unused, %unused {kind = 273 : i32, site = 11 : i32}
  llvm.return
}
llvm.func @entry_multi_run(%output: !llvm.ptr<1>, %scratch: !llvm.ptr<1>, %profile: !llvm.ptr<1>) attributes {allocation.offset = 0 : i32} {
  ttg.warp_specialize(%output) attributes {allocation.offset = 16 : i32, warpGroupStartIds = array<i32: 8>}
  default {
    %layer = llvm.mlir.constant(17 : i32) : i32
    llvm.call @select_multi_run(%layer, %output) : (i32, !llvm.ptr<1>) -> ()
    ttg.warp_yield
  }
  partition0(%arg: !llvm.ptr<1>) num_warps(1) { ttg.warp_return } : (!llvm.ptr<1>) -> ()
  llvm.return
}
}

// -----

module attributes {"ttg.num-warps" = 8 : i32, "ttg.total-num-warps" = 12 : i32, "ttg.num-ctas" = 1 : i32, "tle.ws_trace.capacity" = 64 : i32} {
llvm.mlir.global external @global_smem() {addr_space = 3 : i32, alignment = 16 : i64} : !llvm.array<0 x i8>
llvm.func internal @select_absent(%layer: i32, %output: !llvm.ptr<1>) {
  %unused = llvm.mlir.constant(-1 : i32) : i32
  tle.trace_event %layer, %unused, %unused, %unused {kind = 272 : i32, site = 11 : i32}
  llvm.store %layer, %output : i32, !llvm.ptr<1>
  tle.trace_event %layer, %unused, %unused, %unused {kind = 273 : i32, site = 11 : i32}
  llvm.return
}
llvm.func @entry_absent(%output: !llvm.ptr<1>, %scratch: !llvm.ptr<1>, %profile: !llvm.ptr<1>) attributes {allocation.offset = 0 : i32} {
  ttg.warp_specialize(%output) attributes {allocation.offset = 16 : i32, warpGroupStartIds = array<i32: 8>}
  default {
    %layer = llvm.mlir.constant(17 : i32) : i32
    llvm.call @select_absent(%layer, %output) : (i32, !llvm.ptr<1>) -> ()
    ttg.warp_yield
  }
  partition0(%arg: !llvm.ptr<1>) num_warps(1) { ttg.warp_return } : (!llvm.ptr<1>) -> ()
  llvm.return
}
}

// -----

module attributes {"ttg.num-warps" = 8 : i32, "ttg.total-num-warps" = 12 : i32, "ttg.num-ctas" = 1 : i32, "tle.ws_trace.capacity" = 64 : i32, "tle.ws_trace.typed_ctas" = array<i32>} {
llvm.mlir.global external @global_smem() {addr_space = 3 : i32, alignment = 16 : i64} : !llvm.array<0 x i8>
// expected-error @+1 {{tle.ws_trace.typed_ctas must be nonempty, strictly increasing and nonnegative}}
llvm.func @invalid_empty(%scratch: !llvm.ptr<1>, %profile: !llvm.ptr<1>) attributes {allocation.offset = 0 : i32} {
  ttg.warp_specialize() attributes {allocation.offset = 16 : i32, warpGroupStartIds = array<i32: 8>}
  default { ttg.warp_yield }
  partition0() num_warps(1) { ttg.warp_return } : () -> ()
  llvm.return
}
}

// -----

module attributes {"ttg.num-warps" = 8 : i32, "ttg.total-num-warps" = 12 : i32, "ttg.num-ctas" = 1 : i32, "tle.ws_trace.capacity" = 64 : i32, "tle.ws_trace.typed_ctas" = array<i32: -1, 0>} {
llvm.mlir.global external @global_smem() {addr_space = 3 : i32, alignment = 16 : i64} : !llvm.array<0 x i8>
// expected-error @+1 {{tle.ws_trace.typed_ctas must be nonempty, strictly increasing and nonnegative}}
llvm.func @invalid_negative(%scratch: !llvm.ptr<1>, %profile: !llvm.ptr<1>) attributes {allocation.offset = 0 : i32} {
  ttg.warp_specialize() attributes {allocation.offset = 16 : i32, warpGroupStartIds = array<i32: 8>}
  default { ttg.warp_yield }
  partition0() num_warps(1) { ttg.warp_return } : () -> ()
  llvm.return
}
}

// -----

module attributes {"ttg.num-warps" = 8 : i32, "ttg.total-num-warps" = 12 : i32, "ttg.num-ctas" = 1 : i32, "tle.ws_trace.capacity" = 64 : i32, "tle.ws_trace.typed_ctas" = array<i32: 2, 1>} {
llvm.mlir.global external @global_smem() {addr_space = 3 : i32, alignment = 16 : i64} : !llvm.array<0 x i8>
// expected-error @+1 {{tle.ws_trace.typed_ctas must be nonempty, strictly increasing and nonnegative}}
llvm.func @invalid_unsorted(%scratch: !llvm.ptr<1>, %profile: !llvm.ptr<1>) attributes {allocation.offset = 0 : i32} {
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
llvm.func @invalid_duplicate(%scratch: !llvm.ptr<1>, %profile: !llvm.ptr<1>) attributes {allocation.offset = 0 : i32} {
  ttg.warp_specialize() attributes {allocation.offset = 16 : i32, warpGroupStartIds = array<i32: 8>}
  default { ttg.warp_yield }
  partition0() num_warps(1) { ttg.warp_return } : () -> ()
  llvm.return
}
}

// -----

module attributes {"ttg.num-warps" = 8 : i32, "ttg.total-num-warps" = 12 : i32, "ttg.num-ctas" = 1 : i32, "tle.ws_trace.capacity" = 64 : i32, "tle.ws_trace.typed_ctas" = "invalid"} {
llvm.mlir.global external @global_smem() {addr_space = 3 : i32, alignment = 16 : i64} : !llvm.array<0 x i8>
// expected-error @+1 {{tle.ws_trace.typed_ctas must be a dense i32 array}}
llvm.func @invalid_wrong_type(%scratch: !llvm.ptr<1>, %profile: !llvm.ptr<1>) attributes {allocation.offset = 0 : i32} {
  ttg.warp_specialize() attributes {allocation.offset = 16 : i32, warpGroupStartIds = array<i32: 8>}
  default { ttg.warp_yield }
  partition0() num_warps(1) { ttg.warp_return } : () -> ()
  llvm.return
}
}

// -----

module attributes {"ttg.num-warps" = 8 : i32, "ttg.total-num-warps" = 12 : i32, "ttg.num-ctas" = 1 : i32, "tle.ws_trace.typed_ctas" = array<i32: 0>} {
llvm.mlir.global external @global_smem() {addr_space = 3 : i32, alignment = 16 : i64} : !llvm.array<0 x i8>
// expected-error @+1 {{tle.ws_trace.typed_ctas requires enabled trace capacity}}
llvm.func @invalid_missing_capacity(%scratch: !llvm.ptr<1>, %profile: !llvm.ptr<1>) attributes {allocation.offset = 0 : i32} {
  ttg.warp_specialize() attributes {allocation.offset = 16 : i32, warpGroupStartIds = array<i32: 8>}
  default { ttg.warp_yield }
  partition0() num_warps(1) { ttg.warp_return } : () -> ()
  llvm.return
}
}
