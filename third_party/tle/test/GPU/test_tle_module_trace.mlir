// Copyright 2026 FlagOS Contributors
// SPDX-License-Identifier: MIT
// RUN: triton-opt %s -convert-warp-specialize-to-llvm | FileCheck %s

// Typed mode must be inferred before native entry records are generated.
module attributes {"ttg.num-warps" = 8 : i32, "ttg.total-num-warps" = 12 : i32, "ttg.num-ctas" = 1 : i32, "tle.ws_trace.capacity" = 2048 : i32, "tle.ws_trace.typed_ctas" = array<i32: 0, 15, 64, 127>} {
llvm.mlir.global external @global_smem() {addr_space = 3 : i32, alignment = 16 : i64} : !llvm.array<0 x i8>

// CHECK-LABEL: llvm.func internal @module_call
// CHECK-SAME: tle.ws_trace.forwarded_arg = 2 : i32
// CHECK: setp.eq.u64 match, cta, 0
// CHECK-SAME: setp.eq.u64 match, cta, 15
// CHECK-SAME: setp.eq.u64 match, cta, 64
// CHECK-SAME: setp.eq.u64 match, cta, 127
// CHECK-SAME: @!selected bra TRACE_DONE
// CHECK-SAME: mov.u32 event_value, 272
// CHECK-SAME: st.global.v4.u32 [record+16], {event_value, $1, record_cx, record_cy}
// CHECK: llvm.store
// CHECK: mov.u32 event_value, 273
// CHECK-SAME: st.global.v4.u32 [record+16], {event_value, $1, record_cx, record_cy}
llvm.func internal @module_call(%layer: i32, %output: !llvm.ptr<1>) {
  %unused = llvm.mlir.constant(-1 : i32) : i32
  tle.trace_event %layer, %unused, %unused, %unused {kind = 272 : i32, site = 11 : i32}
  llvm.store %layer, %output : i32, !llvm.ptr<1>
  tle.trace_event %layer, %unused, %unused, %unused {kind = 273 : i32, site = 11 : i32}
  llvm.return
}

// CHECK-LABEL: llvm.func @entry
// CHECK: st.global.u32 [stream+4], 2
// CHECK: llvm.call @module_call({{.*}}) : (i32, !llvm.ptr<1>, !llvm.ptr<1>) -> ()
llvm.func @entry(%output: !llvm.ptr<1>, %global_scratch: !llvm.ptr<1>, %profile_scratch: !llvm.ptr<1>) attributes {allocation.offset = 0 : i32} {
  ttg.warp_specialize(%output) attributes {allocation.offset = 16 : i32, warpGroupStartIds = array<i32: 8>}
  default {
    %layer = llvm.mlir.constant(17 : i32) : i32
    llvm.call @module_call(%layer, %output) : (i32, !llvm.ptr<1>) -> ()
    ttg.warp_yield
  }
  partition0(%arg: !llvm.ptr<1>) num_warps(1) {
    ttg.warp_return
  } : (!llvm.ptr<1>) -> ()
  llvm.return
}
}
