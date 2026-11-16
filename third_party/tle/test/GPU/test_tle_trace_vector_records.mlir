// Copyright 2026 FlagOS Contributors
// SPDX-License-Identifier: MIT
// RUN: triton-opt %s -convert-warp-specialize-to-llvm > %t
// RUN: FileCheck %s < %t
// RUN: %PYTHON %S/Inputs/check_trace_vector_records.py %t

module attributes {"ttg.num-warps" = 8 : i32, "ttg.total-num-warps" = 12 : i32, "ttg.num-ctas" = 1 : i32, "tle.ws_trace.capacity" = 64 : i32, "tle.ws_trace.typed"} {
llvm.mlir.global external @global_smem() {addr_space = 3 : i32, alignment = 16 : i64} : !llvm.array<0 x i8>

// CHECK-LABEL: llvm.func internal @leaf
// CHECK-SAME: tle.ws_trace.forwarded_arg = 5 : i32
// CHECK: mov.u64 timestamp, %globaltimer
// CHECK-SAME: mov.u32 cx, %ctaid.x
// CHECK-SAME: mov.u32 cy, %ctaid.y
// CHECK-SAME: cvt.u64.u32 temp, cy
// CHECK-SAME: cvt.u64.u32 temp, cx
// CHECK-SAME: st.global.v2.u64 [record+0], {timestamp, ordinal}
// CHECK-SAME: .reg .b32 event_value, record_cx, record_cy
// CHECK-SAME: mov.u32 event_value, 256
// CHECK-SAME: mov.u32 record_cx, %ctaid.x
// CHECK-SAME: mov.u32 record_cy, %ctaid.y
// CHECK-SAME: st.global.v4.u32 [record+16], {event_value, $1, record_cx, record_cy}
// CHECK-SAME: st.global.v4.u32 [record+32], {cz, warp, sx, mask}
// CHECK-SAME: st.global.v4.u32 [record+48], {$2, $3, $4, $5}
llvm.func internal @leaf(%layer: i32, %iter: i32, %slot: i32, %phase: i32, %unrelated: !llvm.ptr<1>) {
  "tle.trace_event"(%layer, %iter, %slot, %phase) {kind = 256 : i32, site = 101 : i32} : (i32, i32, i32, i32) -> ()
  llvm.return
}

// CHECK-LABEL: llvm.func internal @outer
// CHECK-SAME: tle.ws_trace.forwarded_arg = 5 : i32
// CHECK: llvm.call @leaf({{.*}}) : (i32, i32, i32, i32, !llvm.ptr<1>, !llvm.ptr<1>) -> ()
llvm.func internal @outer(%layer: i32, %iter: i32, %slot: i32, %phase: i32, %unrelated: !llvm.ptr<1>) {
  llvm.call @leaf(%layer, %iter, %slot, %phase, %unrelated) : (i32, i32, i32, i32, !llvm.ptr<1>) -> ()
  llvm.return
}

// CHECK-LABEL: llvm.func @entry
// CHECK: st.global.v4.u32 [record+48], {zero, zero, zero, zero}
llvm.func @entry(%output: !llvm.ptr<1>, %global_scratch: !llvm.ptr<1>, %profile_scratch: !llvm.ptr<1>) attributes {allocation.offset = 0 : i32} {
  ttg.warp_specialize(%output) attributes {allocation.offset = 16 : i32, warpGroupStartIds = array<i32: 8>}
  default {
    %x = llvm.mlir.constant(7 : i32) : i32
    llvm.call @outer(%x, %x, %x, %x, %output) : (i32, i32, i32, i32, !llvm.ptr<1>) -> ()
    ttg.warp_yield
  }
  partition0(%arg: !llvm.ptr<1>) num_warps(1) {
    ttg.warp_return
  } : (!llvm.ptr<1>) -> ()
  llvm.return
}
}
