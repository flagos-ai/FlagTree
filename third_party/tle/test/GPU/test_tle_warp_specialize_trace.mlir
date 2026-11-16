// Copyright 2026 FlagOS Contributors
// SPDX-License-Identifier: MIT
// RUN: triton-opt %s -convert-warp-specialize-to-llvm -mlir-print-local-scope | FileCheck %s

// An 8-warp default and 1-warp worker leave three real physical warps in
// the runtime's unassigned branch. The common header and exit must record them.
// CHECK-DAG: ttg.profile_scratch_memory_alignment = 64 : i32
// CHECK-DAG: ttg.profile_scratch_memory_size = 25344 : i64
// CHECK-DAG: tle.ws_trace.default_num_warps = 8 : i32
// CHECK-DAG: tle.ws_trace.total_num_warps = 12 : i32
module attributes {"ttg.num-warps" = 8 : i32, "ttg.total-num-warps" = 12 : i32, "ttg.num-ctas" = 1 : i32, "tle.ws_trace.capacity" = 32 : i32} {
llvm.mlir.global external @global_smem() {addr_space = 3 : i32, alignment = 16 : i64} : !llvm.array<0 x i8>

// CHECK-LABEL: llvm.func @trace_control
llvm.func @trace_control(%output: !llvm.ptr<1>, %global_scratch: !llvm.ptr<1>, %profile_scratch: !llvm.ptr<1>) attributes {allocation.offset = 0 : i32} {
  // CHECK: mov.u64 timestamp, %globaltimer
  // CHECK: activemask.b32
  // CHECK: st.global.u32 [stream+24]
  // CHECK: llvm.cond_br
  // CHECK: llvm.switch
  ttg.warp_specialize(%output) attributes {allocation.offset = 16 : i32, warpGroupStartIds = array<i32: 8>}
  default {
    %one = llvm.mlir.constant(1 : i32) : i32
    llvm.store %one, %output : i32, !llvm.ptr<1>
    ttg.warp_yield
  }
  partition0(%arg: !llvm.ptr<1>) num_warps(1) {
    %two = llvm.mlir.constant(2 : i32) : i32
    %next = llvm.getelementptr %arg[1] : (!llvm.ptr<1>) -> !llvm.ptr<1>, i32
    llvm.store %two, %next : i32, !llvm.ptr<1>
    ttg.warp_return
  } : (!llvm.ptr<1>) -> ()
  llvm.return
}
}
