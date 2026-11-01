// Copyright 2026 FlagOS Contributors
// SPDX-License-Identifier: MIT
// RUN: triton-opt %s -convert-warp-specialize-to-llvm | FileCheck %s

// The device library is linked after warp specialization. Its external block
// barrier must carry the caller's physical group, including through a helper.
module attributes {"ttg.num-warps" = 8 : i32, "ttg.total-num-warps" = 16 : i32} {
  llvm.mlir.global external @global_smem() {addr_space = 3 : i32, alignment = 16 : i64} : !llvm.array<0 x i8>
  llvm.func @flagcxIntraBarrierArriveS(!llvm.ptr, i32, i32, i1, i32)
  llvm.func @flagcxIntraBarrierWaitS(!llvm.ptr, i32, i32, i1, i32)
  llvm.func @flagcxIntraBarrierSyncS(!llvm.ptr, i32, i32, i1, i32)

  // CHECK-DAG: llvm.func @flagcxIntraBarrierArriveWarpGroupS(!llvm.ptr, i32, i32, i32, i32, i1, i32)
  // CHECK-DAG: llvm.func @flagcxIntraBarrierWaitWarpGroupS(!llvm.ptr, i32, i32, i32, i32, i1, i32)
  // CHECK-DAG: llvm.func @flagcxIntraBarrierSyncWarpGroupS(!llvm.ptr, i32, i32, i32, i32, i1, i32)
  llvm.func internal @communication(%comm: !llvm.ptr, %index: i32) attributes {noinline} {
    %block = llvm.mlir.constant(2 : i32) : i32
    %false = llvm.mlir.constant(false) : i1
    %order = llvm.mlir.constant(3 : i32) : i32
    llvm.call @flagcxIntraBarrierArriveS(%comm, %block, %index, %false, %order) : (!llvm.ptr, i32, i32, i1, i32) -> ()
    llvm.call @flagcxIntraBarrierWaitS(%comm, %block, %index, %false, %order) : (!llvm.ptr, i32, i32, i1, i32) -> ()
    llvm.call @flagcxIntraBarrierSyncS(%comm, %block, %index, %false, %order) : (!llvm.ptr, i32, i32, i1, i32) -> ()
    llvm.return
  }

  // CHECK-DAG: llvm.func internal @communication__tle_ws_barrier_0_256_start_0
  // CHECK-DAG: llvm.func internal @communication__tle_ws_barrier_2_128_start_8
  // CHECK-DAG: llvm.func internal @communication__tle_ws_barrier_3_128_start_12
  // CHECK-LABEL: llvm.func @entry(
  // CHECK-DAG: llvm.call @communication__tle_ws_barrier_0_256_start_0
  // CHECK-DAG: llvm.call @communication__tle_ws_barrier_2_128_start_8
  // CHECK-DAG: llvm.call @communication__tle_ws_barrier_3_128_start_12
  llvm.func @entry(%comm: !llvm.ptr, %index: i32) attributes {allocation.offset = 0 : i32} {
    ttg.warp_specialize(%comm, %index) attributes {allocation.offset = 0 : i32, warpGroupStartIds = array<i32: 8, 12>}
    default {
      llvm.call @communication(%comm, %index) : (!llvm.ptr, i32) -> ()
      ttg.warp_yield
    }
    partition0(%c: !llvm.ptr, %i: i32) num_warps(4) {
      llvm.call @communication(%c, %i) : (!llvm.ptr, i32) -> ()
      ttg.warp_return
    }
    partition1(%c: !llvm.ptr, %i: i32) num_warps(4) {
      llvm.call @communication(%c, %i) : (!llvm.ptr, i32) -> ()
      ttg.warp_return
    } : (!llvm.ptr, i32) -> ()
    llvm.return
  }
}
