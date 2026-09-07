// Copyright 2026 FlagOS Contributors
// SPDX-License-Identifier: MIT
// RUN: triton-opt %s -convert-warp-specialize-to-llvm | FileCheck %s --implicit-check-not=arithmetic__tle_ws_barrier --implicit-check-not=plain_wrapper__tle_ws_barrier
// RUN: triton-opt %s -convert-warp-specialize-to-llvm | FileCheck %s --check-prefix=DIRECT

// Only the transitive callers of a CTA barrier depend on the warp-group
// execution scope. An ordinary noinline call DAG must remain shared, even
// when both the producer and consumer (including scoped helpers) call it.
module attributes {"ttg.num-warps" = 8 : i32, "ttg.total-num-warps" = 12 : i32} {
  llvm.mlir.global external @global_smem() {addr_space = 3 : i32, alignment = 16 : i64} : !llvm.array<0 x i8>

  // CHECK-LABEL: llvm.func internal @arithmetic(
  llvm.func internal @arithmetic(%value: i32) -> i32 attributes {noinline} {
    %one = llvm.mlir.constant(1 : i32) : i32
    %result = llvm.add %value, %one : i32
    llvm.return %result : i32
  }

  // CHECK-LABEL: llvm.func internal @plain_wrapper(
  // CHECK: llvm.call @arithmetic(
  llvm.func internal @plain_wrapper(%value: i32) -> i32 attributes {noinline} {
    %result = llvm.call @arithmetic(%value) : (i32) -> i32
    llvm.return %result : i32
  }

  llvm.func internal @collective(%value: i32) -> i32 attributes {noinline} {
    nvvm.barrier0
    %result = llvm.call @plain_wrapper(%value) : (i32) -> i32
    llvm.return %result : i32
  }

  llvm.func internal @collective_wrapper(%value: i32) -> i32 attributes {noinline} {
    %result = llvm.call @collective(%value) : (i32) -> i32
    llvm.return %result : i32
  }

  // CHECK-DAG: llvm.func internal @collective__tle_ws_barrier_0_256(
  // CHECK-DAG: llvm.func internal @collective__tle_ws_barrier_2_128(
  // CHECK-DAG: llvm.func internal @collective_wrapper__tle_ws_barrier_0_256(
  // CHECK-DAG: llvm.func internal @collective_wrapper__tle_ws_barrier_2_128(
  // Both direct barriers and calls are handled during the same entry walk.
  // Sibling collective bodies above keep their original barrier until DCE;
  // only the entry and its scoped clones must be free of whole-CTA barriers.
  // DIRECT-LABEL: llvm.func @entry(
  // DIRECT-NOT: nvvm.barrier0
  // CHECK-LABEL: llvm.func @entry(
  // CHECK-DAG: llvm.call @plain_wrapper(
  // CHECK-DAG: llvm.call @collective_wrapper__tle_ws_barrier_0_256(
  // CHECK-DAG: llvm.call @collective_wrapper__tle_ws_barrier_2_128(
  llvm.func @entry(%value: i32, %output: !llvm.ptr<1>) attributes {allocation.offset = 0 : i32} {
    ttg.warp_specialize(%value, %output) attributes {allocation.offset = 0 : i32, warpGroupStartIds = array<i32: 8>}
    default {
      nvvm.barrier0
      %x = llvm.call @plain_wrapper(%value) : (i32) -> i32
      %y = llvm.call @collective_wrapper(%x) : (i32) -> i32
      llvm.store %y, %output : i32, !llvm.ptr<1>
      ttg.warp_yield
    }
    partition0(%arg: i32, %ptr: !llvm.ptr<1>) num_warps(4) {
      nvvm.barrier0
      %x = llvm.call @plain_wrapper(%arg) : (i32) -> i32
      %y = llvm.call @collective_wrapper(%x) : (i32) -> i32
      llvm.store %y, %ptr : i32, !llvm.ptr<1>
      ttg.warp_return
    } : (i32, !llvm.ptr<1>) -> ()
    llvm.return
  }
}
