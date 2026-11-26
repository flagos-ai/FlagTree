// Copyright 2026 FlagOS Contributors
// SPDX-License-Identifier: MIT
// RUN: triton-opt %s -triton-tle-lower-barriers | FileCheck %s

#shared = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [1, 0]}>
#slot = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>
#smem = #ttg.shared_memory
!array = !ttg.memdesc<1x1xi64, #shared, #smem, mutable>
!slot = !ttg.memdesc<1xi64, #slot, #smem, mutable>

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32} {
  // Like MiniMax sparse prefill, one path returns before initialization.
  // CHECK-LABEL: tt.func @early_return
  tt.func @early_return(%skip: i1, %done: i1) {
    // CHECK: cf.cond_br %{{.*}}, ^[[SKIP:bb[0-9]+]], ^[[INIT:bb[0-9]+]]
    cf.cond_br %skip, ^skip, ^init
  ^skip:
    // CHECK: ^[[SKIP]]:
    // CHECK-NEXT: tt.return
    tt.return
  ^init:
    // CHECK: ^[[INIT]]:
    %c0 = arith.constant 0 : i32
    %bar = tle.barrier.alloc {arrive_count = 1 : i32, init_polarity = 0 : i32, num_barriers = 1 : i32} : !array
    %s = ttg.memdesc_index %bar[%c0] : !array -> !slot
    // CHECK: %[[SLOT:.*]] = ttg.memdesc_index
    // CHECK-NEXT: ttng.init_barrier %[[SLOT]]
    // CHECK-NEXT: gpu.barrier
    // CHECK: cf.cond_br %{{.*}}, ^[[EXIT:bb[0-9]+]], ^[[USE:bb[0-9]+]]
    cf.cond_br %done, ^exit, ^use(%s : !slot)
  ^use(%alias: !slot):
    // CHECK: ^[[USE]](%[[ALIAS:[a-zA-Z0-9_]+]]:
    tle.barrier.arrive %alias, %c0 {arrive_count = 1 : i32, backend = "mbarrier", named_id = 0 : i32, named_num_threads = 0 : i32} : !slot
    tle.barrier.wait %alias, %c0 {backend = "mbarrier", named_id = 0 : i32, named_num_threads = 0 : i32} : !slot
    // CHECK: ttng.wait_barrier %[[ALIAS]]
    // CHECK-NEXT: gpu.barrier
    // CHECK-NEXT: ttng.inval_barrier %[[SLOT]]
    // CHECK-NEXT: gpu.barrier
    // CHECK-NEXT: tt.return
    tt.return
  ^exit:
    // CHECK: ^[[EXIT]]:
    // CHECK-NEXT: gpu.barrier
    // CHECK-NEXT: ttng.inval_barrier %[[SLOT]]
    // CHECK-NEXT: gpu.barrier
    // CHECK-NEXT: tt.return
    tt.return
  }

  // A shared continuation is also reachable without initialization. Keep the
  // barrier live on the other successor of the conditional branch.
  // CHECK-LABEL: tt.func @conditional_join
  tt.func @conditional_join(%skip: i1, %done: i1, %value: i32) -> i32 {
    // CHECK: cf.cond_br %{{.*}}, ^[[JOIN:bb[0-9]+]](%{{.*}} : i32), ^[[INIT:bb[0-9]+]]
    cf.cond_br %skip, ^join(%value : i32), ^init
  ^init:
    // CHECK: ^[[INIT]]:
    %c0 = arith.constant 0 : i32
    %bar = tle.barrier.alloc {arrive_count = 1 : i32, init_polarity = 0 : i32, num_barriers = 1 : i32} : !array
    %s = ttg.memdesc_index %bar[%c0] : !array -> !slot
    // CHECK: %[[SLOT:.*]] = ttg.memdesc_index
    // CHECK-NEXT: ttng.init_barrier %[[SLOT]]
    // CHECK: cf.cond_br %{{.*}}, ^[[RETIRE:bb[0-9]+]](%{{.*}} : i32), ^[[USE:bb[0-9]+]]
    cf.cond_br %done, ^join(%value : i32), ^use
  ^use:
    // CHECK: ^[[USE]]:
    tle.barrier.arrive %s, %c0 {arrive_count = 1 : i32, backend = "mbarrier", named_id = 0 : i32, named_num_threads = 0 : i32} : !slot
    tle.barrier.wait %s, %c0 {backend = "mbarrier", named_id = 0 : i32, named_num_threads = 0 : i32} : !slot
    // CHECK: ttng.wait_barrier
    // CHECK-NEXT: gpu.barrier
    // CHECK-NEXT: ttng.inval_barrier %[[SLOT]]
    // CHECK-NEXT: gpu.barrier
    // CHECK-NEXT: cf.br ^[[JOIN]]
    cf.br ^join(%c0 : i32)
  ^join(%result: i32):
    // CHECK: ^[[RETIRE]](%[[VALUE:.*]]: i32):
    // CHECK-NEXT: gpu.barrier
    // CHECK-NEXT: ttng.inval_barrier %[[SLOT]]
    // CHECK-NEXT: gpu.barrier
    // CHECK-NEXT: cf.br ^[[JOIN]](%[[VALUE]] : i32)
    // CHECK: ^[[JOIN]](%[[RESULT:.*]]: i32):
    // CHECK-NEXT: tt.return %[[RESULT]]
    tt.return %result : i32
  }

  // Executing the allocation again must retire the previous instance first.
  // CHECK-LABEL: tt.func @reinitialize_on_backedge
  tt.func @reinitialize_on_backedge(%again: i1) {
    %c0 = arith.constant 0 : i32
    // CHECK: cf.br ^[[INIT:bb[0-9]+]]
    // CHECK: ^[[RETIRE:bb[0-9]+]]:
    // CHECK-NEXT: gpu.barrier
    // CHECK-NEXT: ttng.inval_barrier %[[SLOT:[a-zA-Z0-9_]+]]
    // CHECK-NEXT: gpu.barrier
    // CHECK-NEXT: cf.br ^[[INIT]]
    cf.br ^init
  ^init:
    // CHECK: ^[[INIT]]:
    %bar = tle.barrier.alloc {arrive_count = 1 : i32, init_polarity = 0 : i32, num_barriers = 1 : i32} : !array
    %s = ttg.memdesc_index %bar[%c0] : !array -> !slot
    tle.barrier.arrive %s, %c0 {arrive_count = 1 : i32, backend = "mbarrier", named_id = 0 : i32, named_num_threads = 0 : i32} : !slot
    tle.barrier.wait %s, %c0 {backend = "mbarrier", named_id = 0 : i32, named_num_threads = 0 : i32} : !slot
    // CHECK: %[[SLOT]] = ttg.memdesc_index
    // CHECK-NEXT: ttng.init_barrier %[[SLOT]]
    // CHECK: cf.cond_br %{{.*}}, ^[[RETIRE]], ^[[EXIT:bb[0-9]+]]
    cf.cond_br %again, ^init, ^exit
  ^exit:
    // CHECK: ^[[EXIT]]:
    // CHECK-NEXT: gpu.barrier
    // CHECK-NEXT: ttng.inval_barrier %[[SLOT]]
    // CHECK-NEXT: gpu.barrier
    // CHECK-NEXT: tt.return
    tt.return
  }
}
