// Copyright 2026 FlagOS Contributors
// SPDX-License-Identifier: MIT
// Same full counts two per-copy contributions before one consumer wait.
// RUN: triton-opt %s --triton-nvidia-optimize-descriptor-encoding --triton-tle-lower-barriers --triton-tle-lower-tma-copy | FileCheck %s
#root = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [2, 1, 0]}>
#row = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [1, 0]}>
#bar = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>
#smem = #ttg.shared_memory
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 8 : i32, "ttg.threads-per-warp" = 32 : i32, ttg.target = "cuda:90"} {
  // CHECK-LABEL: tt.func public @joint_full_dynamic_stage
  tt.func public @joint_full_dynamic_stage(%wraw: !tt.ptr<i8>, %xraw: !tt.ptr<i8>, %stage: i32) {
    %c0 = arith.constant 0 : i32
    %wd = ttng.reinterpret_tensor_descriptor %wraw {tle.copy_bound_tensor_map} : !tt.ptr<i8> to !tt.tensordesc<tensor<64x256xbf16>>
    %xd = ttng.reinterpret_tensor_descriptor %xraw {tle.copy_bound_tensor_map} : !tt.ptr<i8> to !tt.tensordesc<tensor<1x64xbf16>>
    %ws = ttg.local_alloc : () -> !ttg.memdesc<3x64x256xbf16, #root, #smem, mutable>
    %xs = ttg.local_alloc : () -> !ttg.memdesc<3x1x64xbf16, #root, #smem, mutable>
    %w = ttg.memdesc_index %ws[%stage] : !ttg.memdesc<3x64x256xbf16, #root, #smem, mutable> -> !ttg.memdesc<64x256xbf16, #row, #smem, mutable>
    %x = ttg.memdesc_index %xs[%stage] : !ttg.memdesc<3x1x64xbf16, #root, #smem, mutable> -> !ttg.memdesc<1x64xbf16, #row, #smem, mutable>
    // CHECK: ttng.init_barrier {{.*}}, 2
    %full = tle.barrier.alloc {arrive_count = 2 : i32, expect_bytes = 32768 : i32, init_polarity = 0 : i32, num_barriers = 1 : i32} : !ttg.memdesc<1x1xi64, #row, #smem, mutable>
    %slot = ttg.memdesc_index %full[%c0] : !ttg.memdesc<1x1xi64, #row, #smem, mutable> -> !ttg.memdesc<1xi64, #bar, #smem, mutable>
    // CHECK: ttg.memdesc_reinterpret
    // CHECK: ttng.barrier_expect %[[FULL:.*]], 32768
    // CHECK: ttng.async_tma_copy_global_to_local {{.*}} %[[FULL]], %true
    ttg.tma_copy %wd, %w, [%c0, %c0], barrier %slot {expect_bytes = 32768 : i32} : !tt.tensordesc<tensor<64x256xbf16>>, !ttg.memdesc<64x256xbf16, #row, #smem, mutable>, !ttg.memdesc<1xi64, #bar, #smem, mutable>
    // CHECK: ttg.memdesc_reinterpret
    // CHECK: ttng.barrier_expect %[[FULL]], 128
    // CHECK: ttng.async_tma_copy_global_to_local {{.*}} %[[FULL]], %true
    ttg.tma_copy %xd, %x, [%c0, %c0], barrier %slot {expect_bytes = 128 : i32} : !tt.tensordesc<tensor<1x64xbf16>>, !ttg.memdesc<1x64xbf16, #row, #smem, mutable>, !ttg.memdesc<1xi64, #bar, #smem, mutable>
    // CHECK-NOT: ttng.wait_barrier
    // CHECK: ttng.wait_barrier %[[FULL]]
    // CHECK-NOT: ttg.tma_copy
    // CHECK-NOT: ttng.wait_barrier
    tle.barrier.wait %slot, %c0 {backend = "mbarrier", named_id = 0 : i32, named_num_threads = 0 : i32} : !ttg.memdesc<1xi64, #bar, #smem, mutable>
    tt.return
  }
}
