// RUN: triton-opt %s -split-input-file -triton-tle-lower-barriers -triton-tle-lower-tma-copy | FileCheck %s

#nvmma = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 32}>
#bar_shared = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [1, 0]}>
#bar_slot = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>
#smem = #ttg.shared_memory

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 32 : i32, ttg.target = "cuda:90"} {
  // CHECK-LABEL: tt.func @explicit_tma_barrier
  tt.func @explicit_tma_barrier(%desc: !tt.tensordesc<tensor<32x64xf32, #nvmma>>) {
    %c0 = arith.constant 0 : i32
    %dst = ttg.local_alloc : () -> !ttg.memdesc<32x64xf32, #nvmma, #smem, mutable>
    // CHECK: %[[BARS:.*]] = ttg.local_alloc : () -> !ttg.memdesc<1x1xi64
    // CHECK: ttng.init_barrier
    %bars = tle.barrier.alloc {arrive_count = 1 : i32, expect_bytes = 8192 : i32, init_polarity = 0 : i32, num_barriers = 1 : i32} : !ttg.memdesc<1x1xi64, #bar_shared, #smem, mutable>
    %slot = ttg.memdesc_index %bars[%c0] : !ttg.memdesc<1x1xi64, #bar_shared, #smem, mutable> -> !ttg.memdesc<1xi64, #bar_slot, #smem, mutable>

    // CHECK-NOT: ttg.local_alloc : () -> !ttg.memdesc<1xi64
    // CHECK: ttng.barrier_expect %[[SLOT:.*]], 8192
    // CHECK: ttng.async_tma_copy_global_to_local {{.*}} %[[SLOT]], %true
    // CHECK-NOT: ttng.inval_barrier %[[SLOT]]
    // CHECK: ttng.wait_barrier %[[SLOT]]
    ttg.tma_copy %desc, %dst, [%c0, %c0], barrier %slot {expect_bytes = 8192 : i32} : !tt.tensordesc<tensor<32x64xf32, #nvmma>>, !ttg.memdesc<32x64xf32, #nvmma, #smem, mutable>, !ttg.memdesc<1xi64, #bar_slot, #smem, mutable>
    tle.barrier.wait %slot, %c0 {backend = "mbarrier", named_id = 0 : i32, named_num_threads = 0 : i32} : !ttg.memdesc<1xi64, #bar_slot, #smem, mutable>
    tt.return
  }
}

// -----

#nvmma = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 32}>
#bar_shared = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [1, 0]}>
#bar_slot = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>
#smem = #ttg.shared_memory

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 32 : i32, ttg.target = "cuda:90"} {
  // CHECK-LABEL: tt.func @explicit_tma_barrier_two_slots
  tt.func @explicit_tma_barrier_two_slots(%a_desc: !tt.tensordesc<tensor<32x64xf32, #nvmma>>, %b_desc: !tt.tensordesc<tensor<32x64xf32, #nvmma>>) {
    %c0 = arith.constant 0 : i32
    %c1 = arith.constant 1 : i32
    %a = ttg.local_alloc : () -> !ttg.memdesc<32x64xf32, #nvmma, #smem, mutable>
    %b = ttg.local_alloc : () -> !ttg.memdesc<32x64xf32, #nvmma, #smem, mutable>
    %bars = tle.barrier.alloc {arrive_count = 1 : i32, expect_bytes = 8192 : i32, init_polarity = 0 : i32, num_barriers = 2 : i32} : !ttg.memdesc<2x1xi64, #bar_shared, #smem, mutable>
    %slot0 = ttg.memdesc_index %bars[%c0] : !ttg.memdesc<2x1xi64, #bar_shared, #smem, mutable> -> !ttg.memdesc<1xi64, #bar_slot, #smem, mutable>
    %slot1 = ttg.memdesc_index %bars[%c1] : !ttg.memdesc<2x1xi64, #bar_shared, #smem, mutable> -> !ttg.memdesc<1xi64, #bar_slot, #smem, mutable>

    // CHECK: ttng.barrier_expect %[[SLOT0:.*]], 8192
    // CHECK: ttng.async_tma_copy_global_to_local {{.*}} %[[SLOT0]], %true
    ttg.tma_copy %a_desc, %a, [%c0, %c0], barrier %slot0 {expect_bytes = 8192 : i32} : !tt.tensordesc<tensor<32x64xf32, #nvmma>>, !ttg.memdesc<32x64xf32, #nvmma, #smem, mutable>, !ttg.memdesc<1xi64, #bar_slot, #smem, mutable>
    // CHECK: ttng.barrier_expect %[[SLOT1:.*]], 8192
    // CHECK: ttng.async_tma_copy_global_to_local {{.*}} %[[SLOT1]], %true
    ttg.tma_copy %b_desc, %b, [%c0, %c0], barrier %slot1 {expect_bytes = 8192 : i32} : !tt.tensordesc<tensor<32x64xf32, #nvmma>>, !ttg.memdesc<32x64xf32, #nvmma, #smem, mutable>, !ttg.memdesc<1xi64, #bar_slot, #smem, mutable>
    tt.return
  }
}
