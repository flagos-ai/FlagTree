// RUN: triton-opt %s -split-input-file -convert-warp-specialize-to-llvm | FileCheck %s

// Changing the physical warp groups must not reuse a named barrier before the
// preceding region has joined, even when the kernel has no shared memory.
module attributes {"ttg.num-warps" = 4 : i32, "ttg.total-num-warps" = 4 : i32, ttg.shared = 0 : i32} {
  // CHECK-LABEL: llvm.func @reuse_barrier_domains
  // CHECK: %[[THREADS0:.*]] = llvm.mlir.constant(128 : i32)
  // CHECK-NEXT: %[[ZERO0:.*]] = llvm.mlir.constant(0 : i32)
  // CHECK-NEXT: nvvm.barrier id = %[[ZERO0]] number_of_threads = %[[THREADS0]]
  // CHECK: llvm.switch
  // CHECK: %[[THREADS1:.*]] = llvm.mlir.constant(64 : i32)
  // CHECK-NEXT: %[[TWO:.*]] = llvm.mlir.constant(2 : i32)
  // CHECK-NEXT: nvvm.barrier id = %[[TWO]] number_of_threads = %[[THREADS1]]
  // CHECK: %[[THREADS2:.*]] = llvm.mlir.constant(64 : i32)
  // CHECK-NEXT: %[[THREE:.*]] = llvm.mlir.constant(3 : i32)
  // CHECK-NEXT: nvvm.barrier id = %[[THREE]] number_of_threads = %[[THREADS2]]
  // CHECK: llvm.br ^[[JOIN:bb[0-9]+]]
  // CHECK: ^[[JOIN]]:
  // CHECK-NEXT: %[[THREADS3:.*]] = llvm.mlir.constant(128 : i32)
  // CHECK-NEXT: %[[ZERO1:.*]] = llvm.mlir.constant(0 : i32)
  // CHECK-NEXT: nvvm.barrier id = %[[ZERO1]] number_of_threads = %[[THREADS3]]
  // CHECK: llvm.switch
  // CHECK: nvvm.bar.warp.sync
  // CHECK: %[[THREADS4:.*]] = llvm.mlir.constant(64 : i32)
  // CHECK-NEXT: %[[REUSED:.*]] = llvm.mlir.constant(2 : i32)
  // CHECK-NEXT: nvvm.barrier id = %[[REUSED]] number_of_threads = %[[THREADS4]]
  // CHECK: nvvm.bar.warp.sync
  // CHECK: %[[THREADS5:.*]] = llvm.mlir.constant(128 : i32)
  // CHECK-NEXT: %[[ZERO2:.*]] = llvm.mlir.constant(0 : i32)
  // CHECK-NEXT: nvvm.barrier id = %[[ZERO2]] number_of_threads = %[[THREADS5]]
  // CHECK-NEXT: llvm.return
  llvm.func @reuse_barrier_domains() {
    ttg.warp_specialize() attributes {reuseDefaultWarps = true, warpGroupStartIds = array<i32: 0, 2>}
    default { ttg.warp_yield }
    partition0() num_warps(2) {
      nvvm.barrier0
      ttg.warp_return
    }
    partition1() num_warps(2) {
      nvvm.barrier0
      ttg.warp_return
    } : () -> ()
    ttg.warp_specialize() attributes {reuseDefaultWarps = true, warpGroupStartIds = array<i32: 0, 1, 3>}
    default { ttg.warp_yield }
    partition0() num_warps(1) {
      nvvm.barrier0
      ttg.warp_return
    }
    partition1() num_warps(2) {
      nvvm.barrier0
      ttg.warp_return
    }
    partition2() num_warps(1) {
      nvvm.barrier0
      ttg.warp_return
    } : () -> ()
    llvm.return
  }
}

// -----

// TLE named arrive/wait operations have already become LLVM intrinsics by this
// lowering stage. They still require a CTA boundary join, while partitions do
// not synchronize with one another.
module attributes {"ttg.num-warps" = 4 : i32, "ttg.total-num-warps" = 4 : i32, ttg.shared = 0 : i32} {
  // CHECK-LABEL: llvm.func @reuse_named_barrier_intrinsics
  // CHECK: %[[THREADS0:.*]] = llvm.mlir.constant(128 : i32)
  // CHECK-NEXT: %[[ZERO0:.*]] = llvm.mlir.constant(0 : i32)
  // CHECK-NEXT: nvvm.barrier id = %[[ZERO0]] number_of_threads = %[[THREADS0]]
  // CHECK: llvm.switch
  // CHECK: llvm.call_intrinsic "llvm.nvvm.barrier.cta.arrive.aligned.count"
  // CHECK: llvm.call_intrinsic "llvm.nvvm.barrier.cta.sync.aligned.count"
  // CHECK: llvm.br ^[[JOIN:bb[0-9]+]]
  // CHECK: ^[[JOIN]]:
  // CHECK-NEXT: %[[THREADS1:.*]] = llvm.mlir.constant(128 : i32)
  // CHECK-NEXT: %[[ZERO1:.*]] = llvm.mlir.constant(0 : i32)
  // CHECK-NEXT: nvvm.barrier id = %[[ZERO1]] number_of_threads = %[[THREADS1]]
  // CHECK-NEXT: llvm.return
  llvm.func @reuse_named_barrier_intrinsics() {
    ttg.warp_specialize() attributes {reuseDefaultWarps = true, warpGroupStartIds = array<i32: 0, 2>}
    default { ttg.warp_yield }
    partition0() num_warps(2) {
      %bar = llvm.mlir.constant(4 : i32) : i32
      %count = llvm.mlir.constant(64 : i32) : i32
      llvm.call_intrinsic "llvm.nvvm.barrier.cta.arrive.aligned.count"(%bar, %count) : (i32, i32) -> ()
      llvm.call_intrinsic "llvm.nvvm.barrier.cta.sync.aligned.count"(%bar, %count) : (i32, i32) -> ()
      ttg.warp_return
    }
    partition1() num_warps(2) { ttg.warp_return } : () -> ()
    llvm.return
  }
}

// -----

// Single-warp barriers and empty partitions consume no named barrier IDs.
module attributes {"ttg.num-warps" = 8 : i32, "ttg.total-num-warps" = 8 : i32, ttg.shared = 0 : i32} {
  // CHECK-LABEL: llvm.func @dense_reuse_barrier_ids
  // CHECK: llvm.switch
  // CHECK: nvvm.bar.warp.sync
  // CHECK: %[[THREADS0:.*]] = llvm.mlir.constant(64 : i32)
  // CHECK-NEXT: %[[TWO:.*]] = llvm.mlir.constant(2 : i32)
  // CHECK-NEXT: nvvm.barrier id = %[[TWO]] number_of_threads = %[[THREADS0]]
  // CHECK: %[[THREADS1:.*]] = llvm.mlir.constant(64 : i32)
  // CHECK-NEXT: %[[THREE:.*]] = llvm.mlir.constant(3 : i32)
  // CHECK-NEXT: nvvm.barrier id = %[[THREE]] number_of_threads = %[[THREADS1]]
  // CHECK: nvvm.bar.warp.sync
  // CHECK: llvm.return
  llvm.func @dense_reuse_barrier_ids() {
    ttg.warp_specialize() attributes {reuseDefaultWarps = true, warpGroupStartIds = array<i32: 0, 1, 3, 5, 7>}
    default { ttg.warp_yield }
    partition0() num_warps(1) {
      nvvm.barrier0
      ttg.warp_return
    }
    partition1() num_warps(2) { ttg.warp_return }
    partition2() num_warps(2) {
      nvvm.barrier0
      ttg.warp_return
    }
    partition3() num_warps(2) {
      nvvm.barrier0
      ttg.warp_return
    }
    partition4() num_warps(1) {
      nvvm.barrier0
      ttg.warp_return
    } : () -> ()
    llvm.return
  }
}

// -----

// Sixteen warp-local barriers need no named barriers or CTA boundary sync.
module attributes {"ttg.num-warps" = 16 : i32, "ttg.total-num-warps" = 16 : i32, ttg.shared = 0 : i32} {
  // CHECK-LABEL: llvm.func @sixteen_warp_local_barriers
  // CHECK-NOT: nvvm.barrier
  // CHECK: llvm.switch
  // CHECK-NOT: nvvm.barrier
  // CHECK: nvvm.bar.warp.sync
  // CHECK-NOT: nvvm.barrier
  // CHECK: nvvm.bar.warp.sync
  // CHECK-NOT: nvvm.barrier
  // CHECK: nvvm.bar.warp.sync
  // CHECK-NOT: nvvm.barrier
  // CHECK: nvvm.bar.warp.sync
  // CHECK-NOT: nvvm.barrier
  // CHECK: nvvm.bar.warp.sync
  // CHECK-NOT: nvvm.barrier
  // CHECK: nvvm.bar.warp.sync
  // CHECK-NOT: nvvm.barrier
  // CHECK: nvvm.bar.warp.sync
  // CHECK-NOT: nvvm.barrier
  // CHECK: nvvm.bar.warp.sync
  // CHECK-NOT: nvvm.barrier
  // CHECK: nvvm.bar.warp.sync
  // CHECK-NOT: nvvm.barrier
  // CHECK: nvvm.bar.warp.sync
  // CHECK-NOT: nvvm.barrier
  // CHECK: nvvm.bar.warp.sync
  // CHECK-NOT: nvvm.barrier
  // CHECK: nvvm.bar.warp.sync
  // CHECK-NOT: nvvm.barrier
  // CHECK: nvvm.bar.warp.sync
  // CHECK-NOT: nvvm.barrier
  // CHECK: nvvm.bar.warp.sync
  // CHECK-NOT: nvvm.barrier
  // CHECK: nvvm.bar.warp.sync
  // CHECK-NOT: nvvm.barrier
  // CHECK: nvvm.bar.warp.sync
  // CHECK-NOT: nvvm.barrier
  // CHECK: llvm.return
  llvm.func @sixteen_warp_local_barriers() {
    ttg.warp_specialize() attributes {reuseDefaultWarps = true, warpGroupStartIds = array<i32: 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15>}
    default { ttg.warp_yield }
    partition0() num_warps(1) {
      nvvm.barrier0
      ttg.warp_return
    }
    partition1() num_warps(1) {
      nvvm.barrier0
      ttg.warp_return
    }
    partition2() num_warps(1) {
      nvvm.barrier0
      ttg.warp_return
    }
    partition3() num_warps(1) {
      nvvm.barrier0
      ttg.warp_return
    }
    partition4() num_warps(1) {
      nvvm.barrier0
      ttg.warp_return
    }
    partition5() num_warps(1) {
      nvvm.barrier0
      ttg.warp_return
    }
    partition6() num_warps(1) {
      nvvm.barrier0
      ttg.warp_return
    }
    partition7() num_warps(1) {
      nvvm.barrier0
      ttg.warp_return
    }
    partition8() num_warps(1) {
      nvvm.barrier0
      ttg.warp_return
    }
    partition9() num_warps(1) {
      nvvm.barrier0
      ttg.warp_return
    }
    partition10() num_warps(1) {
      nvvm.barrier0
      ttg.warp_return
    }
    partition11() num_warps(1) {
      nvvm.barrier0
      ttg.warp_return
    }
    partition12() num_warps(1) {
      nvvm.barrier0
      ttg.warp_return
    }
    partition13() num_warps(1) {
      nvvm.barrier0
      ttg.warp_return
    }
    partition14() num_warps(1) {
      nvvm.barrier0
      ttg.warp_return
    }
    partition15() num_warps(1) {
      nvvm.barrier0
      ttg.warp_return
    } : () -> ()
    llvm.return
  }
}
