// RUN: triton-opt %s -split-input-file --allocate-shared-memory-ppu --convert-triton-gpu-to-llvm-ppu | FileCheck %s

// The barrier TleInsertLocalPointerBarriers emits between the shared-memory
// accesses of tle.gpu.local_ptr carries tle.shared_only: it orders shared
// memory, never global memory. PPU lowers it to a hardware barrier with the
// shared-memory fence (1). A plain gpu.barrier keeps the lowering of
// mlir::gpu::BarrierOp, which reaches ppu-llc as barrier0 and also drains the
// warp's outstanding global accesses.

#blocked = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>

module attributes {"ttg.num-warps" = 4 : i32, "ttg.num-ctas" = 1 : i32, ttg.target = "ppu:80", "ttg.threads-per-warp" = 32 : i32} {
  // CHECK-LABEL: @shared_only_barrier
  tt.func @shared_only_barrier() {
    // CHECK-DAG: %[[ID:.*]] = llvm.mlir.constant(0 : i32)
    // CHECK-DAG: %[[THREADS:.*]] = llvm.mlir.constant(128 : i32)
    // CHECK-DAG: %[[FENCE:.*]] = llvm.mlir.constant(1 : i32)
    // CHECK: llvm.call @__ppukernel_barrier_sync_cnt_void(%[[ID]], %[[THREADS]], %[[FENCE]])
    // CHECK-NOT: nvvm.barrier0
    gpu.barrier {tle.shared_only}
    tt.return
  }
}

// -----

#blocked = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>

module attributes {"ttg.num-warps" = 4 : i32, "ttg.num-ctas" = 1 : i32, ttg.target = "ppu:80", "ttg.threads-per-warp" = 32 : i32} {
  // CHECK-LABEL: @plain_barrier
  tt.func @plain_barrier() {
    // CHECK-NOT: __ppukernel_barrier_sync_cnt_void
    // CHECK: nvvm.barrier0
    gpu.barrier
    tt.return
  }
}
