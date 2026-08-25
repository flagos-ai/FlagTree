// RUN: triton-opt %s --convert-triton-gpu-to-llvm='compute-capability=90 ptx-version=81' -reconcile-unrealized-casts | FileCheck %s

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:90", "ttg.threads-per-warp" = 32 : i32} {
  // CHECK: llvm.mlir.global internal constant @__tle_barrier_offsets_0
  // CHECK-SAME: addr_space = 4
  // CHECK-LABEL: llvm.func @lower_group
  // CHECK: nvvm.read.ptx.sreg.tid.x
  // CHECK: llvm.load
  // CHECK: mbarrier.init.shared::cta.b64
  // CHECK: mbarrier.init.shared::cta.b64
  // CHECK-NOT: tle.init_barrier_group
  tt.func @lower_group() {
    tle.init_barrier_group {counts = array<i32: 1, 256>, offsets = array<i32: 0, 8>, worker_count = 128 : i32}
    tt.return
  }
}
