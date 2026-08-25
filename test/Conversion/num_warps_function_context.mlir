// RUN: triton-opt %s --allocate-shared-memory-nv --convert-triton-gpu-to-llvm -reconcile-unrealized-casts | FileCheck %s

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:90", "ttg.threads-per-warp" = 32 : i32} {
  tt.func public @kernel() {
    tt.call @one_warp_helper() : () -> ()
    tt.return
  }

  // CHECK-LABEL: llvm.func internal @one_warp_helper
  // CHECK-SAME: nvvm.reqntid = array<i32: 32>
  // CHECK-SAME: "ttg.num-warps" = 1 : i32
  tt.func private @one_warp_helper() attributes {"ttg.num-warps" = 1 : i32} {
    tt.return
  }
}
