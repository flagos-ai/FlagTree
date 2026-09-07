// RUN: triton-opt %s -convert-triton-gpu-to-llvm='compute-capability=90 ptx-version=80' | FileCheck %s

// A restored pipe role is a real function, not a region nested in
// warp_specialize. TMA and thread election must keep its local warp count
// after function conversion instead of using the entry's eight warps.
module attributes {"ttg.num-warps" = 8 : i32, "ttg.num-ctas" = 1 : i32, "ttg.threads-per-warp" = 32 : i32, ttg.target = "cuda:90"} {
  // CHECK-LABEL: llvm.func internal @producer
  // CHECK-SAME: "ttg.num-warps" = 1 : i32
  tt.func private @producer() attributes {noinline = true, "ttg.num-warps" = 1 : i32} {
    tt.return
  }
  // CHECK-LABEL: llvm.func internal @consumer
  // CHECK-SAME: "ttg.num-warps" = 8 : i32
  tt.func private @consumer() attributes {noinline = true, "ttg.num-warps" = 8 : i32} {
    tt.return
  }
}
