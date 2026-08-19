// RUN: triton-opt %s --allocate-shared-memory-nv='compute-capability=90 ptx-version=81' --convert-triton-gpu-to-llvm='compute-capability=90 ptx-version=81' -reconcile-unrealized-casts | FileCheck %s

#blocked = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [4, 1], order = [1, 0]}>
#shared = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [1, 0]}>
#smem = #ttg.shared_memory

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:90", "ttg.threads-per-warp" = 32 : i32} {
  // CHECK-LABEL: llvm.func @memdesc_alias_to_llvm
  tt.func @memdesc_alias_to_llvm(%arg0: tensor<64x64xbf16, #blocked>) {
    %src = ttg.local_alloc %arg0 : (tensor<64x64xbf16, #blocked>) -> !ttg.memdesc<64x64xbf16, #shared, #smem, mutable>
    %alias = tle.memdesc_alias %src {offset_bytes = 128 : i64} : !ttg.memdesc<64x64xbf16, #shared, #smem, mutable> -> !ttg.memdesc<1x1xbf16, #shared, #smem, mutable>
    %value = ttg.local_load %alias : !ttg.memdesc<1x1xbf16, #shared, #smem, mutable> -> tensor<1x1xbf16, #blocked>
    tt.return
  }
}

// CHECK: llvm.getelementptr
// CHECK-NOT: tle.memdesc_alias
