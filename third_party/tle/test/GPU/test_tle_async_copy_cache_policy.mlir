// RUN: triton-opt %s -split-input-file --allocate-shared-memory-nv='compute-capability=90 ptx-version=81' --convert-triton-gpu-to-llvm='compute-capability=90 ptx-version=81' -reconcile-unrealized-casts | FileCheck %s

#blocked = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [4, 1], order = [1, 0]}>
#shared = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [1, 0]}>
#smem = #ttg.shared_memory

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:90", "ttg.threads-per-warp" = 32 : i32} {
  // CHECK-LABEL: llvm.func @async_copy_eviction_policy
  // CHECK: createpolicy.fractional.L2::evict_last.b64
  // CHECK: cp.async.cg.shared.global.L2::cache_hint
  // CHECK-NOT: L2::256B
  tt.func @async_copy_eviction_policy(%gptr: tensor<1x4x!tt.ptr<f32>, #blocked>) {
    %c0 = arith.constant dense<0.000000e+00> : tensor<1x4xf32, #blocked>
    %mask = arith.constant dense<true> : tensor<1x4xi1, #blocked>
    %smem = ttg.local_alloc : () -> !ttg.memdesc<1x4xf32, #shared, #smem, mutable>
    ttg.async_copy_global_to_local %gptr, %smem mask %mask other %c0 cacheModifier = cg evictionPolicy = evict_last : tensor<1x4x!tt.ptr<f32>, #blocked> -> <1x4xf32, #shared, #smem, mutable>
    tt.return
  }
}
