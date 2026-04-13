// RUN: triton-opt %s -split-input-file --triton-tle-promote-local-store-staging | FileCheck %s

#blocked = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [32, 1], warpsPerCTA = [1, 1], order = [1, 0]}>
#shared = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [1, 0]}>
#smem = #ttg.shared_memory
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 1 : i32, ttg.target = "cuda:90", "ttg.threads-per-warp" = 32 : i32} {
  // CHECK-LABEL: tt.func @promote_global_load_staging
  tt.func @promote_global_load_staging(%ptr: tensor<32x64x!tt.ptr<bf16>, #blocked>) -> tensor<32x64xbf16, #blocked> {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c4 = arith.constant 4 : index
    %zero = arith.constant dense<0.000000e+00> : tensor<32x64xbf16, #blocked>
    %smem = ttg.local_alloc : () -> !ttg.memdesc<32x64xbf16, #shared, #smem, mutable>
    %out = scf.for %i = %c0 to %c4 step %c1 iter_args(%acc = %zero) -> tensor<32x64xbf16, #blocked> {
      // CHECK: %[[V:.+]] = tt.load
      %v = tt.load %ptr : tensor<32x64x!tt.ptr<bf16>, #blocked>
      // CHECK-NEXT: %[[STAGE:.+]] = ttg.local_alloc %[[V]]
      ttg.local_store %v, %smem : tensor<32x64xbf16, #blocked> -> !ttg.memdesc<32x64xbf16, #shared, #smem, mutable>
      // CHECK-NOT: ttg.local_store
      // CHECK-NOT: gpu.barrier
      // CHECK: ttg.local_load %[[STAGE]]
      gpu.barrier
      %r = ttg.local_load %smem : !ttg.memdesc<32x64xbf16, #shared, #smem, mutable> -> tensor<32x64xbf16, #blocked>
      scf.yield %r : tensor<32x64xbf16, #blocked>
    } {tt.num_stages = 2 : i32}
    tt.return %out : tensor<32x64xbf16, #blocked>
  }

  // CHECK-LABEL: tt.func @keep_single_stage_staging
  tt.func @keep_single_stage_staging(%ptr: tensor<32x64x!tt.ptr<bf16>, #blocked>) -> tensor<32x64xbf16, #blocked> {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c4 = arith.constant 4 : index
    %zero = arith.constant dense<0.000000e+00> : tensor<32x64xbf16, #blocked>
    %smem = ttg.local_alloc : () -> !ttg.memdesc<32x64xbf16, #shared, #smem, mutable>
    %out = scf.for %i = %c0 to %c4 step %c1 iter_args(%acc = %zero) -> tensor<32x64xbf16, #blocked> {
      %v = tt.load %ptr : tensor<32x64x!tt.ptr<bf16>, #blocked>
      // CHECK: ttg.local_store
      ttg.local_store %v, %smem : tensor<32x64xbf16, #blocked> -> !ttg.memdesc<32x64xbf16, #shared, #smem, mutable>
      // CHECK: gpu.barrier
      gpu.barrier
      %r = ttg.local_load %smem : !ttg.memdesc<32x64xbf16, #shared, #smem, mutable> -> tensor<32x64xbf16, #blocked>
      scf.yield %r : tensor<32x64xbf16, #blocked>
    } {tt.num_stages = 1 : i32}
    tt.return %out : tensor<32x64xbf16, #blocked>
  }
}
