// RUN: triton-opt %s -pass-pipeline='builtin.module(allocate-shared-memory-nv{compute-capability=90 ptx-version=80}, tritongpu-global-scratch-memory-allocation, convert-triton-gpu-to-llvm{compute-capability=90 ptx-version=80})' | FileCheck %s

#blocked = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>
#shared = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>
#smem = #ttg.shared_memory

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:90", "ttg.threads-per-warp" = 32 : i32} {
  tt.func public @kernel() {
    %c0_i32 = arith.constant 0 : i32
    %ones = arith.constant dense<1> : tensor<128xi32, #blocked>
    %pred = arith.constant dense<true> : tensor<128xi1, #blocked>
    %smem = ttg.local_alloc : () -> !ttg.memdesc<16xi32, #shared, #smem, mutable>
    %counter_local_ptr = "tle.local_pointers"(%smem, %c0_i32) : (!ttg.memdesc<16xi32, #shared, #smem, mutable>, i32) -> !tt.ptr<i32, 3>
    %counter_remote_ptr = "tle.remote_pointers"(%counter_local_ptr, %c0_i32) : (!tt.ptr<i32, 3>, i32) -> !tt.ptr<i32, 7>
    %counter_ptrs = tt.splat %counter_remote_ptr : !tt.ptr<i32, 7> -> tensor<128x!tt.ptr<i32, 7>, #blocked>
    %old = tt.atomic_rmw add, relaxed, cta, %counter_ptrs, %ones, %pred : (tensor<128x!tt.ptr<i32, 7>, #blocked>, tensor<128xi32, #blocked>, tensor<128xi1, #blocked>) -> tensor<128xi32, #blocked>
    tt.return
  }
}

// CHECK: atom.shared::cluster.cta.relaxed.add.u32
