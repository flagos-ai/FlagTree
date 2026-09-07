// RUN: triton-opt %s -pass-pipeline='builtin.module(allocate-shared-memory-nv{compute-capability=90 ptx-version=80}, tritongpu-global-scratch-memory-allocation, convert-triton-gpu-to-llvm{compute-capability=90 ptx-version=80})' | FileCheck %s

#blocked = #ttg.blocked<{sizePerThread = [4], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>
#shared = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 2, order = [1, 0]}>
#smem = #ttg.shared_memory
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:90", "ttg.threads-per-warp" = 32 : i32} {
  // Row 1 flips the low physical column bit. Logical columns [0,1,2,3]
  // address physical columns [1,0,3,2], not a contiguous aligned vector.
  // CHECK-LABEL: @swizzled_local_pointer
  tt.func public @swizzled_local_pointer() {
    %shared = ttg.local_alloc : () -> !ttg.memdesc<2x64xf32, #shared, #smem, mutable>
    %row = arith.constant dense<1> : tensor<64xi32, #blocked>
    %col = tt.make_range {start = 0 : i32, end = 64 : i32} : tensor<64xi32, #blocked>
    %ptr = "tle.local_pointers"(%shared, %row, %col) : (!ttg.memdesc<2x64xf32, #shared, #smem, mutable>, tensor<64xi32, #blocked>, tensor<64xi32, #blocked>) -> tensor<64x!tt.ptr<f32, 3>, #blocked>
    // CHECK-NOT: ld.shared.v4.b32
    // CHECK: ld.shared.b32
    %value = tt.load %ptr : tensor<64x!tt.ptr<f32, 3>, #blocked>
    tt.return
  }
}
