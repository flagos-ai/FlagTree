// RUN: triton-opt %s -pass-pipeline='builtin.module(allocate-shared-memory-nv{compute-capability=120 ptx-version=88}, tritongpu-global-scratch-memory-allocation, convert-triton-gpu-to-llvm{compute-capability=120 ptx-version=88}, canonicalize, cse, convert-nv-gpu-to-llvm, convert-warp-specialize-to-llvm, canonicalize, cse, symbol-dce, convert-nvvm-to-llvm)' | FileCheck %s

#blocked = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:120", "ttg.threads-per-warp" = 32 : i32} {
  tt.func public @cumsum_to_llvm(%arg0: tensor<128xi32, #blocked>, %out: !tt.ptr<i32>) {
    %exclusive, %total = "tle.exclusive_cumsum"(%arg0) {axis = 0 : i32, reverse = false} : (tensor<128xi32, #blocked>) -> (tensor<128xi32, #blocked>, i32)
    tt.store %out, %total : !tt.ptr<i32>
    tt.return
  }
}

// CHECK: llvm.func @cumsum_to_llvm
// CHECK-NOT: tle.exclusive_cumsum
