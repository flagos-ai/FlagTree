// RUN: triton-opt %s -pass-pipeline='builtin.module(allocate-shared-memory-nv{compute-capability=120 ptx-version=88})' | FileCheck %s

#blocked = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [16], order = [0]}>
#shared = #ttg.nvmma_shared<{swizzlingByteWidth = 0, transposed = true, elementBitWidth = 32, rank = 1}>
#smem = #ttg.shared_memory

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 16 : i32, ttg.target = "cuda:120", "ttg.threads-per-warp" = 32 : i32} {
  // CHECK-LABEL: tt.func @cumsum_scratch_no_alias
  // CHECK: ttg.local_alloc {allocation.offset = 0 : i32
  // CHECK: tle.exclusive_cumsum
  // CHECK-SAME: allocation.offset = 4096 : i32
  tt.func @cumsum_scratch_no_alias(%out: !tt.ptr<i32>) {
    %c0_i32 = arith.constant 0 : i32
    %cst_1 = arith.constant dense<1> : tensor<512xi32, #blocked>
    %offs = tt.make_range {end = 512 : i32, start = 0 : i32} : tensor<512xi32, #blocked>
    %alloc = ttg.local_alloc : () -> !ttg.memdesc<1024xi32, #shared, #smem, mutable>
    %base = "tle.local_pointers"(%alloc, %c0_i32) {tle.barrier_group = 0 : i64} : (!ttg.memdesc<1024xi32, #shared, #smem, mutable>, i32) -> !tt.ptr<i32, 3>
    %base_tensor = tt.splat %base : !tt.ptr<i32, 3> -> tensor<512x!tt.ptr<i32, 3>, #blocked>
    %ptrs = tt.addptr %base_tensor, %offs : tensor<512x!tt.ptr<i32, 3>, #blocked>, tensor<512xi32, #blocked>
    tt.store %ptrs, %cst_1 : tensor<512x!tt.ptr<i32, 3>, #blocked>
    %x = tt.load %ptrs : tensor<512x!tt.ptr<i32, 3>, #blocked>
    %exclusive, %total = "tle.exclusive_cumsum"(%x) {axis = 0 : i32, reverse = false} : (tensor<512xi32, #blocked>) -> (tensor<512xi32, #blocked>, i32)
    tt.store %ptrs, %exclusive : tensor<512x!tt.ptr<i32, 3>, #blocked>
    tt.return
  }
}
