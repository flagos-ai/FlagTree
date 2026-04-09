// RUN: triton-opt %s -pass-pipeline='builtin.module(allocate-shared-memory-nv{compute-capability=120 ptx-version=88}, tritongpu-global-scratch-memory-allocation, convert-triton-gpu-to-llvm{compute-capability=120 ptx-version=88}, canonicalize, cse, convert-nv-gpu-to-llvm, convert-warp-specialize-to-llvm, canonicalize, cse, symbol-dce, convert-nvvm-to-llvm)' | FileCheck %s

#blocked = #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [32, 1], warpsPerCTA = [16, 1], order = [1, 0]}>
#shared = #ttg.swizzled_shared<{vec = 4, perPhase = 1, maxPhase = 1, order = [0]}>
#smem = #ttg.shared_memory

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 16 : i32, ttg.target = "cuda:120", "ttg.threads-per-warp" = 32 : i32} {
  tt.func public @local_ptr_v4_load() {
    %c0_i32 = arith.constant 0 : i32
    %smem = ttg.local_alloc {tle.barrier_group = 0 : i64} : () -> !ttg.memdesc<4096xi32, #shared, #smem, mutable>
    %base = "tle.local_pointers"(%smem, %c0_i32) {tle.barrier_group = 0 : i64} : (!ttg.memdesc<4096xi32, #shared, #smem, mutable>, i32) -> !tt.ptr<i32, 3>
    %basev = tt.splat %base : !tt.ptr<i32, 3> -> tensor<512x4x!tt.ptr<i32, 3>, #blocked>

    %row = tt.make_range {end = 512 : i32, start = 0 : i32} : tensor<512xi32, #ttg.slice<{dim = 1, parent = #blocked}>>
    %row2d = tt.expand_dims %row {axis = 1 : i32} : tensor<512xi32, #ttg.slice<{dim = 1, parent = #blocked}>> -> tensor<512x1xi32, #blocked>
    %rowb = tt.broadcast %row2d : tensor<512x1xi32, #blocked> -> tensor<512x4xi32, #blocked>
    %col = tt.make_range {end = 4 : i32, start = 0 : i32} : tensor<4xi32, #ttg.slice<{dim = 0, parent = #blocked}>>
    %col2d = tt.expand_dims %col {axis = 0 : i32} : tensor<4xi32, #ttg.slice<{dim = 0, parent = #blocked}>> -> tensor<1x4xi32, #blocked>
    %colb = tt.broadcast %col2d : tensor<1x4xi32, #blocked> -> tensor<512x4xi32, #blocked>
    %offs = arith.addi %rowb, %colb : tensor<512x4xi32, #blocked>

    %ptrs = tt.addptr %basev, %offs : tensor<512x4x!tt.ptr<i32, 3>, #blocked>, tensor<512x4xi32, #blocked>
    %vals = tt.load %ptrs : tensor<512x4x!tt.ptr<i32, 3>, #blocked>
    tt.return
  }
}

// CHECK: ld.shared.v4.b32
