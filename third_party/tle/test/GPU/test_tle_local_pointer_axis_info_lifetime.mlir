// RUN: triton-opt %s -pass-pipeline='builtin.module(allocate-shared-memory-nv{compute-capability=90 ptx-version=80}, tritongpu-global-scratch-memory-allocation, convert-triton-gpu-to-llvm{compute-capability=90 ptx-version=80})' | FileCheck %s

// A tensor-valued local_ptr must survive until the load/store conversion uses
// its AxisInfo. Lowering it in a separate conversion replaces the SSA value
// by an unanalyzed materialization cast and silently scalarizes the loads.
#blocked = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [2, 16], warpsPerCTA = [8, 1], order = [0, 1]}>
#shared = #ttg.swizzled_shared<{vec = 8, perPhase = 1, maxPhase = 1, order = [0]}>
#smem = #ttg.shared_memory
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 8 : i32, ttg.target = "cuda:90", "ttg.threads-per-warp" = 32 : i32} {
  // CHECK-LABEL: @local_pointer_axis_info_lifetime
  tt.func public @local_pointer_axis_info_lifetime() {
    %smem = ttg.local_alloc : () -> !ttg.memdesc<2048xbf16, #shared, #smem, mutable>
    %row = tt.make_range {end = 16 : i32, start = 0 : i32} : tensor<16xi32, #ttg.slice<{dim = 1, parent = #blocked}>>
    %rows = tt.expand_dims %row {axis = 1 : i32} : tensor<16xi32, #ttg.slice<{dim = 1, parent = #blocked}>> -> tensor<16x1xi32, #blocked>
    %rows_b = tt.broadcast %rows : tensor<16x1xi32, #blocked> -> tensor<16x128xi32, #blocked>
    %col = tt.make_range {end = 128 : i32, start = 0 : i32} : tensor<128xi32, #ttg.slice<{dim = 0, parent = #blocked}>>
    %cols = tt.expand_dims %col {axis = 0 : i32} : tensor<128xi32, #ttg.slice<{dim = 0, parent = #blocked}>> -> tensor<1x128xi32, #blocked>
    %cols_b = tt.broadcast %cols : tensor<1x128xi32, #blocked> -> tensor<16x128xi32, #blocked>
    %stride = arith.constant dense<128> : tensor<16x128xi32, #blocked>
    %base = arith.muli %rows_b, %stride : tensor<16x128xi32, #blocked>
    %offsets = arith.addi %base, %cols_b : tensor<16x128xi32, #blocked>
    %ptrs = "tle.local_pointers"(%smem, %offsets) {tt.contiguity = dense<[1, 8]> : tensor<2xi32>, tt.divisibility = dense<[1, 16]> : tensor<2xi32>} : (!ttg.memdesc<2048xbf16, #shared, #smem, mutable>, tensor<16x128xi32, #blocked>) -> tensor<16x128x!tt.ptr<bf16, 3>, #blocked>
    // CHECK: ld.shared.v4.b32
    // CHECK-NOT: ld.shared.b16
    %values = tt.load %ptrs : tensor<16x128x!tt.ptr<bf16, 3>, #blocked>
    tt.return
  }
}
