// RUN: triton-opt %s -allocate-shared-memory -convert-triton-amdgpu-to-llvm=arch=gfx1100 | FileCheck %s

// First AMD/RDNA TLE lowering test (Phase 0, FlagMega-on-Radeon).
// The tile-core `tle.local_pointers` op plus a load off the LDS workspace must
// lower to LLVM/ROCDL on gfx1100: the workspace pointer becomes an LDS pointer
// (LLVM address space 3), the body uses ROCDL, and no `tle.` op is left over.

#blocked = #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [32, 1], warpsPerCTA = [16, 1], order = [1, 0]}>
#shared = #ttg.swizzled_shared<{vec = 4, perPhase = 1, maxPhase = 1, order = [0]}>
#smem = #ttg.shared_memory

module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 16 : i32, ttg.target = "hip:gfx1100", "ttg.threads-per-warp" = 32 : i32} {
  // CHECK-LABEL: llvm.func @local_ptr_v4_load
  tt.func public @local_ptr_v4_load() {
    %c0_i32 = arith.constant 0 : i32
    %c4 = arith.constant dense<4> : tensor<512x4xi32, #blocked>
    %smem = ttg.local_alloc {tle.barrier_group = 0 : i64} : () -> !ttg.memdesc<4096xi32, #shared, #smem, mutable>
    %base = "tle.local_pointers"(%smem, %c0_i32) {tle.barrier_group = 0 : i64} : (!ttg.memdesc<4096xi32, #shared, #smem, mutable>, i32) -> !tt.ptr<i32, 3>
    %basev = tt.splat %base : !tt.ptr<i32, 3> -> tensor<512x4x!tt.ptr<i32, 3>, #blocked>

    %row = tt.make_range {end = 512 : i32, start = 0 : i32} : tensor<512xi32, #ttg.slice<{dim = 1, parent = #blocked}>>
    %row2d = tt.expand_dims %row {axis = 1 : i32} : tensor<512xi32, #ttg.slice<{dim = 1, parent = #blocked}>> -> tensor<512x1xi32, #blocked>
    %rowb = tt.broadcast %row2d : tensor<512x1xi32, #blocked> -> tensor<512x4xi32, #blocked>
    %col = tt.make_range {end = 4 : i32, start = 0 : i32} : tensor<4xi32, #ttg.slice<{dim = 0, parent = #blocked}>>
    %col2d = tt.expand_dims %col {axis = 0 : i32} : tensor<4xi32, #ttg.slice<{dim = 0, parent = #blocked}>> -> tensor<1x4xi32, #blocked>
    %colb = tt.broadcast %col2d : tensor<1x4xi32, #blocked> -> tensor<512x4xi32, #blocked>
    %row_scaled = arith.muli %rowb, %c4 : tensor<512x4xi32, #blocked>
    %offs = arith.addi %row_scaled, %colb : tensor<512x4xi32, #blocked>

    %ptrs = tt.addptr %basev, %offs : tensor<512x4x!tt.ptr<i32, 3>, #blocked>, tensor<512x4xi32, #blocked>
    // CHECK: rocdl.workitem.id
    // CHECK: llvm.load %{{.*}} : !llvm.ptr<3>
    // CHECK-NOT: "tle.
    %vals = tt.load %ptrs : tensor<512x4x!tt.ptr<i32, 3>, #blocked>
    tt.return
  }
}
