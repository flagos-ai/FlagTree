// RUN: triton-opt %s -split-input-file -convert-triton-to-tritongpu="target=cuda:90 num-warps=8" | FileCheck %s

// Shared layout-polymorphic sources must not join independent explicit layouts.

// CHECK-LABEL: tt.func public @_independent_layout_roots_with_cse_constants
// CHECK: tt.store
// CHECK: tt.store
// CHECK: tt.return
#blocked = #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [4, 8], warpsPerCTA = [8, 1], order = [1, 0]}>
#blocked1 = #ttg.blocked<{sizePerThread = [1, 2], threadsPerWarp = [2, 16], warpsPerCTA = [8, 1], order = [1, 0]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 8 : i32, "ttg.threads-per-warp" = 32 : i32} {
  tt.func public @_independent_layout_roots_with_cse_constants(%arg0: !tt.ptr<i32> {tt.divisibility = 16 : i32}, %arg1: !tt.ptr<i32> {tt.divisibility = 16 : i32}) attributes {noinline = false} {
    %cst = arith.constant dense<1> : tensor<1x32xi32>
    %cst_0 = arith.constant dense<32> : tensor<1x32xi32>
    %0 = tt.make_range {end = 32 : i32, start = 0 : i32} : tensor<32xi32>
    %1 = tt.expand_dims %0 {axis = 0 : i32} : tensor<32xi32> -> tensor<1x32xi32>
    %2 = tle.gpu.set_layout %1 {target_encoding = #blocked} : tensor<1x32xi32> -> tensor<1x32xi32>
    %3 = tle.gpu.set_layout %1 {target_encoding = #blocked1} : tensor<1x32xi32> -> tensor<1x32xi32>
    %4 = arith.addi %2, %cst_0 : tensor<1x32xi32>
    %5 = arith.addi %3, %cst_0 : tensor<1x32xi32>
    %6 = tt.splat %arg0 : !tt.ptr<i32> -> tensor<1x32x!tt.ptr<i32>>
    %7 = tt.addptr %6, %4 : tensor<1x32x!tt.ptr<i32>>, tensor<1x32xi32>
    tt.store %7, %4 : tensor<1x32x!tt.ptr<i32>>
    %8 = tt.splat %arg1 : !tt.ptr<i32> -> tensor<1x32x!tt.ptr<i32>>
    %9 = tt.addptr %8, %5 : tensor<1x32x!tt.ptr<i32>>, tensor<1x32xi32>
    %10 = arith.addi %5, %cst : tensor<1x32xi32>
    tt.store %9, %10 : tensor<1x32x!tt.ptr<i32>>
    tt.return
  }
}

// -----

// CHECK-LABEL: tt.func public @_independent_layout_roots_with_shared_scalar_source
// CHECK: tt.store
// CHECK: tt.store
// CHECK: tt.return
#blocked = #ttg.blocked<{sizePerThread = [1, 4], threadsPerWarp = [4, 8], warpsPerCTA = [8, 1], order = [1, 0]}>
#blocked1 = #ttg.blocked<{sizePerThread = [1, 2], threadsPerWarp = [2, 16], warpsPerCTA = [8, 1], order = [1, 0]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 8 : i32, "ttg.threads-per-warp" = 32 : i32} {
  tt.func public @_independent_layout_roots_with_shared_scalar_source(%arg0: !tt.ptr<i32> {tt.divisibility = 16 : i32}) attributes {noinline = false} {
    %cst = arith.constant dense<32> : tensor<1x32xi32>
    %0 = tt.make_range {end = 32 : i32, start = 0 : i32} : tensor<32xi32>
    %1 = tt.expand_dims %0 {axis = 0 : i32} : tensor<32xi32> -> tensor<1x32xi32>
    %2 = tle.gpu.set_layout %1 {target_encoding = #blocked} : tensor<1x32xi32> -> tensor<1x32xi32>
    %3 = tle.gpu.set_layout %1 {target_encoding = #blocked1} : tensor<1x32xi32> -> tensor<1x32xi32>
    %4 = tt.splat %arg0 : !tt.ptr<i32> -> tensor<1x32x!tt.ptr<i32>>
    %5 = tt.addptr %4, %2 : tensor<1x32x!tt.ptr<i32>>, tensor<1x32xi32>
    tt.store %5, %2 : tensor<1x32x!tt.ptr<i32>>
    %6 = tt.addptr %4, %3 : tensor<1x32x!tt.ptr<i32>>, tensor<1x32xi32>
    %7 = tt.addptr %6, %cst : tensor<1x32x!tt.ptr<i32>>, tensor<1x32xi32>
    %8 = arith.addi %3, %cst : tensor<1x32xi32>
    tt.store %7, %8 : tensor<1x32x!tt.ptr<i32>>
    tt.return
  }
}
