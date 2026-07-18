// RUN: triton-opt %s -tritongpu-pipeline | FileCheck %s

#blocked = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>

module attributes {"ttg.target" = "cuda:90", "ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 32 : i32} {
  // CHECK-LABEL: tt.func @if_without_else_in_pipelined_loop
  tt.func @if_without_else_in_pipelined_loop(%arg0: !tt.ptr<i32> {tt.divisibility = 16 : i32}, %arg1: !tt.ptr<i32> {tt.divisibility = 16 : i32}, %arg2: i1) attributes {noinline = false} {
    %c1_i32 = arith.constant 1 : i32
    %c0_i32 = arith.constant 0 : i32
    %c17_i32 = arith.constant 17 : i32
    // CHECK: scf.for
    scf.for %iv = %c0_i32 to %c17_i32 step %c1_i32  : i32 {
      %0 = tt.addptr %arg0, %iv : !tt.ptr<i32>, i32
      %1 = tt.splat %0 : !tt.ptr<i32> -> tensor<1x!tt.ptr<i32>, #blocked>
      %2 = tt.load %1 : tensor<1x!tt.ptr<i32>, #blocked>
      // CHECK: scf.if
      scf.if %arg2 {
        %3 = tt.splat %arg1 : !tt.ptr<i32> -> tensor<1x!tt.ptr<i32>, #blocked>
        %4 = tt.addptr %3, %2 : tensor<1x!tt.ptr<i32>, #blocked>, tensor<1xi32, #blocked>
        %5 = arith.addi %iv, %c1_i32 : i32
        %6 = tt.splat %5 : i32 -> tensor<1xi32, #blocked>
        tt.store %4, %6 : tensor<1x!tt.ptr<i32>, #blocked>
      }
    } {tt.num_stages = 1 : i32}
    tt.return
  }
}
