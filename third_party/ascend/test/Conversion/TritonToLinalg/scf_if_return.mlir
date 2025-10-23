// RUN: triton-adapter-opt --triton-to-linalg %s | FileCheck %s
module {
  tt.func public @triton_1(%arg0: !tt.ptr<i32> {tt.divisibility = 16 : i32}, %arg1: !tt.ptr<i32> {tt.divisibility = 16 : i32}, %arg2: !tt.ptr<i32> {tt.divisibility = 16 : i32}, %arg3: i32, %arg4: i32) attributes {noinline = false} {
    %c1_i32 = arith.constant 1 : i32
    %c0_i32 = arith.constant 0 : i32
    %c52_i32 = arith.constant 52 : i32
    %cst = arith.constant dense<0> : tensor<1x1xi32>
    %c50_i32 = arith.constant 50 : i32
    %c2_i32 = arith.constant 2 : i32
    %c25_i32 = arith.constant 25 : i32
    %0 = tt.get_program_id x : i32
    %1 = arith.cmpi slt, %0, %c25_i32 : i32
    %2:2 = scf.if %1 -> (i32, i32) {
    // CHECK: %[[RET:.*]]:2 = scf.if %[[COND:.*]] -> (i32, i32) {
      %3 = arith.muli %0, %c2_i32 : i32
      %4 = arith.addi %3, %c2_i32 : i32
      scf.yield %3, %4 : i32, i32
    } else {
      %3 = arith.subi %0, %c25_i32 : i32
      %4 = arith.muli %3, %c2_i32 : i32
      %5 = arith.addi %4, %c50_i32 : i32
      %6 = arith.addi %4, %c52_i32 : i32
      scf.yield %5, %6 : i32, i32
    }
    scf.for %arg5 = %c0_i32 to %c2_i32 step %c1_i32  : i32 {
      %3 = arith.addi %2#0, %arg5 : i32
      %4 = arith.cmpi slt, %3, %2#1 : i32
      %5 = tt.splat %4 : i1 -> tensor<1x1xi1>
      %6 = tt.addptr %arg0, %3 : !tt.ptr<i32>, i32
      %7 = tt.splat %6 : !tt.ptr<i32> -> tensor<1x1x!tt.ptr<i32>>
      %8 = tt.load %7, %5, %cst : tensor<1x1x!tt.ptr<i32>>
      %9 = tt.addptr %arg1, %3 : !tt.ptr<i32>, i32
      %10 = tt.splat %9 : !tt.ptr<i32> -> tensor<1x1x!tt.ptr<i32>>
      %11 = tt.load %10, %5, %cst : tensor<1x1x!tt.ptr<i32>>
      %12 = arith.addi %8, %11 : tensor<1x1xi32>
      %13 = tt.addptr %arg2, %3 : !tt.ptr<i32>, i32
      %14 = tt.splat %13 : !tt.ptr<i32> -> tensor<1x1x!tt.ptr<i32>>
      tt.store %14, %12, %5 : tensor<1x1x!tt.ptr<i32>>
    }
    tt.return
  }
}
