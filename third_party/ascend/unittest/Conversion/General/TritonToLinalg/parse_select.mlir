// RUN: triton-opt --triton-to-linalg --split-input-file %s -verify-each 2>&1 | FileCheck %s --check-prefix=NOERR
// NOERR-NOT:   parseSelect currently supports all-ones shape unless cond=i1 with dense constants
// CHECK-LABEL: func.func public @triton_for_if_load

module {
  tt.func public @triton_for_if_load(%arg0: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %arg1: !tt.ptr<f32> {tt.divisibility = 16 : i32}) attributes {noinline = false} {
    %cst = arith.constant dense<0> : tensor<16xi32>
    %c0_i32 = arith.constant 0 : i32
    %c16_i32 = arith.constant 16 : i32
    %cst_0 = arith.constant dense<1> : tensor<16xi32>
    %cst_1 = arith.constant dense<32> : tensor<16xi32>
    %cst_2 = arith.constant dense<0.000000e+00> : tensor<16xf32>
    %c2_i32 = arith.constant 2 : i32
    %c1_i32 = arith.constant 1 : i32
    %0 = tt.make_range {end = 16 : i32, start = 0 : i32} : tensor<16xi32>
    %1 = tt.get_program_id x : i32
    %2 = arith.cmpi ne, %1, %c0_i32 : i32
    %3 = tt.splat %arg0 : !tt.ptr<f32> -> tensor<16x!tt.ptr<f32>>
    %4 = tt.splat %arg1 : !tt.ptr<f32> -> tensor<16x!tt.ptr<f32>>
    %5:2 = scf.for %arg2 = %c0_i32 to %c2_i32 step %c1_i32 iter_args(%arg3 = %0, %arg4 = %0) -> (tensor<16xi32>, tensor<16xi32>) : i32 {
      %6 = arith.muli %arg2, %c16_i32 : i32
      %7 = tt.splat %6 : i32 -> tensor<16xi32>
      %8 = arith.addi %arg3, %7 : tensor<16xi32>
      %9 = arith.addi %arg4, %7 : tensor<16xi32>
      %10 = arith.select %2, %cst_0, %cst : tensor<16xi32>
      %11 = arith.addi %8, %10 : tensor<16xi32>
      %12 = tt.addptr %3, %11 : tensor<16x!tt.ptr<f32>>, tensor<16xi32>
      %13 = arith.cmpi slt, %11, %cst_1 : tensor<16xi32>
      %14 = tt.load %12 : tensor<16x!tt.ptr<f32>>
      %15 = arith.select %13, %14, %cst_2 : tensor<16xi1>, tensor<16xf32>
      %16 = tt.addptr %4, %9 : tensor<16x!tt.ptr<f32>>, tensor<16xi32>
      tt.store %16, %15 : tensor<16x!tt.ptr<f32>>
      scf.yield %11, %9 : tensor<16xi32>, tensor<16xi32>
    }
    tt.return
  }
}
