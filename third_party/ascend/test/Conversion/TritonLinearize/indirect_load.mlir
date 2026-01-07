// RUN: triton-adapter-opt %s --triton-linearize --discrete-mask-access-conversion --triton-to-annotation '--triton-to-unstructure=compile-on-910-95=False force-simt-template=False' --triton-to-hivm --triton-to-hfusion --triton-to-llvm --bubble-up-operation '--triton-to-linalg=global-kernel=false named-ops=True enable-nd2nz-on-vector=False compile-on-910-95=False' --split-input-file | FileCheck %s

module {
  tt.func public @test_linearize_indirect_load(%arg0: !tt.ptr<f32>, %arg1: !tt.ptr<f32>, %arg2: !tt.ptr<i64>) {
    %c30_i32 = arith.constant 30 : i32
    %cst = arith.constant dense<5> : tensor<15xi64>
    %c15_i32 = arith.constant 15 : i32
    %0 = tt.get_program_id x : i32
    %1 = tt.make_range {end = 15 : i32, start = 0 : i32} : tensor<15xi32>
    %2 = tt.splat %arg2 : !tt.ptr<i64> -> tensor<15x!tt.ptr<i64>>
    %3 = tt.addptr %2, %1 : tensor<15x!tt.ptr<i64>>, tensor<15xi32>
    %4 = tt.load %3 : tensor<15x!tt.ptr<i64>>
    %5 = arith.muli %0, %c15_i32 : i32
    %6 = tt.splat %5 : i32 -> tensor<15xi32>
    %7 = arith.addi %6, %1 : tensor<15xi32>
    %8 = arith.muli %0, %c30_i32 : i32
    %9 = arith.extsi %8 : i32 to i64
    %10 = tt.splat %9 : i64 -> tensor<15xi64>
    %11 = arith.addi %10, %4 : tensor<15xi64>
    %12 = arith.divsi %11, %cst : tensor<15xi64>
    %13 = arith.remsi %11, %cst : tensor<15xi64>
    %14 = arith.muli %12, %cst : tensor<15xi64>
    %15 = arith.addi %14, %13 : tensor<15xi64>
    %16 = tt.splat %arg1 : !tt.ptr<f32> -> tensor<15x!tt.ptr<f32>>
    %17 = tt.addptr %16, %15 : tensor<15x!tt.ptr<f32>>, tensor<15xi64>
    %18 = tt.load %17 : tensor<15x!tt.ptr<f32>>
    %19 = tt.splat %arg0 : !tt.ptr<f32> -> tensor<15x!tt.ptr<f32>>
    %20 = tt.addptr %19, %7 : tensor<15x!tt.ptr<f32>>, tensor<15xi32>
    tt.store %20, %18 : tensor<15x!tt.ptr<f32>>
    tt.return
  }
}

// CHECK-LABEL: func.func @test_linearize_indirect_load
// CHECK:       {DiscreteMemAccess}
// CHECK:       {ExtractedLoadOrStore}