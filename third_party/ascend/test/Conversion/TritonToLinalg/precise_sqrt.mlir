// RUN: triton-adapter-opt %s --triton-to-linalg | FileCheck %s

module {
  tt.func public @sqrtrn_kernel(%arg0: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %arg1: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %arg2: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %arg3: i32 {tt.divisibility = 16 : i32}) attributes {noinline = false} {
    %c512_i32 = arith.constant 512 : i32
    %0 = tt.get_program_id x : i32
    %1 = arith.muli %0, %c512_i32 : i32
    %2 = tt.make_range {end = 512 : i32, start = 0 : i32} : tensor<512xi32>
    %3 = tt.splat %1 : i32 -> tensor<512xi32>
    %4 = arith.addi %3, %2 : tensor<512xi32>
    %5 = tt.splat %arg3 : i32 -> tensor<512xi32>
    %6 = arith.cmpi slt, %4, %5 : tensor<512xi32>
    %7 = tt.splat %arg0 : !tt.ptr<f32> -> tensor<512x!tt.ptr<f32>>
    %8 = tt.addptr %7, %4 : tensor<512x!tt.ptr<f32>>, tensor<512xi32>
    %9 = tt.load %8, %6 : tensor<512x!tt.ptr<f32>>
    %10 = tt.splat %arg1 : !tt.ptr<f32> -> tensor<512x!tt.ptr<f32>>
    %11 = tt.addptr %10, %4 : tensor<512x!tt.ptr<f32>>, tensor<512xi32>
    %12 = tt.load %11, %6 : tensor<512x!tt.ptr<f32>>
    %13 = tt.precise_sqrt %12 : tensor<512xf32>
    %14 = arith.addf %9, %13 : tensor<512xf32>
    %15 = tt.splat %arg2 : !tt.ptr<f32> -> tensor<512x!tt.ptr<f32>>
    %16 = tt.addptr %15, %4 : tensor<512x!tt.ptr<f32>>, tensor<512xi32>
    tt.store %16, %14, %6 : tensor<512x!tt.ptr<f32>>
    tt.return
  }
}


//CHECK: %[[OUTPUT:.*]] = math.sqrt %[[INPUT:.*]] : f32
