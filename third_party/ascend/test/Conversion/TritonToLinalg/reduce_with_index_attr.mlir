// RUN: triton-adapter-opt --triton-to-linalg %s | FileCheck %s

module {
  tt.func public @argmax_012(%arg0: !tt.ptr<f32>, %arg1: !tt.ptr<i32>, %arg2: i32) {
    %0 = tt.get_program_id x : i32
    %1 = arith.muli %0, %arg2 : i32
    %2 = tt.make_range {end = 4096 : i32, start = 0 : i32} : tensor<4096xi32>
    %3 = tt.splat %1 : i32 -> tensor<4096xi32>
    %4 = arith.addi %3, %2 : tensor<4096xi32>
    %5 = tt.splat %arg0 : !tt.ptr<f32> -> tensor<4096x!tt.ptr<f32>>
    %6 = tt.addptr %5, %4 : tensor<4096x!tt.ptr<f32>>, tensor<4096xi32>
    %7 = tt.load %6 : tensor<4096x!tt.ptr<f32>>
    // CHECK: %[[REDUCED:.*]]:2 = linalg.reduce ins(%[[INPUT1:.*]], %[[INPUT2:.*]] : tensor<4096xf32>, tensor<4096xi32>) outs(%[[OUTPUT1:.*]], %[[OUTPUT2:.*]] : tensor<f32>, tensor<i32>) dimensions = [0]  {reduce_mode = "max_with_index"}
    %8:2 = "tt.reduce"(%7, %2) <{axis = 0 : i32}> ({
    ^bb0(%arg9: f32, %arg10: i32, %arg11: f32, %arg12: i32):
      %11 = arith.cmpf ogt, %arg9, %arg11 : f32
      %12 = arith.cmpf oeq, %arg9, %arg11 : f32
      %13 = arith.cmpi slt, %arg10, %arg12 : i32
      %14 = arith.andi %12, %13 : i1
      %15 = arith.ori %11, %14 : i1
      %16 = arith.select %15, %arg9, %arg11 : f32
      %17 = arith.select %15, %arg10, %arg12 : i32
      tt.reduce.return %16, %17 : f32, i32
  }) : (tensor<4096xf32>, tensor<4096xi32>) -> (f32, i32)
    %9 = tt.addptr %arg1, %0 : !tt.ptr<i32>, i32
    tt.store %9, %8#1 : !tt.ptr<i32>
    tt.return
  }
}