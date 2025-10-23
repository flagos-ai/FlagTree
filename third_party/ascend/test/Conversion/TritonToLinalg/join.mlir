// RUN: triton-adapter-opt --triton-to-linalg %s | FileCheck %s
module {
  tt.func public @fn_triton_join(%arg0: !tt.ptr<f32>, %arg1: !tt.ptr<f32>, %arg2: !tt.ptr<f32>, %arg3: !tt.ptr<f32>, %arg4: !tt.ptr<f32>) attributes {noinline = false} {
    %cst = arith.constant dense<2> : tensor<1x8x1xi32>
    %cst_0 = arith.constant dense<2> : tensor<8x1x1xi32>
    %cst_1 = arith.constant dense<8> : tensor<8x1x1xi32>
    %cst_2 = arith.constant dense<8> : tensor<8x1xi32>
    %0 = tt.make_range {end = 8 : i32, start = 0 : i32} : tensor<8xi32>
    %1 = tt.expand_dims %0 {axis = 1 : i32} : tensor<8xi32> -> tensor<8x1xi32>
    %2 = arith.muli %1, %cst_2 : tensor<8x1xi32>
    %3 = tt.expand_dims %0 {axis = 0 : i32} : tensor<8xi32> -> tensor<1x8xi32>
    %4 = tt.broadcast %2 : tensor<8x1xi32> -> tensor<8x8xi32>
    %5 = tt.broadcast %3 : tensor<1x8xi32> -> tensor<8x8xi32>
    %6 = arith.addi %4, %5 : tensor<8x8xi32>
    %7 = tt.splat %arg1 : !tt.ptr<f32> -> tensor<8x8x!tt.ptr<f32>>
    %8 = tt.addptr %7, %6 : tensor<8x8x!tt.ptr<f32>>, tensor<8x8xi32>
    %9 = tt.load %8 : tensor<8x8x!tt.ptr<f32>>
    %10 = tt.splat %arg2 : !tt.ptr<f32> -> tensor<8x8x!tt.ptr<f32>>
    %11 = tt.addptr %10, %6 : tensor<8x8x!tt.ptr<f32>>, tensor<8x8xi32>
    %12 = tt.load %11 : tensor<8x8x!tt.ptr<f32>>
    %13 = tt.join %9, %12 : tensor<8x8xf32> -> tensor<8x8x2xf32>
    %14 = tt.expand_dims %1 {axis = 2 : i32} : tensor<8x1xi32> -> tensor<8x1x1xi32>
    %15 = arith.muli %14, %cst_1 : tensor<8x1x1xi32>
    %16 = arith.muli %15, %cst_0 : tensor<8x1x1xi32>
    %17 = tt.expand_dims %3 {axis = 2 : i32} : tensor<1x8xi32> -> tensor<1x8x1xi32>
    %18 = arith.muli %17, %cst : tensor<1x8x1xi32>
    %19 = tt.broadcast %16 : tensor<8x1x1xi32> -> tensor<8x8x1xi32>
    %20 = tt.broadcast %18 : tensor<1x8x1xi32> -> tensor<8x8x1xi32>
    %21 = arith.addi %19, %20 : tensor<8x8x1xi32>
    %22 = tt.make_range {end = 2 : i32, start = 0 : i32} : tensor<2xi32>
    %23 = tt.expand_dims %22 {axis = 0 : i32} : tensor<2xi32> -> tensor<1x2xi32>
    %24 = tt.expand_dims %23 {axis = 1 : i32} : tensor<1x2xi32> -> tensor<1x1x2xi32>
    %25 = tt.broadcast %21 : tensor<8x8x1xi32> -> tensor<8x8x2xi32>
    %26 = tt.broadcast %24 : tensor<1x1x2xi32> -> tensor<8x8x2xi32>
    %27 = arith.addi %25, %26 : tensor<8x8x2xi32>
    %28 = tt.splat %arg0 : !tt.ptr<f32> -> tensor<8x8x2x!tt.ptr<f32>>
    %29 = tt.addptr %28, %27 : tensor<8x8x2x!tt.ptr<f32>>, tensor<8x8x2xi32>
    tt.store %29, %13 : tensor<8x8x2x!tt.ptr<f32>>
    tt.return
  }
}
//CHECK-LABEL: @fn_triton_join
//CHECK-NOT: tt.join
//CHECK: %[[IN0:.*]] = bufferization.to_tensor %[[ADDR0:.*]] restrict writable : memref<8x8xf32>
//CHECK: %[[IN1:.*]] = bufferization.to_tensor %[[ADDR1:.*]] restrict writable : memref<8x8xf32>
//CHECK: %[[ZERO:.*]] = tensor.empty() : tensor<8x8x2xf32>
//CHECK: %[[INSERT0:.*]] = tensor.insert_slice %[[IN0]] into %[[ZERO]][0, 0, 0] [8, 8, 1] [1, 1, 2] : tensor<8x8xf32> into tensor<8x8x2xf32>
//CHECK: %[[INSERT1:.*]] = tensor.insert_slice %[[IN1]] into %[[INSERT0]][0, 0, 1] [8, 8, 1] [1, 1, 2] : tensor<8x8xf32> into tensor<8x8x2xf32>