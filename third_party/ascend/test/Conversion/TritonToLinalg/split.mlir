// RUN: triton-adapter-opt --triton-to-linalg %s | FileCheck %s
tt.func public @fn_npu_split(%arg0: !tt.ptr<f32>, %arg1: !tt.ptr<f32>, %arg2: !tt.ptr<f32>, %arg3: !tt.ptr<f32>, %arg4: !tt.ptr<f32>) attributes {noinline = false} {
    %cst = arith.constant dense<256> : tensor<16x1xi32>
    %cst_0 = arith.constant dense<2> : tensor<1x256x1xi32>
    %cst_1 = arith.constant dense<2> : tensor<16x1x1xi32>
    %cst_2 = arith.constant dense<256> : tensor<16x1x1xi32>
    %0 = tt.make_range {end = 16 : i32, start = 0 : i32} : tensor<16xi32>
    %1 = tt.make_range {end = 256 : i32, start = 0 : i32} : tensor<256xi32>
    %2 = tt.make_range {end = 2 : i32, start = 0 : i32} : tensor<2xi32>
    %3 = tt.expand_dims %0 {axis = 1 : i32} : tensor<16xi32> -> tensor<16x1xi32>
    %4 = tt.expand_dims %3 {axis = 2 : i32} : tensor<16x1xi32> -> tensor<16x1x1xi32>
    %5 = arith.muli %4, %cst_2 : tensor<16x1x1xi32>
    %6 = arith.muli %5, %cst_1 : tensor<16x1x1xi32>
    %7 = tt.expand_dims %1 {axis = 0 : i32} : tensor<256xi32> -> tensor<1x256xi32>
    %8 = tt.expand_dims %7 {axis = 2 : i32} : tensor<1x256xi32> -> tensor<1x256x1xi32>
    %9 = arith.muli %8, %cst_0 : tensor<1x256x1xi32>
    %10 = tt.broadcast %6 : tensor<16x1x1xi32> -> tensor<16x256x1xi32>
    %11 = tt.broadcast %9 : tensor<1x256x1xi32> -> tensor<16x256x1xi32>
    %12 = arith.addi %10, %11 : tensor<16x256x1xi32>
    %13 = tt.expand_dims %2 {axis = 0 : i32} : tensor<2xi32> -> tensor<1x2xi32>
    %14 = tt.expand_dims %13 {axis = 1 : i32} : tensor<1x2xi32> -> tensor<1x1x2xi32>
    %15 = tt.broadcast %12 : tensor<16x256x1xi32> -> tensor<16x256x2xi32>
    %16 = tt.broadcast %14 : tensor<1x1x2xi32> -> tensor<16x256x2xi32>
    %17 = arith.addi %15, %16 : tensor<16x256x2xi32>
    %18 = tt.splat %arg1 : !tt.ptr<f32> -> tensor<16x256x2x!tt.ptr<f32>>
    %19 = tt.addptr %18, %17 : tensor<16x256x2x!tt.ptr<f32>>, tensor<16x256x2xi32>
    %20 = tt.load %19 : tensor<16x256x2x!tt.ptr<f32>>
    %outLHS, %outRHS = tt.split %20 : tensor<16x256x2xf32> -> tensor<16x256xf32>
    %21 = arith.muli %3, %cst : tensor<16x1xi32>
    %22 = tt.broadcast %21 : tensor<16x1xi32> -> tensor<16x256xi32>
    %23 = tt.broadcast %7 : tensor<1x256xi32> -> tensor<16x256xi32>
    %24 = arith.addi %22, %23 : tensor<16x256xi32>
    %25 = tt.splat %arg0 : !tt.ptr<f32> -> tensor<16x256x!tt.ptr<f32>>
    %26 = tt.addptr %25, %24 : tensor<16x256x!tt.ptr<f32>>, tensor<16x256xi32>
    tt.store %26, %outLHS : tensor<16x256x!tt.ptr<f32>>
    %27 = tt.splat %arg4 : !tt.ptr<f32> -> tensor<16x256x!tt.ptr<f32>>
    %28 = tt.addptr %27, %24 : tensor<16x256x!tt.ptr<f32>>, tensor<16x256xi32>
    tt.store %28, %outRHS : tensor<16x256x!tt.ptr<f32>>
    tt.return
}

//CHECK-LABEL: @fn_npu_split
//CHECK-NOT: tt.split
//CHECK: %[[VAL0:.*]] = bufferization.to_tensor %[[ADDR:.*]] restrict writable : memref<16x256x2xf32>
//CHECK: %[[EXT0:.*]] = tensor.extract_slice %[[VAL0]][0, 0, 0] [16, 256, 1] [1, 1, 2] : tensor<16x256x2xf32> to tensor<16x256xf32>
//CHECK: %[[EXT1:.*]] = tensor.extract_slice %[[VAL0]][0, 0, 1] [16, 256, 1] [1, 1, 2] : tensor<16x256x2xf32> to tensor<16x256xf32>
