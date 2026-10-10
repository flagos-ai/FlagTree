// Copyright 2026- Xcoresigma Technology Co., Ltd

// RUN: triton-opt %s -reduce-sum-strength -triton-to-linalg-incubated='global-kernel=false' | FileCheck %s

module {
  tt.func @kernel(%afloat : !tt.ptr<f32>, %res : !tt.ptr<f32>) -> () {
    %0 = tt.make_range {end = 512 : i32, start = 0 : i32} : tensor<512xi32>
    %c256 = arith.constant 256 : i32
    %ct256 = tt.splat %c256 : i32 -> tensor<512xi32>
    %ws = arith.muli %ct256, %0 : tensor<512xi32>
    %1 = tt.expand_dims %ws {axis = 1 : i32} : tensor<512xi32> -> tensor<512x1xi32>
    %moff = tt.broadcast %1 : tensor<512x1xi32> -> tensor<512x256xi32>
    %3 = tt.make_range {end = 256 : i32, start = 0 : i32} : tensor<256xi32>
    %4 = tt.expand_dims %3 {axis = 0 : i32} : tensor<256xi32> -> tensor<1x256xi32>
    %koff = tt.broadcast %4 : tensor<1x256xi32> -> tensor<512x256xi32>
    %mkoff = arith.addi %moff, %koff : tensor<512x256xi32>
    %8 = tt.splat %afloat : !tt.ptr<f32> -> tensor<512x256x!tt.ptr<f32>>
    %9 = tt.addptr %8, %mkoff : tensor<512x256x!tt.ptr<f32>>, tensor<512x256xi32>
    %18 = tt.splat %res : !tt.ptr<f32> -> tensor<256x!tt.ptr<f32>>
    %19 = tt.addptr %18, %3 : tensor<256x!tt.ptr<f32>>, tensor<256xi32>
    %afm = tt.load %9 : tensor<512x256x!tt.ptr<f32>>
    %5 = "tt.reduce"(%afm) ({
    ^bb0(%arg5: f32, %arg6: f32):
      %21 = arith.addf %arg5, %arg6 : f32
      tt.reduce.return %21 : f32
    }) {axis = 0 : i32} : (tensor<512x256xf32>) -> tensor<256xf32>
    tt.store %19, %5 : tensor<256x!tt.ptr<f32>>
    tt.return
  }
}

// CHECK-LABEL: func.func @kernel
// 512 -> 256 -> 128 along the reduced axis, final linalg.reduce on 128x256.
// CHECK: %[[LO:.*]] = tensor.extract_slice {{.*}}[0, 0] [256, 256] [1, 1] : tensor<512x256xf32> to tensor<256x256xf32>
// CHECK: %[[HI:.*]] = tensor.extract_slice {{.*}}[256, 0] [256, 256] [1, 1] : tensor<512x256xf32> to tensor<256x256xf32>
// CHECK: linalg.generic {{.*}} ins(%[[LO]], %[[HI]] : tensor<256x256xf32>, tensor<256x256xf32>)
// CHECK: tensor.extract_slice {{.*}}[0, 0] [128, 256] [1, 1] : tensor<256x256xf32> to tensor<128x256xf32>
// CHECK: tensor.extract_slice {{.*}}[128, 0] [128, 256] [1, 1] : tensor<256x256xf32> to tensor<128x256xf32>
// CHECK: linalg.generic {{.*}} ins({{.*}} : tensor<128x256xf32>, tensor<128x256xf32>)
// CHECK: linalg.reduce ins({{.*}} : tensor<128x256xf32>) {{.*}} dimensions = [0]
// CHECK: bufferization.materialize_in_destination
