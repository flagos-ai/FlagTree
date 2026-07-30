// RUN: spine-triton-opt --tle-to-linalg %s | FileCheck %s

// Test: xtle.insert_tile with static index → tensor.insert_slice
//
// Source tensor: 8x16xf32, tile shape: 4x8xf32
// Grid = [2, 2], total 4 tiles

module {
  func.func @insert_tile_static_idx0(%src: tensor<8x16xf32>, %tile: tensor<4x8xf32>) -> tensor<8x16xf32> {
    %c0 = arith.constant 0 : i32
    %result = xtle.insert_tile %src[%c0] = %tile : tensor<8x16xf32>, i32, tensor<4x8xf32> -> tensor<8x16xf32>
    return %result : tensor<8x16xf32>
  }
  // CHECK-LABEL: func.func @insert_tile_static_idx0
  // CHECK:         tensor.insert_slice %{{.*}} into %{{.*}}[0, 0] [4, 8] [1, 1] : tensor<4x8xf32> into tensor<8x16xf32>

  func.func @insert_tile_static_idx2(%src: tensor<8x16xf32>, %tile: tensor<4x8xf32>) -> tensor<8x16xf32> {
    %c2 = arith.constant 2 : i32
    %result = xtle.insert_tile %src[%c2] = %tile : tensor<8x16xf32>, i32, tensor<4x8xf32> -> tensor<8x16xf32>
    return %result : tensor<8x16xf32>
  }
  // CHECK-LABEL: func.func @insert_tile_static_idx2
  // CHECK:         tensor.insert_slice %{{.*}} into %{{.*}}[4, 0] [4, 8] [1, 1] : tensor<4x8xf32> into tensor<8x16xf32>

  // 1D case
  func.func @insert_tile_1d(%src: tensor<1024xf32>, %tile: tensor<256xf32>) -> tensor<1024xf32> {
    %c3 = arith.constant 3 : i32
    %result = xtle.insert_tile %src[%c3] = %tile : tensor<1024xf32>, i32, tensor<256xf32> -> tensor<1024xf32>
    return %result : tensor<1024xf32>
  }
  // CHECK-LABEL: func.func @insert_tile_1d
  // CHECK:         tensor.insert_slice %{{.*}} into %{{.*}}[768] [256] [1] : tensor<256xf32> into tensor<1024xf32>
}
