// RUN: spine-triton-opt --tle-to-linalg %s | FileCheck %s

// Test: xtle.extract_tile with static index → tensor.extract_slice
//
// Source tensor: 8x16xf32, tile shape: 4x8xf32
// Grid = [8/4, 16/8] = [2, 2], total 4 tiles
// Index 0 → offset [0, 0]
// Index 1 → offset [0, 8]
// Index 2 → offset [4, 0]
// Index 3 → offset [4, 8]

module {
  func.func @extract_tile_static_idx0(%src: tensor<8x16xf32>) -> tensor<4x8xf32> {
    %c0 = arith.constant 0 : i32
    %result = xtle.extract_tile %src[%c0] {tile_shape = array<i64: 4, 8>} : tensor<8x16xf32>, i32 -> tensor<4x8xf32>
    return %result : tensor<4x8xf32>
  }
  // CHECK-LABEL: func.func @extract_tile_static_idx0
  // CHECK:         tensor.extract_slice %{{.*}}[0, 0] [4, 8] [1, 1] : tensor<8x16xf32> to tensor<4x8xf32>

  func.func @extract_tile_static_idx3(%src: tensor<8x16xf32>) -> tensor<4x8xf32> {
    %c3 = arith.constant 3 : i32
    %result = xtle.extract_tile %src[%c3] {tile_shape = array<i64: 4, 8>} : tensor<8x16xf32>, i32 -> tensor<4x8xf32>
    return %result : tensor<4x8xf32>
  }
  // CHECK-LABEL: func.func @extract_tile_static_idx3
  // CHECK:         tensor.extract_slice %{{.*}}[4, 8] [4, 8] [1, 1] : tensor<8x16xf32> to tensor<4x8xf32>

  // 1D case: source 1024xf32, tile 256xf32, grid = [4]
  // Index 2 → offset [512]
  func.func @extract_tile_1d(%src: tensor<1024xf32>) -> tensor<256xf32> {
    %c2 = arith.constant 2 : i32
    %result = xtle.extract_tile %src[%c2] {tile_shape = array<i64: 256>} : tensor<1024xf32>, i32 -> tensor<256xf32>
    return %result : tensor<256xf32>
  }
  // CHECK-LABEL: func.func @extract_tile_1d
  // CHECK:         tensor.extract_slice %{{.*}}[512] [256] [1] : tensor<1024xf32> to tensor<256xf32>
}
