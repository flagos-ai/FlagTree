// RUN: spine-triton-opt --tle-to-linalg %s | FileCheck %s

// Test: xtle.extract_tile / xtle.insert_tile with dynamic (runtime) index
// The lowering should emit arith ops to delinearize the index.

module {
  // Dynamic extract: 8x16xf32, tile 4x8xf32, grid [2,2]
  func.func @extract_tile_dynamic(%src: tensor<8x16xf32>, %idx: i32) -> tensor<4x8xf32> {
    %result = xtle.extract_tile %src[%idx] {tile_shape = array<i64: 4, 8>} : tensor<8x16xf32>, i32 -> tensor<4x8xf32>
    return %result : tensor<4x8xf32>
  }
  // CHECK-LABEL: func.func @extract_tile_dynamic
  // CHECK-DAG:     arith.index_cast
  // CHECK-DAG:     arith.divui
  // CHECK-DAG:     arith.remui
  // CHECK-DAG:     arith.muli
  // CHECK:         tensor.extract_slice

  // Dynamic insert: 8x16xf32, tile 4x8xf32
  func.func @insert_tile_dynamic(%src: tensor<8x16xf32>, %tile: tensor<4x8xf32>, %idx: i32) -> tensor<8x16xf32> {
    %result = xtle.insert_tile %src[%idx] = %tile : tensor<8x16xf32>, i32, tensor<4x8xf32> -> tensor<8x16xf32>
    return %result : tensor<8x16xf32>
  }
  // CHECK-LABEL: func.func @insert_tile_dynamic
  // CHECK-DAG:     arith.index_cast
  // CHECK-DAG:     arith.divui
  // CHECK-DAG:     arith.remui
  // CHECK-DAG:     arith.muli
  // CHECK:         tensor.insert_slice

  // Dynamic 1D: no divui/remui needed (grid=[4], stride=1)
  func.func @extract_tile_dynamic_1d(%src: tensor<1024xf32>, %idx: i32) -> tensor<256xf32> {
    %result = xtle.extract_tile %src[%idx] {tile_shape = array<i64: 256>} : tensor<1024xf32>, i32 -> tensor<256xf32>
    return %result : tensor<256xf32>
  }
  // CHECK-LABEL: func.func @extract_tile_dynamic_1d
  // CHECK:         arith.index_cast
  // CHECK:         arith.muli
  // CHECK:         tensor.extract_slice
}
