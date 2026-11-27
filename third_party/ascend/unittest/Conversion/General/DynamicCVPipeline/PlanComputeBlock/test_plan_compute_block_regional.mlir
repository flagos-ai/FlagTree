// RUN: triton-opt --plan-compute-block %s | FileCheck %s

module {
  // ============================================================================
  // 1. extracted_load_store_stays_vector
  // ============================================================================
  //
  // CHECK-LABEL: func.func @extracted_load_store_stays_vector(
  // CHECK: %[[ALLOC:[A-Za-z0-9_]+]] = memref.alloc() {ssbuffer.block_id = [[B_ID:[0-9]+]] : i32, ssbuffer.core_type = "VECTOR"}
  // CHECK: scf.for
  // CHECK:   %[[SUBVIEW:[A-Za-z0-9_]+]] = memref.subview %[[ALLOC]]
  // CHECK:   memref.copy %{{.*}}, %[[SUBVIEW]] {ssbuffer.block_id = [[B_ID]] : i32, ssbuffer.core_type = "VECTOR"}
  // CHECK: %[[LHS:[A-Za-z0-9_]+]] = bufferization.to_tensor %[[ALLOC]] restrict writable {ssbuffer.block_id = [[B_ID]] : i32, ssbuffer.core_type = "VECTOR"}
  // CHECK: linalg.matmul {ssbuffer.block_id = {{[0-9]+}} : i32, ssbuffer.core_type = "CUBE"}
  func.func @extracted_load_store_stays_vector(
      %arg0: memref<?xf16>,
      %rhs: tensor<64x64xf16>) -> tensor<32x64xf32> {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c32 = arith.constant 32 : index
    %alloc = memref.alloc() : memref<32x64xf16>

    scf.for %arg1 = %c0 to %c32 step %c1 {
      %subview = memref.subview %alloc[%arg1, 0] [1, 64] [1, 1] : memref<32x64xf16> to memref<1x64xf16, strided<[64, 1], offset: ?>>
      %reinterpret_cast = memref.reinterpret_cast %arg0 to offset: [0], sizes: [1, 64], strides: [64, 1] : memref<?xf16> to memref<1x64xf16, strided<[64, 1], offset: ?>>
      memref.copy %reinterpret_cast, %subview : memref<1x64xf16, strided<[64, 1], offset: ?>> to memref<1x64xf16, strided<[64, 1], offset: ?>>
    } {ExtractedLoadOrStore}

    %lhs = bufferization.to_tensor %alloc restrict writable : memref<32x64xf16> to tensor<32x64xf16>
    %out = tensor.empty() : tensor<32x64xf32>
    %cst_f32 = arith.constant 0.0 : f32
    %init = linalg.fill ins(%cst_f32 : f32) outs(%out : tensor<32x64xf32>) -> tensor<32x64xf32>
    %mm = linalg.matmul ins(%lhs, %rhs : tensor<32x64xf16>, tensor<64x64xf16>) outs(%init : tensor<32x64xf32>) -> tensor<32x64xf32>
    return %mm : tensor<32x64xf32>
  }

  // ============================================================================
  // 2. cube_control_flow_inheritance
  // ============================================================================
  //
  // CHECK-LABEL: func.func @cube_control_flow_inheritance(
  // CHECK: %[[ALLOC:[A-Za-z0-9_]+]] = memref.alloc() {ssbuffer.block_id = [[B_ID:[0-9]+]] : i32, ssbuffer.core_type = "CUBE"}
  // CHECK: scf.if
  // CHECK:   linalg.fill {ssbuffer.block_id = [[B_ID]] : i32, ssbuffer.core_type = "CUBE"}
  // CHECK: %[[LHS:[A-Za-z0-9_]+]] = bufferization.to_tensor %[[ALLOC]] restrict writable {ssbuffer.block_id = [[B_ID]] : i32, ssbuffer.core_type = "CUBE"}
  // CHECK: linalg.matmul {ssbuffer.block_id = [[B_ID]] : i32, ssbuffer.core_type = "CUBE"}
  func.func @cube_control_flow_inheritance(
      %cond: i1,
      %arg0: tensor<64x64xf16>,
      %arg1: tensor<64x64xf32>) -> tensor<64x64xf32> {
    %alloc = memref.alloc() {ssbuffer.core_type = "CUBE"} : memref<64x64xf16>
    %cst = arith.constant 0.0 : f16

    scf.if %cond {
      linalg.fill {ssbuffer.core_type = "CUBE"} ins(%cst : f16) outs(%alloc : memref<64x64xf16>)
    }

    %lhs = bufferization.to_tensor %alloc restrict writable {ssbuffer.core_type = "CUBE"} : memref<64x64xf16> to tensor<64x64xf16>
    %mm = linalg.matmul {ssbuffer.core_type = "CUBE"} ins(%lhs, %arg0 : tensor<64x64xf16>, tensor<64x64xf16>) outs(%arg1 : tensor<64x64xf32>) -> tensor<64x64xf32>
    return %mm : tensor<64x64xf32>
  }

  // ============================================================================
  // 3. vector_control_flow_inheritance
  // ============================================================================
  //
  // CHECK-LABEL: func.func @vector_control_flow_inheritance(
  // CHECK: %[[ALLOC:[A-Za-z0-9_]+]] = memref.alloc() {ssbuffer.block_id = [[B_ID:[0-9]+]] : i32, ssbuffer.core_type = "VECTOR"}
  // CHECK: scf.if
  // CHECK:   linalg.fill {ssbuffer.block_id = [[B_ID]] : i32, ssbuffer.core_type = "VECTOR"}
  // CHECK: memref.copy %[[ALLOC]], %{{.*}} {ssbuffer.block_id = [[B_ID]] : i32, ssbuffer.core_type = "VECTOR"}
  func.func @vector_control_flow_inheritance(%cond: i1, %arg0: memref<64x64xf32>) {
    %cst = arith.constant 0.0 : f32
    %alloc = memref.alloc() {ssbuffer.core_type = "VECTOR"} : memref<64x64xf32>

    scf.if %cond {
      linalg.fill {ssbuffer.core_type = "VECTOR"} ins(%cst : f32) outs(%alloc : memref<64x64xf32>)
    }

    memref.copy %alloc, %arg0 {ssbuffer.core_type = "VECTOR"} : memref<64x64xf32> to memref<64x64xf32>
    return
  }
}
