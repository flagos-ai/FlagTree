// RUN: triton-opt --plan-compute-block %s | FileCheck %s

module {
  // CHECK-LABEL: func.func @copy_to_subview_feeds_matmul(
  // CHECK: memref.copy {{.*}}ssbuffer.core_type = "CUBE"
  // CHECK: bufferization.to_tensor {{.*}}ssbuffer.core_type = "CUBE"
  // CHECK: linalg.matmul {{.*}}ssbuffer.core_type = "CUBE"
  func.func @copy_to_subview_feeds_matmul(%src: memref<64x64xf16>, %rhs: tensor<64x64xf16>) -> tensor<64x64xf32> {
    %c0 = arith.constant 0 : index
    %c64 = arith.constant 64 : index
    %alloc = memref.alloc() {ssbuffer.core_type = "VECTOR"} : memref<64x64xf16>
    %src_view = memref.subview %src[%c0, %c0] [%c64, %c64] [1, 1]
      : memref<64x64xf16> to memref<?x?xf16, strided<[64, 1], offset: ?>>
    %dst_view = memref.subview %alloc[%c0, %c0] [%c64, %c64] [1, 1]
      : memref<64x64xf16> to memref<?x?xf16, strided<[64, 1], offset: ?>>
    memref.copy %src_view, %dst_view {ssbuffer.core_type = "VECTOR"}
      : memref<?x?xf16, strided<[64, 1], offset: ?>> to memref<?x?xf16, strided<[64, 1], offset: ?>>
    %lhs = bufferization.to_tensor %alloc restrict writable {ssbuffer.core_type = "VECTOR"}
      : memref<64x64xf16> to tensor<64x64xf16>
    %out = tensor.empty() {ssbuffer.core_type = "VECTOR"} : tensor<64x64xf32>
    %mm = linalg.matmul {input_precision = "ieee", ssbuffer.core_type = "CUBE"}
      ins(%lhs, %rhs : tensor<64x64xf16>, tensor<64x64xf16>)
      outs(%out : tensor<64x64xf32>) -> tensor<64x64xf32>
    return %mm : tensor<64x64xf32>
  }

  // CHECK-LABEL: func.func @top_level_arg_subviews_do_not_alias(
  // CHECK: [[B_VIEW:%[A-Za-z0-9_]+]] = memref.subview %arg1
  // CHECK-NEXT: linalg.fill {ssbuffer.block_id = [[VECTOR:[0-9]+]] : i32, ssbuffer.core_type = "VECTOR"} ins({{.*}} : f16) outs([[B_VIEW]]
  // CHECK: [[A_VIEW:%[A-Za-z0-9_]+]] = memref.subview %arg0
  // CHECK-NEXT: [[A_TENSOR:%[0-9]+]] = bufferization.to_tensor [[A_VIEW]] restrict writable {ssbuffer.block_id = [[CUBE:[0-9]+]] : i32, ssbuffer.core_type = "CUBE"}
  // CHECK: linalg.matmul {{.*}}ssbuffer.block_id = [[CUBE]] : i32, ssbuffer.core_type = "CUBE"
  func.func @top_level_arg_subviews_do_not_alias(
    %lhs_gm: memref<64x64xf16>,
    %rhs_gm: memref<64x64xf16>,
    %rhs: tensor<64x64xf16>) -> tensor<64x64xf32> {
    %c0 = arith.constant 0 : index
    %zero_f16 = arith.constant 0.0 : f16
    %zero_f32 = arith.constant 0.0 : f32

    %rhs_view = memref.subview %rhs_gm[%c0, %c0] [64, 64] [1, 1]
      : memref<64x64xf16> to memref<64x64xf16, strided<[64, 1], offset: ?>>
    linalg.fill ins(%zero_f16 : f16) outs(%rhs_view : memref<64x64xf16, strided<[64, 1], offset: ?>>)

    %lhs_view = memref.subview %lhs_gm[%c0, %c0] [64, 64] [1, 1]
      : memref<64x64xf16> to memref<64x64xf16, strided<[64, 1], offset: ?>>
    %lhs = bufferization.to_tensor %lhs_view restrict writable
      : memref<64x64xf16, strided<[64, 1], offset: ?>> to tensor<64x64xf16>
    %out = tensor.empty() : tensor<64x64xf32>
    %init = linalg.fill ins(%zero_f32 : f32) outs(%out : tensor<64x64xf32>) -> tensor<64x64xf32>
    %mm = linalg.matmul {input_precision = "ieee"}
      ins(%lhs, %rhs : tensor<64x64xf16>, tensor<64x64xf16>)
      outs(%init : tensor<64x64xf32>) -> tensor<64x64xf32>
    return %mm : tensor<64x64xf32>
  }
}
