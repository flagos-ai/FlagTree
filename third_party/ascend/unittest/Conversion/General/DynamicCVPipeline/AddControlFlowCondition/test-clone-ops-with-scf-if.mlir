// RUN: triton-opt --clone-ops %s --allow-unregistered-dialect | FileCheck %s

// CHECK: func.func @test_clone_ops
func.func @test_clone_ops(%arg0: index, %arg1: index, %arg2: tensor<32x128xbf16>, %arg3: index, %arg4: index, %arg5: memref<?xbf16>) {
  %c0_i32 = arith.constant 0 : i32
  %c1_i32 = arith.constant 1 : i32
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c32 = arith.constant 32 : index
  %cst_7 = arith.constant 7.0 : bf16
  %for_op = scf.for %i = %c0 to %c1 step %c1 iter_args(%iter = %c0_i32) -> i32 {
    %cmp = arith.cmpi slt, %arg0, %c32 {ssbuffer.block_id = 7 : i32} : index
    %alloc_50 = memref.alloc() {ssbuffer.block_id = 7 : i32} : memref<32x128xbf16>
    scf.if %cmp {
      linalg.fill {ssbuffer.block_id = 7 : i32} ins(%cst_7 : bf16) outs(%alloc_50 : memref<32x128xbf16>)
    } {hivm.unlikely_condition, ssbuffer.block_id = 7 : i32}
    %reinterpret_cast_51 = memref.reinterpret_cast %arg5 to offset: [%arg0], sizes: [32, 128], strides: [640, 1] {ssbuffer.block_id = 7 : i32} : memref<?xbf16> to memref<32x128xbf16, strided<[640, 1], offset: ?>>
    %subview_52 = memref.subview %reinterpret_cast_51[0, 0] [%arg1, 128] [1, 1] {ssbuffer.block_id = 7 : i32} : memref<32x128xbf16, strided<[640, 1], offset: ?>> to memref<?x128xbf16, strided<[640, 1], offset: ?>>
    %subview_53 = memref.subview %alloc_50[0, 0] [%arg1, 128] [1, 1] {ssbuffer.block_id = 7 : i32} : memref<32x128xbf16> to memref<?x128xbf16, strided<[128, 1]>>
    memref.copy %subview_52, %subview_53 {ssbuffer.block_id = 7 : i32} : memref<?x128xbf16, strided<[640, 1], offset: ?>> to memref<?x128xbf16, strided<[128, 1]>>
    %92 = bufferization.to_tensor %alloc_50 restrict writable {ssbuffer.block_id = 7 : i32} : memref<32x128xbf16> to tensor<32x128xbf16>
    %alloc_60 = memref.alloc() {ssbuffer.block_id = 8 : i32} : memref<32x128xbf16>
    scf.if %cmp {
      linalg.fill {ssbuffer.block_id = 8 : i32} ins(%cst_7 : bf16) outs(%alloc_60 : memref<32x128xbf16>)
    } {hivm.unlikely_condition, ssbuffer.block_id = 8 : i32}
    %reinterpret_cast_61 = memref.reinterpret_cast %arg5 to offset: [%arg0], sizes: [32, 128], strides: [640, 1] {ssbuffer.block_id = 8 : i32} : memref<?xbf16> to memref<32x128xbf16, strided<[640, 1], offset: ?>>
    %subview_62 = memref.subview %reinterpret_cast_61[0, 0] [%arg1, 128] [1, 1] {ssbuffer.block_id = 8 : i32} : memref<32x128xbf16, strided<[640, 1], offset: ?>> to memref<?x128xbf16, strided<[640, 1], offset: ?>>
    %subview_63 = memref.subview %alloc_60[0, 0] [%arg1, 128] [1, 1] {ssbuffer.block_id = 8 : i32} : memref<32x128xbf16> to memref<?x128xbf16, strided<[128, 1]>>
    memref.copy %subview_62, %subview_63 {ssbuffer.block_id = 8 : i32} : memref<?x128xbf16, strided<[640, 1], offset: ?>> to memref<?x128xbf16, strided<[128, 1]>>
    %93 = bufferization.to_tensor %alloc_60 restrict writable {ssbuffer.block_id = 8 : i32} : memref<32x128xbf16> to tensor<32x128xbf16>
    %empty_transpose = tensor.empty() {ssbuffer.block_id = 8 : i32} : tensor<128x32xbf16>
    // CHECK: scf.if
    // CHECK: } {hivm.unlikely_condition, ssbuffer.block_id = 8 : i32, ssbuffer.clone = 7 : i32}
    // CHECK: {{.*}} = memref.reinterpret_cast {{.*}}{ssbuffer.block_id = 8 : i32, ssbuffer.clone = 7 : i32}
    // CHECK: {{.*}} = memref.subview {{.*}}{ssbuffer.block_id = 8 : i32, ssbuffer.clone = 7 : i32}
    // CHECK: memref.copy {{.*}} {ssbuffer.block_id = 8 : i32, ssbuffer.clone = 7 : i32}
    // CHECK: {{.*}} = bufferization.to_tensor {{.*}} {ssbuffer.block_id = 8 : i32, ssbuffer.clone = 7 : i32}
    %transposed_54 = linalg.transpose ins(%92 : tensor<32x128xbf16>) outs(%empty_transpose : tensor<128x32xbf16>) permutation = [1, 0] {ssbuffer.block_id = 8 : i32}
    %94 = tensor.empty() {ssbuffer.block_id = 8 : i32} : tensor<32x32xf32>
    %95 = linalg.matmul {input_precision = "ieee", ssbuffer.block_id = 8 : i32} ins(%93, %transposed_54 : tensor<32x128xbf16>, tensor<128x32xbf16>) outs(%94 : tensor<32x32xf32>) -> tensor<32x32xf32>

    scf.yield %iter : i32
  } {ssbuffer.main_loop}

  return
}
