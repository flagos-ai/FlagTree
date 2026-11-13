// RUN: triton-opt --add-block-id-for-control-ops --data-dependency-analysis --inter-core-transfer-and-sync --mark-main-loop %s | FileCheck %s

module {
  func.func @test_unaligned_int8_transpose() {
    %cst = arith.constant {ssbuffer.block_id = 0 : i32, ssbuffer.core_type = "VECTOR"} 1 : i8
    %t0 = tensor.empty() {ssbuffer.block_id = 1 : i32, ssbuffer.core_type = "VECTOR"} : tensor<3x111xi8>
    %fill = linalg.fill {ssbuffer.block_id = 1 : i32, ssbuffer.core_type = "VECTOR"} ins(%cst : i8) outs(%t0 : tensor<3x111xi8>) -> tensor<3x111xi8>
    %exp = arith.addi %fill, %fill {ssbuffer.block_id = 1 : i32, ssbuffer.core_type = "VECTOR"} : tensor<3x111xi8>

    %alloc = memref.alloc() {ssbuffer.block_id = 2 : i32, ssbuffer.core_type = "CUBE"} : memref<3x104xi8>
    %t1 = bufferization.to_tensor %alloc {ssbuffer.block_id = 2 : i32, ssbuffer.core_type = "CUBE"} : memref<3x104xi8> to tensor<3x104xi8>
    %empty = tensor.empty() {ssbuffer.block_id = 2 : i32, ssbuffer.core_type = "CUBE"} : tensor<111x104xi32>
    %cst_cube = arith.constant {ssbuffer.block_id = 2 : i32, ssbuffer.core_type = "CUBE"} 0 : i32
    %init = linalg.fill {ssbuffer.block_id = 2 : i32, ssbuffer.core_type = "CUBE"} ins(%cst_cube : i32) outs(%empty : tensor<111x104xi32>) -> tensor<111x104xi32>
    %empty2 = tensor.empty() {ssbuffer.block_id = 2 : i32, ssbuffer.core_type = "CUBE"} : tensor<111x3xi8>
    %trans = linalg.transpose ins(%exp : tensor<3x111xi8>) outs(%empty2 : tensor<111x3xi8>) permutation = [1, 0] {ssbuffer.block_id = 2 : i32, ssbuffer.core_type = "CUBE"}
    %mat = linalg.matmul {ssbuffer.block_id = 2 : i32, ssbuffer.core_type = "CUBE"} ins(%trans, %t1 : tensor<111x3xi8>, tensor<3x104xi8>) outs(%init : tensor<111x104xi32>) -> tensor<111x104xi32>
    return
  }
}

// CHECK-LABEL: func.func @test_unaligned_int8_transpose

// CHECK: %[[ADD_2:[a-z0-9_]+]] = arith.addi
// CHECK: tensor.insert_slice %[[ADD_2]] into {{.*}}[0, 0] [3, 111] [1, 1] {{.*}} : tensor<3x111xi8> into tensor<32x128xi8>
// CHECK: tensor.reshape
// CHECK: linalg.transpose
// CHECK: %[[RESHAPE_1:[a-z0-9_]+]] = tensor.reshape
// CHECK: hivm.hir.copy ins(%[[RESHAPE_1]] : tensor<4x1x32x32xi8>) outs({{.*}} : memref<4x1x32x32xi8, #hivm.address_space<cbuf>>)

// CHECK: hivm.hir.convert_layout {{.*}} output_shape [3, 111] {{.*}} : (memref<4x1x32x32xi8, #hivm.address_space<cbuf>>) -> memref<3x111xi8, #hivm.address_space<cbuf>>
// CHECK: memref.memory_space_cast
// CHECK: %[[TENSOR_11:[a-z0-9_]+]] = bufferization.to_tensor
// CHECK: %[[TRANSPOSED_4:[a-z0-9_]+]] = linalg.transpose ins(%[[TENSOR_11]] : tensor<3x111xi8>)
// CHECK: linalg.matmul {{.*}} ins(%[[TRANSPOSED_4]], {{.*}} : tensor<111x3xi8>, tensor<3x104xi8>) outs({{.*}} : tensor<111x104xi32>) -> tensor<111x104xi32>
