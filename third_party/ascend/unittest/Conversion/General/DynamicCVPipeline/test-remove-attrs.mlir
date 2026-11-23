// RUN: triton-opt --remove-ssbuf-attr %s | FileCheck %s

module {
  // CHECK-LABEL: func.func @test_remove_core_type_and_block_id
  func.func @test_remove_core_type_and_block_id(%arg0: memref<1024x1024xf32>) {
    // CHECK: memref.alloc() : memref<1024x1024xf32>
    // CHECK-NOT: ssbuffer.core_type
    // CHECK-NOT: ssbuffer.block_id
    %memref = memref.alloc() {ssbuffer.core_type = "CUBE", ssbuffer.block_id = 1 : i32} : memref<1024x1024xf32>

    // CHECK: bufferization.to_tensor %{{.*}} : memref<1024x1024xf32> to tensor<1024x1024xf32>
    // CHECK-NOT: ssbuffer.core_type
    %tensor = bufferization.to_tensor %memref {ssbuffer.core_type = "CUBE"} : memref<1024x1024xf32> to tensor<1024x1024xf32>
    return
  }

  // CHECK-LABEL: func.func @test_remove_transfer_and_order_attrs
  func.func @test_remove_transfer_and_order_attrs(%arg0: tensor<128x128xf32>) {
    // CHECK: linalg.matmul
    // CHECK-NOT: ssbuffer.transfer_id
    // CHECK-NOT: ssbuffer.cube_first
    // CHECK-NOT: ssbuffer.vector_first
    %init = tensor.empty() : tensor<128x128xf32>
    %result = linalg.matmul {
      ssbuffer.transfer_id = 42 : i32,
      ssbuffer.cube_first,
      ssbuffer.vector_first
    } ins(%arg0, %arg0 : tensor<128x128xf32>, tensor<128x128xf32>)
      outs(%init : tensor<128x128xf32>) -> tensor<128x128xf32>

    return
  }

  // CHECK-LABEL: func.func @test_preserve_other_attributes
  func.func @test_preserve_other_attributes(%arg0: tensor<64x64xf32>) {
    // CHECK: linalg.transpose
    // CHECK-SAME: permutation = [1, 0]
    // CHECK-NOT: ssbuffer.core_type
    %out = tensor.empty() : tensor<64x64xf32>
    %transposed = linalg.transpose
      ins(%arg0 : tensor<64x64xf32>)
      outs(%out : tensor<64x64xf32>)
      permutation = [1, 0]
      {ssbuffer.core_type = "VECTOR"}

    return
  }

  // CHECK-LABEL: func.func @test_nested_ops
  func.func @test_nested_ops(%arg0: i1, %arg1: tensor<128xf32>) {
    // CHECK: scf.if
    %res = scf.if %arg0 -> (tensor<128xf32>) {
      // CHECK: arith.addf
      // CHECK-NOT: ssbuffer.core_type
      %add = arith.addf %arg1, %arg1 {ssbuffer.core_type = "VECTOR"} : tensor<128xf32>
      scf.yield %add : tensor<128xf32>
    } else {
      scf.yield %arg1 : tensor<128xf32>
    }
    return
  }
}
