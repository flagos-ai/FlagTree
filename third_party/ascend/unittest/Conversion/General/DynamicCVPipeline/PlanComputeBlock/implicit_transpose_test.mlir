// RUN: triton-opt --plan-compute-block %s | FileCheck %s

module {
  // Test Case 1 & 2: Implicit Transpose with unlikely condition fill
  // Verifies that:
  // 1. %to_tensor, %alloc, and %fill are colored as VECTOR instead of CUBE.
  // 2. The fill inside the unlikely scf.if is colored as VECTOR.
  // 3. The annotation.mark has the same block_id as its defining op (to_tensor).
  func.func @test_implicit_transpose_vector_coloring(%arg0: memref<?xf32>, %arg1: memref<?xf32>, %cond: i1) -> tensor<64x64xf32> {
    // CHECK-LABEL: func.func @test_implicit_transpose_vector_coloring(

    // CHECK: %[[ALLOC:.+]] = memref.alloc() {ssbuffer.block_id = [[B0:[0-9]+]] : i32, ssbuffer.core_type = "VECTOR"}
    %alloc = memref.alloc() : memref<64x64xf32>
    %cst = arith.constant 0.000000e+00 : f32

    // CHECK: scf.if
    // CHECK: linalg.fill {ssbuffer.block_id = {{[0-9]+}} : i32, ssbuffer.core_type = "VECTOR"}
    scf.if %cond {
      linalg.fill ins(%cst : f32) outs(%alloc : memref<64x64xf32>)
    } {hivm.unlikely_condition}

    // CHECK: %[[TENSOR:.+]] = bufferization.to_tensor %[[ALLOC]] restrict writable {ssbuffer.block_id = [[B1:[0-9]+]] : i32, ssbuffer.core_type = "VECTOR"}
    %to_tensor = bufferization.to_tensor %alloc restrict writable : memref<64x64xf32> to tensor<64x64xf32>
    // CHECK: annotation.mark %[[TENSOR]] {MayImplicitTransposeWithLastAxis, ssbuffer.block_id = [[B1]] : i32, ssbuffer.core_type = "VECTOR"}
    annotation.mark %to_tensor {MayImplicitTransposeWithLastAxis} : tensor<64x64xf32>

    %empty = tensor.empty() : tensor<64x64xf32>
    %fill = linalg.fill ins(%cst : f32) outs(%empty : tensor<64x64xf32>) -> tensor<64x64xf32>
    %other = tensor.empty() : tensor<64x64xf32>

    %matmul = linalg.matmul ins(%to_tensor, %other : tensor<64x64xf32>, tensor<64x64xf32>) outs(%fill : tensor<64x64xf32>) -> tensor<64x64xf32>
    return %matmul : tensor<64x64xf32>
  }

  // Test Case 3: Normal ToTensor without Implicit Transpose
  // Verifies that a standard matmul-feeding to_tensor without the annotation is still colored as CUBE.
  func.func @test_normal_to_tensor_cube_coloring(%arg0: memref<?xf32>) -> tensor<64x64xf32> {
    // CHECK-LABEL: func.func @test_normal_to_tensor_cube_coloring(

    // CHECK: %[[ALLOC_CUBE:.+]] = memref.alloc() {ssbuffer.block_id = [[B2:[0-9]+]] : i32, ssbuffer.core_type = "CUBE"}
    %alloc = memref.alloc() : memref<64x64xf32>

    // CHECK: %[[TENSOR_CUBE:.+]] = bufferization.to_tensor %[[ALLOC_CUBE]] restrict writable {ssbuffer.block_id = [[B3:[0-9]+]] : i32, ssbuffer.core_type = "CUBE"}
    %to_tensor = bufferization.to_tensor %alloc restrict writable : memref<64x64xf32> to tensor<64x64xf32>
    %cst = arith.constant 0.000000e+00 : f32
    %empty = tensor.empty() : tensor<64x64xf32>
    %fill = linalg.fill ins(%cst : f32) outs(%empty : tensor<64x64xf32>) -> tensor<64x64xf32>
    %other = tensor.empty() : tensor<64x64xf32>

    %matmul = linalg.matmul ins(%to_tensor, %other : tensor<64x64xf32>, tensor<64x64xf32>) outs(%fill : tensor<64x64xf32>) -> tensor<64x64xf32>
    return %matmul : tensor<64x64xf32>
  }

  // Test Case 4: Block ID alignment check for annotation.mark
  // Verifies that annotation.mark always follows its defining operation (%to_tensor)
  // in terms of block assignment, preventing mismatched block IDs.
  func.func @test_mark_block_id_alignment(%arg0: memref<?xf32>, %cond: i1) -> tensor<64x64xf32> {
    // CHECK-LABEL: func.func @test_mark_block_id_alignment(

    %alloc = memref.alloc() : memref<64x64xf32>

    // CHECK: %[[TENSOR_VAL:.+]] = bufferization.to_tensor %{{.+}} restrict writable {ssbuffer.block_id = [[B4:[0-9]+]] : i32, ssbuffer.core_type = "VECTOR"}
    %to_tensor = bufferization.to_tensor %alloc restrict writable : memref<64x64xf32> to tensor<64x64xf32>
    // CHECK: annotation.mark %[[TENSOR_VAL]] {MayImplicitTransposeWithLastAxis, ssbuffer.block_id = [[B4]] : i32, ssbuffer.core_type = "VECTOR"}
    annotation.mark %to_tensor {MayImplicitTransposeWithLastAxis} : tensor<64x64xf32>

    %cst = arith.constant 0.000000e+00 : f32
    %empty = tensor.empty() : tensor<64x64xf32>
    %fill = linalg.fill ins(%cst : f32) outs(%empty : tensor<64x64xf32>) -> tensor<64x64xf32>
    %other = tensor.empty() : tensor<64x64xf32>

    %matmul = linalg.matmul ins(%to_tensor, %other : tensor<64x64xf32>, tensor<64x64xf32>) outs(%fill : tensor<64x64xf32>) -> tensor<64x64xf32>
    return %matmul : tensor<64x64xf32>
  }
}
