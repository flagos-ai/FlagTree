// RUN: triton-opt --merge-cube-block %s | FileCheck %s

module attributes {hacc.target = #hacc.target<"Ascend950PR_9579">} {
  // ============================================
  // Test Case 1: @test_merge_cube_blocks_with_vector
  // ============================================
  // Scenario: Two cube blocks with same vector predecessors and successors
  // - Vector block 1 (block_id=1) produces vec_tensor1
  // - Cube block 1 (block_id=2) uses vec_tensor1, produces cube_tensor1
  // - Cube block 2 (block_id=3) uses vec_tensor1, produces cube_tensor2
  // - Vector block 2 (block_id=4) uses cube_tensor1 and cube_tensor2
  // Expected: The two cube blocks should be merged (block_id=3 -> block_id=2)
  // Wrapped in two-layer for loop (only innermost should be processed)
  // ============================================
  // CHECK-LABEL: func.func @test_merge_cube_blocks_with_vector
  func.func @test_merge_cube_blocks_with_vector(%arg0: memref<?xbf16>, %arg1: memref<?xbf16>, %arg2: memref<?xf32>) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c128 = arith.constant 128 : index
    %cst = arith.constant 0.000000e+00 : bf16
    %cst_f32 = arith.constant 0.000000e+00 : f32

    scf.for %i = %c0 to %c128 step %c1 {
      scf.for %j = %c0 to %c128 step %c1 {
        %vec_alloc1 = memref.alloc() {ssbuffer.block_id = 1 : i32, ssbuffer.core_type = "VECTOR"} : memref<128x128xbf16>
        %vec_cond1 = arith.constant 1 : i1
        scf.if %vec_cond1 {
          linalg.fill {ssbuffer.block_id = 1 : i32, ssbuffer.core_type = "VECTOR"} ins(%cst : bf16) outs(%vec_alloc1 : memref<128x128xbf16>)
        } {hivm.unlikely_condition, ssbuffer.block_id = 1 : i32}
        %vec_tensor1 = bufferization.to_tensor %vec_alloc1 restrict writable {ssbuffer.block_id = 1 : i32, ssbuffer.core_type = "VECTOR"} : memref<128x128xbf16> to tensor<128x128xbf16>

        // CHECK: %{{.*}} = memref.alloc() {ssbuffer.block_id = 2 : i32, ssbuffer.core_type = "CUBE"} : memref<128x128xbf16>
        %cube_alloc1 = memref.alloc() {ssbuffer.block_id = 2 : i32, ssbuffer.core_type = "CUBE"} : memref<128x128xbf16>
        %cube_cond1 = arith.constant 1 : i1
        // CHECK: scf.if %{{.*}} {
        // CHECK:   linalg.fill {ssbuffer.block_id = 2 : i32, ssbuffer.core_type = "CUBE"}
        scf.if %cube_cond1 {
          linalg.fill {ssbuffer.block_id = 2 : i32, ssbuffer.core_type = "CUBE"} ins(%cst : bf16) outs(%cube_alloc1 : memref<128x128xbf16>)
        } {hivm.unlikely_condition, ssbuffer.block_id = 2 : i32}
        %cube_tensor1 = bufferization.to_tensor %cube_alloc1 restrict writable {ssbuffer.block_id = 2 : i32, ssbuffer.core_type = "CUBE"} : memref<128x128xbf16> to tensor<128x128xbf16>
        %cube_out1 = tensor.empty() {ssbuffer.block_id = 2 : i32, ssbuffer.core_type = "CUBE"} : tensor<128x128xf32>
        // CHECK: linalg.matmul {input_precision = "ieee", ssbuffer.block_id = 2 : i32, ssbuffer.core_type = "CUBE"}
        %cube_matmul1 = linalg.matmul {input_precision = "ieee", ssbuffer.block_id = 2 : i32, ssbuffer.core_type = "CUBE"} ins(%vec_tensor1, %cube_tensor1 : tensor<128x128xbf16>, tensor<128x128xbf16>) outs(%cube_out1 : tensor<128x128xf32>) -> tensor<128x128xf32>

        // CHECK: %{{.*}} = memref.alloc() {ssbuffer.block_id = 2 : i32, ssbuffer.core_type = "CUBE"} : memref<128x128xbf16>
        %cube_alloc2 = memref.alloc() {ssbuffer.block_id = 3 : i32, ssbuffer.core_type = "CUBE"} : memref<128x128xbf16>
        %cube_cond2 = arith.constant 1 : i1
        // CHECK: scf.if %{{.*}} {
        // CHECK:   linalg.fill {ssbuffer.block_id = 2 : i32, ssbuffer.core_type = "CUBE"}
        scf.if %cube_cond2 {
          linalg.fill {ssbuffer.block_id = 3 : i32, ssbuffer.core_type = "CUBE"} ins(%cst : bf16) outs(%cube_alloc2 : memref<128x128xbf16>)
        } {hivm.unlikely_condition, ssbuffer.block_id = 3 : i32}
        %cube_tensor2 = bufferization.to_tensor %cube_alloc2 restrict writable {ssbuffer.block_id = 3 : i32, ssbuffer.core_type = "CUBE"} : memref<128x128xbf16> to tensor<128x128xbf16>
        %cube_out2 = tensor.empty() {ssbuffer.block_id = 3 : i32, ssbuffer.core_type = "CUBE"} : tensor<128x128xf32>
        // CHECK: linalg.matmul {input_precision = "ieee", ssbuffer.block_id = 2 : i32, ssbuffer.core_type = "CUBE"}
        %cube_matmul2 = linalg.matmul {input_precision = "ieee", ssbuffer.block_id = 3 : i32, ssbuffer.core_type = "CUBE"} ins(%vec_tensor1, %cube_tensor2 : tensor<128x128xbf16>, tensor<128x128xbf16>) outs(%cube_out2 : tensor<128x128xf32>) -> tensor<128x128xf32>

        %vec_alloc2 = memref.alloc() {ssbuffer.block_id = 4 : i32, ssbuffer.core_type = "VECTOR"} : memref<128x128xf32>
        %vec_cond2 = arith.constant 1 : i1
        scf.if %vec_cond2 {
          linalg.fill {ssbuffer.block_id = 4 : i32, ssbuffer.core_type = "VECTOR"} ins(%cst_f32 : f32) outs(%vec_alloc2 : memref<128x128xf32>)
        } {hivm.unlikely_condition, ssbuffer.block_id = 4 : i32}
        // Uses both cube_matmul1 and cube_matmul2
        %vec_add = arith.addf %cube_matmul1, %cube_matmul2 {ssbuffer.block_id = 4 : i32, ssbuffer.core_type = "VECTOR"} : tensor<128x128xf32>
        %vec_tensor2 = bufferization.to_tensor %vec_alloc2 restrict writable {ssbuffer.block_id = 4 : i32, ssbuffer.core_type = "VECTOR"} : memref<128x128xf32> to tensor<128x128xf32>

        scf.yield
      }
    }

    return
  }

  // ============================================
  // Test Case 2: @test_no_merge_with_extra_vector_dependency
  // ============================================
  // Scenario: Based on scenario 1, add an extra vector node pointing to one cube block
  // - Vector block 5 (block_id=5) produces vec_tensor1
  // - Vector block 9 (block_id=9) produces vec_tensor3
  // - Cube block 6 (block_id=6) uses vec_tensor1, produces cube_matmul1
  // - Cube block 7 (block_id=7) uses vec_tensor3, produces cube_matmul2
  // - Vector block 8 (block_id=8) uses cube_matmul1 and cube_matmul2
  // Expected: The two cube blocks should NOT be merged (different source nodes)
  // Wrapped in two-layer for loop
  // ============================================
  // CHECK-LABEL: func.func @test_no_merge_with_extra_vector_dependency
  func.func @test_no_merge_with_extra_vector_dependency(%arg0: memref<?xbf16>, %arg1: memref<?xbf16>, %arg2: memref<?xf32>) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c128 = arith.constant 128 : index
    %cst = arith.constant 0.000000e+00 : bf16
    %cst_f32 = arith.constant 0.000000e+00 : f32

    // Outer for loop
    scf.for %i = %c0 to %c128 step %c1 {
      // Inner for loop (innermost, should be processed)
      scf.for %j = %c0 to %c128 step %c1 {
        // Vector block 1 (predecessor for cube block 1)
        %vec_alloc1 = memref.alloc() {ssbuffer.block_id = 5 : i32, ssbuffer.core_type = "VECTOR"} : memref<128x128xbf16>
        %vec_cond1 = arith.constant 1 : i1
        scf.if %vec_cond1 {
          linalg.fill {ssbuffer.block_id = 5 : i32, ssbuffer.core_type = "VECTOR"} ins(%cst : bf16) outs(%vec_alloc1 : memref<128x128xbf16>)
        } {hivm.unlikely_condition, ssbuffer.block_id = 5 : i32}
        %vec_tensor1 = bufferization.to_tensor %vec_alloc1 restrict writable {ssbuffer.block_id = 5 : i32, ssbuffer.core_type = "VECTOR"} : memref<128x128xbf16> to tensor<128x128xbf16>

        // Cube block 1 (block_id=2) - uses vec_tensor1
        // CHECK: memref.alloc() {ssbuffer.block_id = 7 : i32, ssbuffer.core_type = "CUBE"}
        %cube_alloc1 = memref.alloc() {ssbuffer.block_id = 6 : i32, ssbuffer.core_type = "CUBE"} : memref<128x128xbf16>
        %cube_cond1 = arith.constant 1 : i1
        // CHECK: scf.if %{{.*}} {
        // CHECK:   linalg.fill {ssbuffer.block_id = 7 : i32, ssbuffer.core_type = "CUBE"}
        scf.if %cube_cond1 {
          linalg.fill {ssbuffer.block_id = 6 : i32, ssbuffer.core_type = "CUBE"} ins(%cst : bf16) outs(%cube_alloc1 : memref<128x128xbf16>)
        } {hivm.unlikely_condition, ssbuffer.block_id = 6 : i32}
        %cube_tensor1 = bufferization.to_tensor %cube_alloc1 restrict writable {ssbuffer.block_id = 6 : i32, ssbuffer.core_type = "CUBE"} : memref<128x128xbf16> to tensor<128x128xbf16>
        %cube_out1 = tensor.empty() {ssbuffer.block_id = 6 : i32, ssbuffer.core_type = "CUBE"} : tensor<128x128xf32>
        // CHECK: linalg.matmul {input_precision = "ieee", ssbuffer.block_id = 7 : i32, ssbuffer.core_type = "CUBE"}
        %cube_matmul1 = linalg.matmul {input_precision = "ieee", ssbuffer.block_id = 6 : i32, ssbuffer.core_type = "CUBE"} ins(%vec_tensor1, %cube_tensor1 : tensor<128x128xbf16>, tensor<128x128xbf16>) outs(%cube_out1 : tensor<128x128xf32>) -> tensor<128x128xf32>

        // Extra vector block (block_id=5) - only used by cube block 2
        %vec_alloc3 = memref.alloc() {ssbuffer.block_id = 9 : i32, ssbuffer.core_type = "VECTOR"} : memref<128x128xbf16>
        %vec_cond3 = arith.constant 1 : i1
        scf.if %vec_cond3 {
          linalg.fill {ssbuffer.block_id = 9 : i32, ssbuffer.core_type = "VECTOR"} ins(%cst : bf16) outs(%vec_alloc3 : memref<128x128xbf16>)
        } {hivm.unlikely_condition, ssbuffer.block_id = 9 : i32}
        %vec_tensor3 = bufferization.to_tensor %vec_alloc3 restrict writable {ssbuffer.block_id = 9 : i32, ssbuffer.core_type = "VECTOR"} : memref<128x128xbf16> to tensor<128x128xbf16>

        // Cube block 2 (block_id=3) - should NOT be merged (different source nodes)
        // Uses vec_tensor3
        // CHECK: memref.alloc() {ssbuffer.block_id = 7 : i32, ssbuffer.core_type = "CUBE"}
        %cube_alloc2 = memref.alloc() {ssbuffer.block_id = 7 : i32, ssbuffer.core_type = "CUBE"} : memref<128x128xbf16>
        %cube_cond2 = arith.constant 1 : i1
        // CHECK: scf.if %{{.*}} {
        // CHECK:   linalg.fill {ssbuffer.block_id = 7 : i32, ssbuffer.core_type = "CUBE"}
        scf.if %cube_cond2 {
          linalg.fill {ssbuffer.block_id = 7 : i32, ssbuffer.core_type = "CUBE"} ins(%cst : bf16) outs(%cube_alloc2 : memref<128x128xbf16>)
        } {hivm.unlikely_condition, ssbuffer.block_id = 7 : i32}
        %cube_tensor2 = bufferization.to_tensor %cube_alloc2 restrict writable {ssbuffer.block_id = 7 : i32, ssbuffer.core_type = "CUBE"} : memref<128x128xbf16> to tensor<128x128xbf16>
        %cube_out2 = tensor.empty() {ssbuffer.block_id = 7 : i32, ssbuffer.core_type = "CUBE"} : tensor<128x128xf32>
        // CHECK: linalg.matmul {input_precision = "ieee", ssbuffer.block_id = 7 : i32, ssbuffer.core_type = "CUBE"}
        // Uses vec_tensor3 as one of the inputs
        %cube_matmul2 = linalg.matmul {input_precision = "ieee", ssbuffer.block_id = 7 : i32, ssbuffer.core_type = "CUBE"} ins(%vec_tensor3, %cube_tensor2 : tensor<128x128xbf16>, tensor<128x128xbf16>) outs(%cube_out2 : tensor<128x128xf32>) -> tensor<128x128xf32>

        // Vector block 2 (successor for both cube blocks)
        // Uses both cube_matmul1 and cube_matmul2
        %vec_alloc2 = memref.alloc() {ssbuffer.block_id = 8 : i32, ssbuffer.core_type = "VECTOR"} : memref<128x128xf32>
        %vec_cond2 = arith.constant 1 : i1
        scf.if %vec_cond2 {
          linalg.fill {ssbuffer.block_id = 8 : i32, ssbuffer.core_type = "VECTOR"} ins(%cst_f32 : f32) outs(%vec_alloc2 : memref<128x128xf32>)
        } {hivm.unlikely_condition, ssbuffer.block_id = 8 : i32}
        %vec_add = arith.addf %cube_matmul1, %cube_matmul2 {ssbuffer.block_id = 8 : i32, ssbuffer.core_type = "VECTOR"} : tensor<128x128xf32>
        %vec_tensor2 = bufferization.to_tensor %vec_alloc2 restrict writable {ssbuffer.block_id = 8 : i32, ssbuffer.core_type = "VECTOR"} : memref<128x128xf32> to tensor<128x128xf32>

        scf.yield
      }
    }

    return
  }

  // ============================================
  // Test Case 3: @test_no_merge_single_layer_for
  // ============================================
  // Scenario: Same as scenario 1, but only one layer of for loop
  // - Vector block 1 (block_id=1) produces vec_tensor1
  // - Cube block 1 (block_id=2) uses vec_tensor1, produces cube_matmul1
  // - Cube block 2 (block_id=3) uses vec_tensor1, produces cube_matmul2
  // - Vector block 2 (block_id=4) uses cube_matmul1 and cube_matmul2
  // Expected: The two cube blocks should NOT be merged (not innermost loop)
  // Only one layer of for loop
  // ============================================
  // CHECK-LABEL: func.func @test_no_merge_single_layer_for
  func.func @test_no_merge_single_layer_for(%arg0: memref<?xbf16>, %arg1: memref<?xbf16>, %arg2: memref<?xf32>) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c128 = arith.constant 128 : index
    %cst = arith.constant 0.000000e+00 : bf16
    %cst_f32 = arith.constant 0.000000e+00 : f32

    // Single layer for loop (not innermost, should NOT be processed)
    scf.for %i = %c0 to %c128 step %c1 {
      // Vector block 1 (predecessor for both cube blocks)
      %vec_alloc1 = memref.alloc() {ssbuffer.block_id = 15 : i32, ssbuffer.core_type = "VECTOR"} : memref<128x128xbf16>
      %vec_cond1 = arith.constant 1 : i1
      scf.if %vec_cond1 {
        linalg.fill {ssbuffer.block_id = 15 : i32, ssbuffer.core_type = "VECTOR"} ins(%cst : bf16) outs(%vec_alloc1 : memref<128x128xbf16>)
      } {hivm.unlikely_condition, ssbuffer.block_id = 15 : i32}
      %vec_tensor1 = bufferization.to_tensor %vec_alloc1 restrict writable {ssbuffer.block_id = 15 : i32, ssbuffer.core_type = "VECTOR"} : memref<128x128xbf16> to tensor<128x128xbf16>

      // Cube block 1 (block_id=2) - should remain unchanged
      // CHECK: memref.alloc() {ssbuffer.block_id = 16 : i32, ssbuffer.core_type = "CUBE"}
      %cube_alloc1 = memref.alloc() {ssbuffer.block_id = 16 : i32, ssbuffer.core_type = "CUBE"} : memref<128x128xbf16>
      %cube_cond1 = arith.constant 1 : i1
      // CHECK: scf.if %{{.*}} {
      // CHECK:   linalg.fill {ssbuffer.block_id = 16 : i32, ssbuffer.core_type = "CUBE"}
      scf.if %cube_cond1 {
        linalg.fill {ssbuffer.block_id = 16 : i32, ssbuffer.core_type = "CUBE"} ins(%cst : bf16) outs(%cube_alloc1 : memref<128x128xbf16>)
      } {hivm.unlikely_condition, ssbuffer.block_id = 16 : i32}
      %cube_tensor1 = bufferization.to_tensor %cube_alloc1 restrict writable {ssbuffer.block_id = 16 : i32, ssbuffer.core_type = "CUBE"} : memref<128x128xbf16> to tensor<128x128xbf16>
      %cube_out1 = tensor.empty() {ssbuffer.block_id = 16 : i32, ssbuffer.core_type = "CUBE"} : tensor<128x128xf32>
      // CHECK: linalg.matmul {input_precision = "ieee", ssbuffer.block_id = 16 : i32, ssbuffer.core_type = "CUBE"}
      %cube_matmul1 = linalg.matmul {input_precision = "ieee", ssbuffer.block_id = 16 : i32, ssbuffer.core_type = "CUBE"} ins(%vec_tensor1, %cube_tensor1 : tensor<128x128xbf16>, tensor<128x128xbf16>) outs(%cube_out1 : tensor<128x128xf32>) -> tensor<128x128xf32>

      // Cube block 2 (block_id=3) - should remain unchanged
      // CHECK: memref.alloc() {ssbuffer.block_id = 17 : i32, ssbuffer.core_type = "CUBE"}
      %cube_alloc2 = memref.alloc() {ssbuffer.block_id = 17 : i32, ssbuffer.core_type = "CUBE"} : memref<128x128xbf16>
      %cube_cond2 = arith.constant 1 : i1
      // CHECK: scf.if %{{.*}} {
      // CHECK:   linalg.fill {ssbuffer.block_id = 17 : i32, ssbuffer.core_type = "CUBE"}
      scf.if %cube_cond2 {
        linalg.fill {ssbuffer.block_id = 17 : i32, ssbuffer.core_type = "CUBE"} ins(%cst : bf16) outs(%cube_alloc2 : memref<128x128xbf16>)
      } {hivm.unlikely_condition, ssbuffer.block_id = 17 : i32}
      %cube_tensor2 = bufferization.to_tensor %cube_alloc2 restrict writable {ssbuffer.block_id = 17 : i32, ssbuffer.core_type = "CUBE"} : memref<128x128xbf16> to tensor<128x128xbf16>
      %cube_out2 = tensor.empty() {ssbuffer.block_id = 17 : i32, ssbuffer.core_type = "CUBE"} : tensor<128x128xf32>
      // CHECK: linalg.matmul {input_precision = "ieee", ssbuffer.block_id = 17 : i32, ssbuffer.core_type = "CUBE"}
      %cube_matmul2 = linalg.matmul {input_precision = "ieee", ssbuffer.block_id = 17 : i32, ssbuffer.core_type = "CUBE"} ins(%vec_tensor1, %cube_tensor2 : tensor<128x128xbf16>, tensor<128x128xbf16>) outs(%cube_out2 : tensor<128x128xf32>) -> tensor<128x128xf32>

      // Vector block 2 (successor for both cube blocks)
      // Uses both cube_matmul1 and cube_matmul2
      %vec_alloc2 = memref.alloc() {ssbuffer.block_id = 18 : i32, ssbuffer.core_type = "VECTOR"} : memref<128x128xf32>
      %vec_cond2 = arith.constant 1 : i1
      scf.if %vec_cond2 {
        linalg.fill {ssbuffer.block_id = 18 : i32, ssbuffer.core_type = "VECTOR"} ins(%cst_f32 : f32) outs(%vec_alloc2 : memref<128x128xf32>)
      } {hivm.unlikely_condition, ssbuffer.block_id = 18 : i32}
      %vec_add = arith.addf %cube_matmul1, %cube_matmul2 {ssbuffer.block_id = 18 : i32, ssbuffer.core_type = "VECTOR"} : tensor<128x128xf32>
      %vec_tensor2 = bufferization.to_tensor %vec_alloc2 restrict writable {ssbuffer.block_id = 18 : i32, ssbuffer.core_type = "VECTOR"} : memref<128x128xf32> to tensor<128x128xf32>


      %cube_alloc3 = memref.alloc() {ssbuffer.block_id = 19 : i32, ssbuffer.core_type = "CUBE"} : memref<128x128xf32>
      %cube_cond3 = arith.constant 1 : i1
      scf.if %cube_cond3 {
        linalg.fill {ssbuffer.block_id = 19 : i32, ssbuffer.core_type = "CUBE"} ins(%cst : bf16) outs(%cube_alloc3 : memref<128x128xf32>)
      } {hivm.unlikely_condition, ssbuffer.block_id = 19 : i32}
      %cube_tensor3 = bufferization.to_tensor %cube_alloc3 restrict writable {ssbuffer.block_id = 19 : i32, ssbuffer.core_type = "CUBE"} : memref<128x128xf32> to tensor<128x128xf32>
      %cube_out3 = tensor.empty() {ssbuffer.block_id = 19 : i32, ssbuffer.core_type = "CUBE"} : tensor<128x128xf32>
      %cube_matmul3 = linalg.matmul {input_precision = "ieee", ssbuffer.block_id = 19 : i32, ssbuffer.core_type = "CUBE"} ins(%vec_tensor2, %cube_tensor3 : tensor<128x128xf32>, tensor<128x128xf32>) outs(%cube_out3 : tensor<128x128xf32>) -> tensor<128x128xf32>


      %vec_add3 = arith.addf %cube_matmul2, %cube_matmul3 {ssbuffer.block_id = 20 : i32, ssbuffer.core_type = "VECTOR"} : tensor<128x128xf32>
      scf.yield
    }

    return
  }
}
