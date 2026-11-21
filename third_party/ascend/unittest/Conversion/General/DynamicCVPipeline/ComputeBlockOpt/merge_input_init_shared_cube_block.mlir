// RUN: triton-opt --merge-input-init-shared-cube-block %s | FileCheck %s

module {
  // ============================================
  // Test Case: @test_merge_shared_input_init
  // ============================================
  // Scenario: matmulA -> matmulB(input, init) with no extra dependency path.
  // Merging matmulB's block into matmulA's block does not create a cycle, so the
  // two matmuls end up in the same block.
  // ============================================
  // CHECK-LABEL: func.func @test_merge_shared_input_init
  func.func @test_merge_shared_input_init(%a: tensor<128x128xf32>, %b: tensor<128x128xf32>) -> tensor<128x128xf32> {
    %init_a = tensor.empty() : tensor<128x128xf32>
    // CHECK: linalg.matmul {{.*}}ssbuffer.block_id = 12 : i32, ssbuffer.core_type = "CUBE"
    %mat0 = linalg.matmul {input_precision = "ieee", ssbuffer.block_id = 12 : i32, ssbuffer.core_type = "CUBE"} ins(%a, %b : tensor<128x128xf32>, tensor<128x128xf32>) outs(%init_a : tensor<128x128xf32>) -> tensor<128x128xf32>

    // CHECK: linalg.matmul {{.*}}ssbuffer.block_id = 12 : i32, ssbuffer.core_type = "CUBE"
    // CHECK-NOT: ssbuffer.add_from_matmul
    %mat1 = linalg.matmul {input_precision = "ieee", ssbuffer.block_id = 13 : i32, ssbuffer.core_type = "CUBE"} ins(%mat0, %b : tensor<128x128xf32>, tensor<128x128xf32>) outs(%mat0 : tensor<128x128xf32>) -> tensor<128x128xf32>
    return %mat1 : tensor<128x128xf32>
  }
}
