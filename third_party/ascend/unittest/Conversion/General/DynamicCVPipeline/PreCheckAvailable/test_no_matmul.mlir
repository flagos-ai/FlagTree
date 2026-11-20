// RUN: triton-opt --pre-check-matmul --debug %s 2>&1 | FileCheck %s

// Test Case: Module without linalg.matmul operation
// PreCheckMatmul should fail because no matmul is found

// CHECK: SSBUFFER will be skipped because no linalg.matmul operation was found
module attributes {hacc.target = #hacc.target<"Ascend950PR_9579">} {
  func.func @test_no_matmul() {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c32 = arith.constant 32 : index
    %cst_f32 = arith.constant 0.0 : f32

    %empty = tensor.empty() : tensor<32x32xf32>
    %filled = linalg.fill ins(%cst_f32 : f32) outs(%empty : tensor<32x32xf32>) -> tensor<32x32xf32>

    %cst_1 = arith.constant 1.0 : f32

    scf.for %arg0 = %c0 to %c32 step %c1 {
    }

    return
  }
}
