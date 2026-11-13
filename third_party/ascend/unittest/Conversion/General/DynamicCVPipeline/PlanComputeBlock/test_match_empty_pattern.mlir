// RUN: triton-opt --op-classifier %s | FileCheck %s

// Test tensor.empty pattern matching: empty outside loop, used as iter_arg, matmul uses iter_arg as output

module {
  // CHECK-LABEL: func.func @test_empty_loop_iter_arg
  // CHECK: tensor.empty() {ssbuffer.core_type = "CUBE"}
  // CHECK: scf.for
  // CHECK: linalg.matmul {ssbuffer.core_type = "CUBE"}
  func.func @test_empty_loop_iter_arg(%a: tensor<64x64xf16>, %b: tensor<64x64xf16>, %n: index) -> tensor<64x64xf32> {
    %empty = tensor.empty() : tensor<64x64xf32>
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %result = scf.for %i = %c0 to %n step %c1 iter_args(%acc = %empty) -> (tensor<64x64xf32>) {
      %mm = linalg.matmul ins(%a, %b : tensor<64x64xf16>, tensor<64x64xf16>) outs(%acc : tensor<64x64xf32>) -> tensor<64x64xf32>
      scf.yield %mm : tensor<64x64xf32>
    }
    return %result : tensor<64x64xf32>
  }
}
