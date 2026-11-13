// RUN: triton-opt --pre-check-blacklist --debug %s 2>&1 | FileCheck %s

// Test Case: Module with scope.scope operation (blacklist op)
// Should be detected by PreCheckBlacklistPass

//CHECK: SSBUFFER will be skipped because scope.scope operation was found
module attributes {hacc.target = #hacc.target<"Ascend950PR_9579">} {
  func.func @test_scope_blacklist() {
    %c0 = arith.constant 0 : i32

    scope.scope : () -> () {
      %cst = arith.constant 0.0 : f32
      %empty = tensor.empty() : tensor<32x32xf32>
      %filled = linalg.fill ins(%cst : f32) outs(%empty : tensor<32x32xf32>) -> tensor<32x32xf32>
    }

    return
  }
}
