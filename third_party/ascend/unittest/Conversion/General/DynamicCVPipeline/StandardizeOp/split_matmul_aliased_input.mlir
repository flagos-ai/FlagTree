// RUN: triton-opt --ssbuf-standardize-op-pattern-match %s | FileCheck %s

// The RHS and accumulator alias. Splitting must replace only the accumulator.
module {
  // CHECK-LABEL: func.func @split_matmul_with_aliased_rhs_and_accumulator(
  // CHECK-SAME: %[[A:[^:]+]]: tensor<32x32xf32>, %[[B:[^:]+]]: tensor<32x32xf32>
  // CHECK: %[[ZERO:.*]] = linalg.fill
  // CHECK: %[[MATMUL:.*]] = linalg.matmul {{.*}}ins(%[[A]], %[[B]] : tensor<32x32xf32>, tensor<32x32xf32>) outs(%[[ZERO]] : tensor<32x32xf32>)
  // CHECK: %[[RESULT:.*]] = arith.addf %[[MATMUL]], %[[B]] {{.*}}ssbuffer.add_from_matmul
  // CHECK: return %[[RESULT]] : tensor<32x32xf32>
  func.func @split_matmul_with_aliased_rhs_and_accumulator(
      %a: tensor<32x32xf32>, %b: tensor<32x32xf32>)
      -> tensor<32x32xf32> {
    %matmul = linalg.matmul {input_precision = "ieee"}
        ins(%a, %b : tensor<32x32xf32>, tensor<32x32xf32>)
        outs(%b : tensor<32x32xf32>) -> tensor<32x32xf32>
    return %matmul : tensor<32x32xf32>
  }
}
