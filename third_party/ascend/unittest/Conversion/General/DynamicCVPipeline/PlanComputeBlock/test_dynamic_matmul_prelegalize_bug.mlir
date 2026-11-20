// RUN: triton-opt --plan-compute-block %s | FileCheck %s

// This documents a current PlanComputeBlock failure:
// preLegalizeMatmul builds tensor.dim from the original matmul result before
// that result dominates the new tensor.dim users.

module {
  // CHECK-LABEL: func.func @dynamic_matmul_prelegalize_uses_result_before_def(
  // CHECK: linalg.matmul {{.*}}ssbuffer.core_type = "CUBE"
  // CHECK: return
  func.func @dynamic_matmul_prelegalize_uses_result_before_def(
      %a: tensor<?x?xf16>,
      %b: tensor<?x?xf16>,
      %out: tensor<?x?xf32>) -> tensor<?x?xf32> {
    %mm = linalg.matmul
      ins(%a, %b : tensor<?x?xf16>, tensor<?x?xf16>)
      outs(%out : tensor<?x?xf32>) -> tensor<?x?xf32>
    return %mm : tensor<?x?xf32>
  }
}
