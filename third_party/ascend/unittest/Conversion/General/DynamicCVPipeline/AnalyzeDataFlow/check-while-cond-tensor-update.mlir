// RUN: triton-opt --analyze-while-condition-args %s | FileCheck %s

// The loop-carried arg %arg3 feeds the scf.condition and is updated in the
// do-region by a scalar produced from a tensor reduction (tl.sum style).
// The dynamic CV pipeline cannot handle this, so the fallback attr must be set.
module attributes {hacc.target = #hacc.target<"Ascend950PR_9579">} {
  func.func @while_cond_updated_by_tensor_sum(%arg0: memref<?xf32>, %arg1: i64) {
    %c0_i64 = arith.constant 0 : i64
    %cst = arith.constant 0.000000e+00 : f32
    %0 = tensor.empty() : tensor<64xf32>
    %1 = linalg.fill ins(%cst : f32) outs(%0 : tensor<64xf32>) -> tensor<64xf32>
    %2:2 = scf.while (%arg2 = %c0_i64, %arg3 = %c0_i64) : (i64, i64) -> (i64, i64) {
      %3 = arith.cmpi slt, %arg3, %arg1 : i64
      scf.condition(%3) %arg2, %arg3 : i64, i64
    } do {
    ^bb0(%arg2: i64, %arg3: i64):
      %3 = tensor.empty() : tensor<f32>
      %4 = linalg.fill ins(%cst : f32) outs(%3 : tensor<f32>) -> tensor<f32>
      %reduced = linalg.reduce ins(%1 : tensor<64xf32>) outs(%4 : tensor<f32>) dimensions = [0]
        (%in: f32, %init: f32) {
          %7 = arith.addf %in, %init : f32
          linalg.yield %7 : f32
        }
      %5 = tensor.extract %reduced[] : tensor<f32>
      %6 = arith.fptosi %5 : f32 to i64
      %7 = arith.addi %arg3, %6 : i64
      scf.yield %arg2, %7 : i64, i64
    } attributes {ssbuffer.main_loop = 1 : i64}
    return
  }
}

// CHECK: triton_ascend.dynamic_cv_pipeline.rc
