// RUN: triton-opt --analyze-while-condition-args %s | FileCheck %s

// The tensor reduction feeding the loop-carried arg %arg3 is computed outside
// the scf.while; the do-region only extracts a scalar out of it. The non-scalar
// computation is not part of the scope that has to be cloned, so the fallback
// attr must NOT be set.
module attributes {hacc.target = #hacc.target<"Ascend950PR_9579">} {
  func.func @while_cond_updated_by_outside_tensor(%arg0: memref<?xf32>, %arg1: i64) {
    %c0_i64 = arith.constant 0 : i64
    %cst = arith.constant 0.000000e+00 : f32
    %0 = tensor.empty() : tensor<64xf32>
    %1 = linalg.fill ins(%cst : f32) outs(%0 : tensor<64xf32>) -> tensor<64xf32>
    %2 = tensor.empty() : tensor<f32>
    %3 = linalg.fill ins(%cst : f32) outs(%2 : tensor<f32>) -> tensor<f32>
    %reduced = linalg.reduce ins(%1 : tensor<64xf32>) outs(%3 : tensor<f32>) dimensions = [0]
      (%in: f32, %init: f32) {
        %4 = arith.addf %in, %init : f32
        linalg.yield %4 : f32
      }
    %5:2 = scf.while (%arg2 = %c0_i64, %arg3 = %c0_i64) : (i64, i64) -> (i64, i64) {
      %6 = arith.cmpi slt, %arg3, %arg1 : i64
      scf.condition(%6) %arg2, %arg3 : i64, i64
    } do {
    ^bb0(%arg2: i64, %arg3: i64):
      %6 = tensor.extract %reduced[] : tensor<f32>
      %7 = arith.fptosi %6 : f32 to i64
      %8 = arith.addi %arg3, %7 : i64
      scf.yield %arg2, %8 : i64, i64
    } attributes {ssbuffer.main_loop = 1 : i64}
    return
  }
}

// CHECK-NOT: triton_ascend.dynamic_cv_pipeline.rc
// CHECK: return
