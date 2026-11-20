// RUN: triton-opt %s -triton-to-hfusion | FileCheck %s

// CHECK-LABEL: tt.func @test_fp32_to_fp16_rtz
tt.func @test_fp32_to_fp16_rtz(%arg0: tensor<1024xf32>) -> tensor<1024xf16> {
  // CHECK: %[[EMPTY:.*]] = tensor.empty() : tensor<1024xf16>
  // CHECK: %[[RESULT:.*]] = hfusion.cast {mode = #hfusion.round_mode<trunc>} ins(%arg0 : tensor<1024xf32>) outs(%[[EMPTY]] : tensor<1024xf16>) -> tensor<1024xf16>
  %0 = tt.fp_to_fp %arg0, rounding = rtz : tensor<1024xf32> -> tensor<1024xf16>
  // CHECK: return %[[RESULT]]
  tt.return %0 : tensor<1024xf16>
}


// CHECK-LABEL: tt.func @test_fp32_to_fp16_rtz_fail
tt.func @test_fp32_to_fp16_rtz_fail(%arg0: tensor<1024xf32>) -> tensor<1024xf16> {
  %0 = tt.fp_to_fp %arg0, rounding = rtne : tensor<1024xf32> -> tensor<1024xf16>
  // CHECK: %{{.*}} = tt.fp_to_fp %arg0, rounding = rtne : tensor<1024xf32> -> tensor<1024xf16>
  tt.return %0 : tensor<1024xf16>
}
