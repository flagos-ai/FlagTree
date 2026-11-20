// RUN: triton-opt --triton-to-hfusion --triton-to-linalg %s | FileCheck %s

// CHECK-LABEL: func.func @scalar_fp32_to_fp8_rtne
// CHECK-NOT: tt.fp_to_fp
// CHECK: %[[CAST:.*]] = arith.truncf %{{.*}} {round_mode = #hfusion.round_mode<rint>} : f32 to f8E4M3FN
// CHECK-NOT: tt.fp_to_fp
// CHECK: return %[[CAST]] : f8E4M3FN
// CHECK-LABEL: func.func @scalar_fp8_to_fp32_rtne
// CHECK-NOT: tt.fp_to_fp
// CHECK: %[[FP8_TO_F32:.*]] = arith.extf %{{.*}} {round_mode = #hfusion.round_mode<rint>} : f8E4M3FN to f32
// CHECK-NOT: tt.fp_to_fp
// CHECK: return %[[FP8_TO_F32]] : f32
// CHECK-LABEL: func.func @scalar_f8e4m3fn_to_f8e5m2_rtne
// CHECK-NOT: tt.fp_to_fp
// CHECK: %[[E4_TO_F32:.*]] = arith.extf %{{.*}} {round_mode = #hfusion.round_mode<rint>} : f8E4M3FN to f32
// CHECK-NOT: tt.fp_to_fp
// CHECK: %[[F32_TO_E5:.*]] = arith.truncf %[[E4_TO_F32]] {round_mode = #hfusion.round_mode<rint>} : f32 to f8E5M2
// CHECK-NOT: tt.fp_to_fp
// CHECK: return %[[F32_TO_E5]] : f8E5M2
// CHECK-LABEL: func.func @scalar_f8e5m2_to_f8e4m3fn_default
// CHECK-NOT: tt.fp_to_fp
// CHECK: %[[E5_TO_F32:.*]] = arith.extf %{{.*}} {round_mode = #hfusion.round_mode<rint>} : f8E5M2 to f32
// CHECK-NOT: tt.fp_to_fp
// CHECK: %[[F32_TO_E4:.*]] = arith.truncf %[[E5_TO_F32]] {round_mode = #hfusion.round_mode<rint>} : f32 to f8E4M3FN
// CHECK-NOT: tt.fp_to_fp
// CHECK: return %[[F32_TO_E4]] : f8E4M3FN
// CHECK-LABEL: func.func @scalar_fp16_to_bf16_rtne
// CHECK-NOT: tt.fp_to_fp
// CHECK: %[[F16_TO_F32:.*]] = arith.extf %{{.*}} {round_mode = #hfusion.round_mode<rint>} : f16 to f32
// CHECK-NOT: tt.fp_to_fp
// CHECK: %[[F32_TO_BF16:.*]] = arith.truncf %[[F16_TO_F32]] {round_mode = #hfusion.round_mode<rint>} : f32 to bf16
// CHECK-NOT: tt.fp_to_fp
// CHECK: return %[[F32_TO_BF16]] : bf16
// CHECK-LABEL: func.func @tensor_f8e4m3fn_to_f8e5m2_rtne
// CHECK-NOT: tt.fp_to_fp
// CHECK: arith.extf %{{.*}} {round_mode = #hfusion.round_mode<rint>} : tensor<8xf8E4M3FN> to tensor<8xf32>
// CHECK-NOT: tt.fp_to_fp
// CHECK: arith.truncf %{{.*}} {round_mode = #hfusion.round_mode<rint>} : tensor<8xf32> to tensor<8xf8E5M2>
// CHECK-NOT: tt.fp_to_fp
// CHECK: return
// CHECK-LABEL: func.func @scalar_f8e4m3fn_identity
// CHECK-NOT: tt.fp_to_fp
// CHECK: return %{{.*}} : f8E4M3FN
module {
  tt.func @scalar_fp32_to_fp8_rtne(%arg0: f32) -> f8E4M3FN {
    %0 = tt.fp_to_fp %arg0, rounding = rtne : f32 -> f8E4M3FN
    tt.return %0 : f8E4M3FN
  }

  tt.func @scalar_fp8_to_fp32_rtne(%arg0: f8E4M3FN) -> f32 {
    %0 = tt.fp_to_fp %arg0, rounding = rtne : f8E4M3FN -> f32
    tt.return %0 : f32
  }

  tt.func @scalar_f8e4m3fn_to_f8e5m2_rtne(%arg0: f8E4M3FN) -> f8E5M2 {
    %0 = tt.fp_to_fp %arg0, rounding = rtne : f8E4M3FN -> f8E5M2
    tt.return %0 : f8E5M2
  }

  tt.func @scalar_f8e5m2_to_f8e4m3fn_default(%arg0: f8E5M2) -> f8E4M3FN {
    %0 = tt.fp_to_fp %arg0 : f8E5M2 -> f8E4M3FN
    tt.return %0 : f8E4M3FN
  }

  tt.func @scalar_fp16_to_bf16_rtne(%arg0: f16) -> bf16 {
    %0 = tt.fp_to_fp %arg0, rounding = rtne : f16 -> bf16
    tt.return %0 : bf16
  }

  tt.func @tensor_f8e4m3fn_to_f8e5m2_rtne(%arg0: f8E4M3FN, %arg1: !tt.ptr<f8E5M2>) {
    %input = tt.splat %arg0 : f8E4M3FN -> tensor<8xf8E4M3FN>
    %0 = tt.fp_to_fp %input, rounding = rtne : tensor<8xf8E4M3FN> -> tensor<8xf8E5M2>
    %offsets = tt.make_range {end = 8 : i32, start = 0 : i32} : tensor<8xi32>
    %output_ptrs = tt.splat %arg1 : !tt.ptr<f8E5M2> -> tensor<8x!tt.ptr<f8E5M2>>
    %output_addrs = tt.addptr %output_ptrs, %offsets : tensor<8x!tt.ptr<f8E5M2>>, tensor<8xi32>
    tt.store %output_addrs, %0 : tensor<8x!tt.ptr<f8E5M2>>
    tt.return
  }

  tt.func @scalar_f8e4m3fn_identity(%arg0: f8E4M3FN) -> f8E4M3FN {
    %0 = tt.fp_to_fp %arg0 : f8E4M3FN -> f8E4M3FN
    tt.return %0 : f8E4M3FN
  }
}
