// RUN: triton-adapter-opt --triton-to-linalg %s | FileCheck %s

module {
  tt.func public @triton_unk_fused_pow_0(%arg0: !tt.ptr<f16> {tt.divisibility = 16 : i32}, %arg1: !tt.ptr<f16> {tt.divisibility = 16 : i32}) attributes {noinline = false} {
    %c0_i32 = arith.constant 0 : i32
    %0 = tt.addptr %arg0, %c0_i32 : !tt.ptr<f16>, i32
    %1 = tt.load %0 : !tt.ptr<f16>
    %2 = arith.extf %1 : f16 to f32
    // CHECK: %[[TENSOR:.*]] = tensor.from_elements %[[INPUT:.*]] : tensor<1xf32>
    // CHECK: %[[MAPPED:.*]] = linalg.map { func.call {callee = @__hmf_sqrtf} } ins(%[[TENSOR]] : tensor<1xf32>) outs(%[[TENSOR]] : tensor<1xf32>)
    // CHECK: %[[VAR_EXTRACETED:.+]] = tensor.extract %[[MAPPED]][%[[C0:.+]]] : tensor<1xf32>
    %3 = tt.extern_elementwise %2 {libname = "", libpath = "", pure = true, symbol = "__hmf_sqrtf"} : (f32) -> f32
    %4 = tt.addptr %arg1, %c0_i32 : !tt.ptr<f16>, i32
    %5 = tt.splat %4 : !tt.ptr<f16> -> tensor<1x!tt.ptr<f16>>
    %6 = arith.truncf %3 : f32 to f16
    %7 = tt.splat %6 : f16 -> tensor<1xf16>
    tt.store %5, %7 : tensor<1x!tt.ptr<f16>>
    tt.return
  }
}
