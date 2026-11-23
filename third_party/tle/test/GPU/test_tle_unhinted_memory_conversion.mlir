// RUN: triton-opt %s -convert-triton-to-tritongpu='target=cuda:90 num-warps=4' | FileCheck %s
// RUN: triton-opt %s -convert-triton-to-tritongpu='target=hip:gfx942 num-warps=4' | FileCheck %s

// Explicit memory patterns must coexist with ordinary access legalization.
// A layout hint on another function must not be required to convert this copy.
module {
  // CHECK-LABEL: tt.func @ordinary_masked_copy
  tt.func @ordinary_masked_copy(%input: !tt.ptr<f32>, %output: !tt.ptr<f32>) {
    %index = tt.make_range {start = 0 : i32, end = 128 : i32} : tensor<128xi32>
    %limit = arith.constant dense<73> : tensor<128xi32>
    %mask = arith.cmpi slt, %index, %limit : tensor<128xi32>
    %zero = arith.constant dense<0.0> : tensor<128xf32>
    %input_base = tt.splat %input : !tt.ptr<f32> -> tensor<128x!tt.ptr<f32>>
    %input_ptr = tt.addptr %input_base, %index : tensor<128x!tt.ptr<f32>>, tensor<128xi32>
    // CHECK: %[[VALUE:.*]] = tt.load {{.*}} : tensor<128x!tt.ptr<f32>, #blocked>
    %value = tt.load %input_ptr, %mask, %zero : tensor<128x!tt.ptr<f32>>
    %output_base = tt.splat %output : !tt.ptr<f32> -> tensor<128x!tt.ptr<f32>>
    %output_ptr = tt.addptr %output_base, %index : tensor<128x!tt.ptr<f32>>, tensor<128xi32>
    // CHECK: tt.store {{.*}}, %[[VALUE]], {{.*}} : tensor<128x!tt.ptr<f32>, #blocked>
    tt.store %output_ptr, %value, %mask : tensor<128x!tt.ptr<f32>>
    tt.return
  }
}
