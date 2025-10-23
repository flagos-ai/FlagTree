// RUN: triton-adapter-opt --triton-to-linalg %s | FileCheck %s
module {
  // CHECK: func.func private @triton_print_0(i32) attributes {hex = false, prefix = " pid =: "}
  // CHECK-NEXT: func.func private @triton_print_1(tensor<1024xf32>) attributes {hex = true, prefix = " Val =: "}
  // CHECK: func.func @test_print
  tt.func public @test_print(%arg0: i32, %arg1: !tt.ptr<f32>) {
    %0 = tt.splat %arg1 : !tt.ptr<f32> -> tensor<1024x!tt.ptr<f32>>
    %1 = tt.make_range {end = 1024 : i32, start = 0 : i32} : tensor<1024xi32>
    %2 = tt.addptr %0, %1 : tensor<1024x!tt.ptr<f32>>, tensor<1024xi32>
    %3 = tt.load %2 : tensor<1024x!tt.ptr<f32>>
    // CHECK: call @triton_print_0
    tt.print " pid =: " {hex = false,  isSigned = array<i32: 0>} : %arg0 : i32
    %4 = arith.addf %3, %3 : tensor<1024xf32>
    // CHECK: call @triton_print_1
    tt.print " Val =: " {hex = true, isSigned = array<i32: 0>} : %3 : tensor<1024xf32>
    tt.return
  }
}