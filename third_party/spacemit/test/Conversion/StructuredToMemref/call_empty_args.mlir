// RUN: triton-shared-opt --triton-to-linalg-experimental %s | FileCheck %s

module {
  tt.func @_sum_combine__fp32() -> f32{
    %0 = arith.constant 42.0 : f32
    tt.return %0 : f32
  }
  tt.func @test() -> f32{
    %0 = tt.call @_sum_combine__fp32() : () -> f32
    tt.return %0 : f32
  }
}

// CHECK: module {
// CHECK:   func.func @_sum_combine__fp32(%arg0: i32, %arg1: i32, %arg2: i32, %arg3: i32, %arg4: i32, %arg5: i32) -> f32 {
// CHECK:     [[CST_:%.+]] = arith.constant 4.200000e+01 : f32
// CHECK:     return [[CST_]] : f32
// CHECK:   }
// CHECK:   func.func @test(%arg0: i32, %arg1: i32, %arg2: i32, %arg3: i32, %arg4: i32, %arg5: i32) -> f32 {
// CHECK:     [[VAR_0_:%.+]] = call @_sum_combine__fp32(%arg0, %arg1, %arg2, %arg3, %arg4, %arg5) : (i32, i32, i32, i32, i32, i32) -> f32
// CHECK:     return [[VAR_0_]] : f32
// CHECK:   }
// CHECK: }
