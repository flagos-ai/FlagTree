// RUN: triton-opt  '--triton-to-structured=enable-mask-fallback-conversion=false optimize-dynamic-offset=true'  --split-input-file %s | FileCheck %s

module {
  tt.func public @test_stride_not_one(%arg0: !tt.ptr<f32> {tt.divisibility = 16 : i32}) {
    %c0_f32 = arith.constant dense<0.000000e+00> : tensor<4xf32>
    %fake_range_mask = arith.constant dense<[false, false, false, false]> : tensor<4xi1>
    %0 = tt.make_range {end = 4 : i32, start = 0 : i32} : tensor<4xi32>
    %1 = tt.splat %arg0 : !tt.ptr<f32> -> tensor<4x!tt.ptr<f32>>
    %ptr = tt.addptr %1, %0 : tensor<4x!tt.ptr<f32>>, tensor<4xi32>
    %2 = tt.load %ptr, %fake_range_mask, %c0_f32 : tensor<4x!tt.ptr<f32>>
    tt.store %ptr, %2 : tensor<4x!tt.ptr<f32>>
    tt.return
  }
}

// CHECK-LABEL: tt.func public @test_stride_not_one(
// CHECK-SAME:                                %[[VAL_0:.*]]: !tt.ptr<f32> {tt.divisibility = 16 : i32}) {
// CHECK:           %[[VAL_1:.*]] = arith.constant dense<0.000000e+00> : tensor<4xf32>
// CHECK:           %[[VAL_2:.*]] = tt.make_range {end = 4 : i32, start = 0 : i32} : tensor<4xi32>
// CHECK:           %[[VAL_3:.*]] = tt.splat %[[VAL_0]] : !tt.ptr<f32> -> tensor<4x!tt.ptr<f32>>
// CHECK:           %[[VAL_4:.*]] = tt.addptr %[[VAL_3]], %[[VAL_2]] : tensor<4x!tt.ptr<f32>>, tensor<4xi32>
// CHECK:           tt.store %[[VAL_4]], %[[VAL_1]] : tensor<4x!tt.ptr<f32>>
// CHECK:           tt.return
// CHECK:         }
