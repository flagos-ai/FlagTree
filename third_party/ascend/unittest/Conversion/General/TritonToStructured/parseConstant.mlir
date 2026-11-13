// RUN: triton-opt  '--triton-to-structured=enable-mask-fallback-conversion=false optimize-dynamic-offset=true'  --split-input-file %s | FileCheck %s

tt.func public @test_non_splat_mask(%arg0: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %arg1: !tt.ptr<f32> {tt.divisibility = 16 : i32}) {
  %non_splat_mask = arith.constant dense<[false, true]> : tensor<2xi1>
  %c0_f32 = arith.constant dense<0.000000e+00> : tensor<2xf32>
  %0 = tt.make_range {end = 2 : i32, start = 0 : i32} : tensor<2xi32>
  %1 = tt.splat %arg0 : !tt.ptr<f32> -> tensor<2x!tt.ptr<f32>>
  %ptr_load = tt.addptr %1, %0 : tensor<2x!tt.ptr<f32>>, tensor<2xi32>
  %2 = tt.load %ptr_load, %non_splat_mask, %c0_f32 : tensor<2x!tt.ptr<f32>>
  %3 = tt.splat %arg1 : !tt.ptr<f32> -> tensor<2x!tt.ptr<f32>>
  %ptr_store = tt.addptr %3, %0 : tensor<2x!tt.ptr<f32>>, tensor<2xi32>
  tt.store %ptr_store, %2 : tensor<2x!tt.ptr<f32>>
  tt.return
}

// CHECK-LABEL:   tt.func public @test_non_splat_mask(
// CHECK-SAME:                                %[[VAL_0:.*]]: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %[[VAL_1:.*]]: !tt.ptr<f32> {tt.divisibility = 16 : i32}) {
// CHECK:           %[[VAL_2:.*]] = arith.constant dense<[false, true]> : tensor<2xi1>
// CHECK:           %[[VAL_3:.*]] = arith.constant dense<0.000000e+00> : tensor<2xf32>
// CHECK:           %[[VAL_4:.*]] = tt.make_range {end = 2 : i32, start = 0 : i32} : tensor<2xi32>
// CHECK:           %[[VAL_5:.*]] = tt.splat %[[VAL_0]] : !tt.ptr<f32> -> tensor<2x!tt.ptr<f32>>
// CHECK:           %[[VAL_6:.*]] = tt.addptr %[[VAL_5]], %[[VAL_4]] : tensor<2x!tt.ptr<f32>>, tensor<2xi32>
// CHECK:           %[[VAL_7:.*]] = tt.load %[[VAL_6]], %[[VAL_2]], %[[VAL_3]] : tensor<2x!tt.ptr<f32>>
// CHECK:           %[[VAL_8:.*]] = tt.splat %[[VAL_1]] : !tt.ptr<f32> -> tensor<2x!tt.ptr<f32>>
// CHECK:           %[[VAL_9:.*]] = tt.addptr %[[VAL_8]], %[[VAL_4]] : tensor<2x!tt.ptr<f32>>, tensor<2xi32>
// CHECK:           tt.store %[[VAL_9]], %[[VAL_7]] : tensor<2x!tt.ptr<f32>>
// CHECK:           tt.return
// CHECK:         }
