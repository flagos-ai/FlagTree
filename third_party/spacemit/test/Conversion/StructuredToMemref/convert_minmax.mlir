// RUN: triton-shared-opt --split-input-file --triton-to-linalg-experimental  %s | FileCheck %s
module {
  tt.func public @minmax_olt(%arg0: !tt.ptr<f32>, %arg1: f32, %arg2: f32) {
    %0 = arith.cmpf olt, %arg1, %arg2 : f32
    %1 = arith.select %0, %arg1, %arg2 : f32
    tt.store %arg0, %1 : !tt.ptr<f32>
    tt.return
  }
}

// CHECK-LABEL:  func.func @minmax_olt
// CHECK-SAME:   ([[PARAM_0_:%.+]]: memref<*xf32>, [[PARAM_1_:%.+]]: f32, [[PARAM_2_:%.+]]: f32, [[PARAM_3_:%.+]]: i32, [[PARAM_4_:%.+]]: i32, [[PARAM_5_:%.+]]: i32, [[PARAM_6_:%.+]]: i32, [[PARAM_7_:%.+]]: i32, [[PARAM_8_:%.+]]: i32) {
// CHECK:       [[VAR_0_:%.+]] = arith.minimumf [[PARAM_1_]], [[PARAM_2_]] : f32

// -----

module {
  tt.func public @minmax_ole(%arg0: !tt.ptr<f32>, %arg1: f32, %arg2: f32) {
    %0 = arith.cmpf ole, %arg1, %arg2 : f32
    %1 = arith.select %0, %arg1, %arg2 : f32
    tt.store %arg0, %1 : !tt.ptr<f32>
    tt.return
  }
}

// CHECK-LABEL:  func.func @minmax_ole
// CHECK-SAME:   ([[PARAM_0_:%.+]]: memref<*xf32>, [[PARAM_1_:%.+]]: f32, [[PARAM_2_:%.+]]: f32, [[PARAM_3_:%.+]]: i32, [[PARAM_4_:%.+]]: i32, [[PARAM_5_:%.+]]: i32, [[PARAM_6_:%.+]]: i32, [[PARAM_7_:%.+]]: i32, [[PARAM_8_:%.+]]: i32) {
// CHECK:       [[VAR_0_:%.+]] = arith.minimumf [[PARAM_1_]], [[PARAM_2_]] : f32

// -----

module {
  tt.func public @minmax_ogt(%arg0: !tt.ptr<f32>, %arg1: f32, %arg2: f32) {
    %0 = arith.cmpf ogt, %arg1, %arg2 : f32
    %1 = arith.select %0, %arg1, %arg2 : f32
    tt.store %arg0, %1 : !tt.ptr<f32>
    tt.return
  }
}

// CHECK-LABEL:  func.func @minmax_ogt
// CHECK-SAME:   ([[PARAM_0_:%.+]]: memref<*xf32>, [[PARAM_1_:%.+]]: f32, [[PARAM_2_:%.+]]: f32, [[PARAM_3_:%.+]]: i32, [[PARAM_4_:%.+]]: i32, [[PARAM_5_:%.+]]: i32, [[PARAM_6_:%.+]]: i32, [[PARAM_7_:%.+]]: i32, [[PARAM_8_:%.+]]: i32) {
// CHECK:       [[VAR_0_:%.+]] = arith.maximumf [[PARAM_1_]], [[PARAM_2_]] : f32

// -----

module {
  tt.func public @minmax_oge(%arg0: !tt.ptr<f32>, %arg1: f32, %arg2: f32) {
    %0 = arith.cmpf oge, %arg1, %arg2 : f32
    %1 = arith.select %0, %arg1, %arg2 : f32
    tt.store %arg0, %1 : !tt.ptr<f32>
    tt.return
  }
}

// CHECK-LABEL:  func.func @minmax_oge
// CHECK-SAME:   ([[PARAM_0_:%.+]]: memref<*xf32>, [[PARAM_1_:%.+]]: f32, [[PARAM_2_:%.+]]: f32, [[PARAM_3_:%.+]]: i32, [[PARAM_4_:%.+]]: i32, [[PARAM_5_:%.+]]: i32, [[PARAM_6_:%.+]]: i32, [[PARAM_7_:%.+]]: i32, [[PARAM_8_:%.+]]: i32) {
// CHECK:       [[VAR_0_:%.+]] = arith.maximumf [[PARAM_1_]], [[PARAM_2_]] : f32
