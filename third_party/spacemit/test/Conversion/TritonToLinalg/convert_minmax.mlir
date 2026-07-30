// RUN: triton-shared-opt --triton-to-linalg --split-input-file %s | FileCheck %s
module {
  tt.func public @minmax_olt(%arg0: !tt.ptr<f32>, %arg1: f32, %arg2: f32) {
    %0 = arith.cmpf olt, %arg1, %arg2 : f32
    %1 = arith.select %0, %arg1, %arg2 : f32
    tt.store %arg0, %1 : !tt.ptr<f32>
    tt.return
  }
}
// CHECK:  func.func @minmax_olt
// CHECK:  %[[VAL:.*]] = arith.minimumf %arg1, %arg2 : f32

// -----

module {
  tt.func public @minmax_ole(%arg0: !tt.ptr<f32>, %arg1: f32, %arg2: f32) {
    %0 = arith.cmpf ole, %arg1, %arg2 : f32
    %1 = arith.select %0, %arg1, %arg2 : f32
    tt.store %arg0, %1 : !tt.ptr<f32>
    tt.return
  }
}
// CHECK:  func.func @minmax_ole
// CHECK:  %[[VAL:.*]] = arith.minimumf %arg1, %arg2 : f32

// -----

module {
  tt.func public @minmax_ogt(%arg0: !tt.ptr<f32>, %arg1: f32, %arg2: f32) {
    %0 = arith.cmpf ogt, %arg1, %arg2 : f32
    %1 = arith.select %0, %arg1, %arg2 : f32
    tt.store %arg0, %1 : !tt.ptr<f32>
    tt.return
  }
}
// CHECK:  func.func @minmax_ogt
// CHECK:  %[[VAL:.*]] = arith.maximumf %arg1, %arg2 : f32

// -----

module {
  tt.func public @minmax_oge(%arg0: !tt.ptr<f32>, %arg1: f32, %arg2: f32) {
    %0 = arith.cmpf oge, %arg1, %arg2 : f32
    %1 = arith.select %0, %arg1, %arg2 : f32
    tt.store %arg0, %1 : !tt.ptr<f32>
    tt.return
  }
}
// CHECK:  func.func @minmax_oge
// CHECK:  %[[VAL:.*]] = arith.maximumf %arg1, %arg2 : f32
