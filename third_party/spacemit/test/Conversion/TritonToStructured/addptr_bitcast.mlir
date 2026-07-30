// RUN: triton-shared-opt --triton-to-structured %s | FileCheck %s

module {
  tt.func @test(%arg0: !tt.ptr<i64>, %arg1: !tt.ptr<f32>) {
    %0 = tt.load %arg0 : !tt.ptr<i64>
    %1 = tt.int_to_ptr %0 : i64 -> !tt.ptr<f16>
    %2 = tt.make_range {end = 32 : i32, start = 0 : i32} : tensor<32xi32>
    %3 = tt.splat %1 : !tt.ptr<f16> -> tensor<32x!tt.ptr<f16>>
    %4 = tt.addptr %3, %2 : tensor<32x!tt.ptr<f16>>, tensor<32xi32>
    %5 = tt.load %4 : tensor<32x!tt.ptr<f16>>
    %6 = tt.bitcast %arg1 : !tt.ptr<f32> -> !tt.ptr<f16>
    %7 = tt.splat %6 : !tt.ptr<f16> -> tensor<32x!tt.ptr<f16>>
    %8 = tt.addptr %7, %2 : tensor<32x!tt.ptr<f16>>, tensor<32xi32>
    tt.store %8, %5 : tensor<32x!tt.ptr<f16>>
    tt.return
  }
}

// CHECK: [[IN_SRC:%.+]] = tt.int_to_ptr
// CHECK: [[IN_PTR:%.+]] = tts.make_tptr [[IN_SRC]]
// CHECK: [[OUT_SRC:%.+]] = tt.bitcast
// CHECK: [[OUT_PTR:%.+]] = tts.make_tptr [[OUT_SRC]]
