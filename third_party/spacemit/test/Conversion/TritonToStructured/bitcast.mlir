// RUN: triton-shared-opt --triton-to-structured --remove-dead-values --canonicalize %s | FileCheck %s

module {
  tt.func @kernel(%a : !tt.ptr<i32>, %b : !tt.ptr<f32>) -> () {
    // offset calculations
    %0 = tt.make_range {end = 1024 : i32, start = 0 : i32} : tensor<1024xi32>

    // a pointer
    %8 = tt.splat %a : !tt.ptr<i32> -> tensor<1024x!tt.ptr<i32>>
    %9 = tt.addptr %8, %0 : tensor<1024x!tt.ptr<i32>>, tensor<1024xi32>

    // b pointer
    %18 = tt.splat %b : !tt.ptr<f32> -> tensor<1024x!tt.ptr<f32>>
    %19 = tt.addptr %18, %0 : tensor<1024x!tt.ptr<f32>>, tensor<1024xi32>

    %am = tt.load %9 : tensor<1024x!tt.ptr<i32>>

    // cast result before doing float add
    %am_bitcast = tt.bitcast %am : tensor<1024xi32> -> tensor<1024xf32>


    tt.store %19, %am_bitcast : tensor<1024x!tt.ptr<f32>>
    tt.return
  }
}

// CHECK:         tt.func @kernel([[PARAM_0_:%.+]]: !tt.ptr<i32>, [[PARAM_1_:%.+]]: !tt.ptr<f32>) {
// CHECK-DAG:       [[VAR_0_:%.+]] = tts.make_tptr [[PARAM_0_]] to sizes: [1024], strides: [1], offsets: [0], shape: [0], order: [] : <i32> to tensor<1024x!tt.ptr<i32>>
// CHECK-DAG:       [[VAR_1_:%.+]] = tts.make_tptr [[PARAM_1_]] to sizes: [1024], strides: [1], offsets: [0], shape: [0], order: [] : <f32> to tensor<1024x!tt.ptr<f32>>
// CHECK:           [[VAR_2_:%.+]] = "tts.load"([[VAR_0_]]) <{operandSegmentSizes = array<i32: 1, 0, 0>, static_mask_dims = array<i64>}> : (tensor<1024x!tt.ptr<i32>>) -> tensor<1024xi32>
// CHECK:           [[VAR_3_:%.+]] = tt.bitcast [[VAR_2_]] : tensor<1024xi32> -> tensor<1024xf32>
// CHECK:           "tts.store"([[VAR_1_]], [[VAR_3_]]) <{static_mask_dims = array<i64>}> : (tensor<1024x!tt.ptr<f32>>, tensor<1024xf32>) -> ()
// CHECK:           tt.return
// CHECK:         }
