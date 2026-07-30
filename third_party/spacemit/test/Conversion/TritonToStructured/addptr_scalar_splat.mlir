// RUN: triton-shared-opt --triton-to-structured --remove-dead-values --canonicalize %s | FileCheck %s

module {
  tt.func @kernel (%arg0: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %arg1: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %arg2: i32, %arg3: i32, %arg4: i32) {
    %0 = tt.get_program_id x : i32
    %1 = arith.muli %0, %arg2 : i32
    %2 = tt.addptr %arg1, %1 : !tt.ptr<f32>, i32
    // source = %arg1, offset = %1, size = 1, strides = 0
    %3 = tt.make_range {end = 1024 : i32, start = 0 : i32} : tensor<1024xi32>
    // offset = 0, size = 1024, strides = 1
    %4 = tt.splat %2 : !tt.ptr<f32> -> tensor<1024x!tt.ptr<f32>>
    // source = %arg1, offset = %1, size = 1024, strides = 0
    %5 = tt.addptr %4, %3 : tensor<1024x!tt.ptr<f32>>, tensor<1024xi32>
    // source = %arg1, offset = %1, size = 1024, strides = 1
    %8 = tt.load %5 : tensor<1024x!tt.ptr<f32>>
    %17 = math.exp %8 : tensor<1024xf32>
    %18 = arith.muli %0, %arg3 : i32
    %19 = tt.addptr %arg0, %18 : !tt.ptr<f32>, i32
    // source = %arg0, offset = %18, size = 1, strides = 0
    %20 = tt.splat %19 : !tt.ptr<f32> -> tensor<1024x!tt.ptr<f32>>
    // source = %arg0, offset = %18, size = 1024, strides = 0
    %21 = tt.addptr %20, %3 : tensor<1024x!tt.ptr<f32>>, tensor<1024xi32>
    // source = %arg0, offset = %18, size = 1024, strides = 1
    tt.store %21, %17 : tensor<1024x!tt.ptr<f32>>
    tt.return
  }
}

// CHECK:         tt.func @kernel([[PARAM_0_:%.+]]: !tt.ptr<f32> {tt.divisibility = 16 : i32}, [[PARAM_1_:%.+]]: !tt.ptr<f32> {tt.divisibility = 16 : i32}, [[PARAM_2_:%.+]]: i32, [[PARAM_3_:%.+]]: i32, [[PARAM_4_:%.+]]: i32) {
// CHECK:           [[VAR_0_:%.+]] = tt.get_program_id x : i32
// CHECK:           [[VAR_1_:%.+]] = arith.muli [[VAR_0_]], [[PARAM_2_]] : i32
// CHECK:           [[VAR_2_:%.+]] = arith.index_cast [[VAR_1_]] : i32 to index
// CHECK:           [[VAR_3_:%.+]] = tts.make_tptr [[PARAM_1_]] to sizes: [1024], strides: [1], offsets: {{.}}[[VAR_2_]]{{.}}, shape: [0], order: [] : <f32> to tensor<1024x!tt.ptr<f32>>
// CHECK:           [[VAR_4_:%.+]] = "tts.load"([[VAR_3_]]) <{operandSegmentSizes = array<i32: 1, 0, 0>, static_mask_dims = array<i64>}> : (tensor<1024x!tt.ptr<f32>>) -> tensor<1024xf32>
// CHECK-DAG:       [[VAR_5_:%.+]] = math.exp [[VAR_4_]] : tensor<1024xf32>
// CHECK-DAG:       [[VAR_6_:%.+]] = arith.muli [[VAR_0_]], [[PARAM_3_]] : i32
// CHECK:           [[VAR_7_:%.+]] = arith.index_cast [[VAR_6_]] : i32 to index
// CHECK:           [[VAR_8_:%.+]] = tts.make_tptr [[PARAM_0_]] to sizes: [1024], strides: [1], offsets: {{.}}[[VAR_7_]]{{.}}, shape: [0], order: [] : <f32> to tensor<1024x!tt.ptr<f32>>
// CHECK:           "tts.store"([[VAR_8_]], [[VAR_5_]]) <{static_mask_dims = array<i64>}> : (tensor<1024x!tt.ptr<f32>>, tensor<1024xf32>) -> ()
// CHECK:           tt.return
// CHECK:         }
