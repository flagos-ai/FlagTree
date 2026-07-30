// RUN: triton-shared-opt --triton-to-structured --remove-dead-values --canonicalize %s | FileCheck %s

module {
  tt.func @kernel (%arg0: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %arg1: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %arg2: i32, %arg3: i32, %arg4: i32) {
    %0 = tt.get_program_id x : i32
    %1 = arith.muli %0, %arg2 : i32
    %2 = tt.addptr %arg1, %1 : !tt.ptr<f32>, i32
    // source = arg1, offset = %1, size = 1, strides = 0
    %3 = arith.muli %0, %arg3 : i32
    %4 = tt.addptr %2, %3 : !tt.ptr<f32>, i32
    // source = arg1, offset = %1+%3, size = 1, strides = 0
    %5 = arith.muli %0, %arg4 : i32
    %6 = tt.addptr %4, %5 : !tt.ptr<f32>, i32
    // source = arg1, offset = %1+%3+%5, size = 1, strides = 0
    %7 = tt.make_range {end = 1024 : i32, start = 0 : i32} : tensor<1024xi32>
    // offset = 0, size = 1024, strides = 1
    %8 = tt.splat %6 : !tt.ptr<f32> -> tensor<1024x!tt.ptr<f32>>
    // source = arg1, offset = %1, size = 1024, strides = 0
    %9 = tt.addptr %8, %7 : tensor<1024x!tt.ptr<f32>>, tensor<1024xi32>
    // source = arg1, offset = %1+%3+%5, size = 1024, strides = 1
    %10 = tt.load %9 : tensor<1024x!tt.ptr<f32>>
    %17 = math.exp %10 : tensor<1024xf32>
    %18 = arith.muli %0, %arg3 : i32
    %19 = tt.addptr %arg0, %18 : !tt.ptr<f32>, i32
    // source = arg0, offset = %18, size = 1, strides = 0
    %20 = tt.splat %19 : !tt.ptr<f32> -> tensor<1024x!tt.ptr<f32>>
    // source = arg0, offset = %18, size = 1024, strides = 0
    %21 = tt.addptr %20, %7 : tensor<1024x!tt.ptr<f32>>, tensor<1024xi32>
    // source = arg0, offset = %18, size = 1024, strides = 1
    tt.store %21, %17 : tensor<1024x!tt.ptr<f32>>
    tt.return
  }
}

// CHECK:         tt.func @kernel([[PARAM_0_:%.+]]: !tt.ptr<f32> {tt.divisibility = 16 : i32}, [[PARAM_1_:%.+]]: !tt.ptr<f32> {tt.divisibility = 16 : i32}, [[PARAM_2_:%.+]]: i32, [[PARAM_3_:%.+]]: i32, [[PARAM_4_:%.+]]: i32) {
// CHECK:           [[VAR_0_:%.+]] = tt.get_program_id x : i32
// CHECK:           [[VAR_1_:%.+]] = arith.muli [[VAR_0_]], [[PARAM_2_]] : i32
// CHECK-DAG:       [[VAR_2_:%.+]] = arith.index_cast [[VAR_1_]] : i32 to index
// CHECK-DAG:       [[VAR_3_:%.+]] = arith.muli [[VAR_0_]], [[PARAM_3_]] : i32
// CHECK:           [[VAR_4_:%.+]] = arith.index_cast [[VAR_3_]] : i32 to index
// CHECK-DAG:       [[VAR_5_:%.+]] = arith.addi [[VAR_2_]], [[VAR_4_]] : index
// CHECK-DAG:       [[VAR_6_:%.+]] = arith.muli [[VAR_0_]], [[PARAM_4_]] : i32
// CHECK:           [[VAR_7_:%.+]] = arith.index_cast [[VAR_6_]] : i32 to index
// CHECK:           [[VAR_8_:%.+]] = arith.addi [[VAR_5_]], [[VAR_7_]] : index
// CHECK:           [[VAR_9_:%.+]] = tts.make_tptr [[PARAM_1_]] to sizes: [1024], strides: [1], offsets: {{.}}[[VAR_8_]]{{.}}, shape: [0], order: [] : <f32> to tensor<1024x!tt.ptr<f32>>
// CHECK:           [[VAR_10_:%.+]] = "tts.load"([[VAR_9_]]) <{operandSegmentSizes = array<i32: 1, 0, 0>, static_mask_dims = array<i64>}> : (tensor<1024x!tt.ptr<f32>>) -> tensor<1024xf32>
// CHECK-DAG:       [[VAR_11_:%.+]] = math.exp [[VAR_10_]] : tensor<1024xf32>
// CHECK-DAG:       [[VAR_12_:%.+]] = arith.muli [[VAR_0_]], [[PARAM_3_]] : i32
// CHECK:           [[VAR_13_:%.+]] = arith.index_cast [[VAR_12_]] : i32 to index
// CHECK:           [[VAR_14_:%.+]] = tts.make_tptr [[PARAM_0_]] to sizes: [1024], strides: [1], offsets: {{.}}[[VAR_13_]]{{.}}, shape: [0], order: [] : <f32> to tensor<1024x!tt.ptr<f32>>
// CHECK:           "tts.store"([[VAR_14_]], [[VAR_11_]]) <{static_mask_dims = array<i64>}> : (tensor<1024x!tt.ptr<f32>>, tensor<1024xf32>) -> ()
// CHECK:           tt.return
// CHECK:         }
