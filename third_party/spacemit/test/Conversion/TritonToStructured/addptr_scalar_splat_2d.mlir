// RUN: triton-shared-opt --triton-to-structured --remove-dead-values --canonicalize %s | FileCheck %s

module {
  tt.func @kernel (%arg0: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %arg1: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %arg2: i32, %arg3: i32, %arg4: i32) {
    %0 = tt.get_program_id x : i32
    %1 = arith.muli %0, %arg2 : i32
    %2 = tt.addptr %arg1, %1 : !tt.ptr<f32>, i32
    %3 = tt.splat %2 : !tt.ptr<f32> -> tensor<128x128x!tt.ptr<f32>>
    // source = %arg1, offset = [%1, 0], size = [128, 128], strides = [0, 0]
    %4 = tt.make_range {end = 128 : i32, start = 0 : i32} : tensor<128xi32>
    %5 = tt.expand_dims %4 {axis = 0 : i32} : tensor<128xi32> -> tensor<1x128xi32>
    %6 = tt.broadcast %5 : tensor<1x128xi32> -> tensor<128x128xi32>
    // offset = [0, 0], size = [128, 128], strides = [0, 1]
    %7 = tt.make_range {end = 256 : i32, start = 128 : i32} : tensor<128xi32>
    // offset = 128, size = 128, strides = 1
    %8 = tt.expand_dims %7 {axis = 1 : i32} : tensor<128xi32> -> tensor<128x1xi32>
    %9 = tt.broadcast %8 : tensor<128x1xi32> -> tensor<128x128xi32>
    // offset = [128, 0], size = [128, 128], strides = [1, 0]
    %10 = arith.addi %6, %9 : tensor<128x128xi32>
    // offset = [128, 0], size = [128, 128], strides = [1, 1]
    %11 = tt.addptr %3, %10 : tensor<128x128x!tt.ptr<f32>>, tensor<128x128xi32>
    // source = %arg1, offset = [%1 + 128, 0], size = [128, 128], strides = [1, 1]
    %12 = tt.load %11 : tensor<128x128x!tt.ptr<f32>>
    %17 = math.exp %12 : tensor<128x128xf32>
    %18 = arith.muli %0, %arg3 : i32
    %19 = tt.addptr %arg0, %18 : !tt.ptr<f32>, i32
    // source = arg0, offset = %18, size = 1, strides = 0
    %20 = tt.splat %19 : !tt.ptr<f32> -> tensor<128x128x!tt.ptr<f32>>
    // source = arg0, offset = [%18, 0], size = [128, 128], strides = [0, 0]
    %21 = tt.addptr %20, %10 : tensor<128x128x!tt.ptr<f32>>, tensor<128x128xi32>
    // source = %arg0, offset = [%18 + 128, 0], size = [128, 128], strides = [1, 1]
    tt.store %21, %17 : tensor<128x128x!tt.ptr<f32>>
    tt.return
  }
}

// CHECK:         tt.func @kernel([[PARAM_0_:%.+]]: !tt.ptr<f32> {tt.divisibility = 16 : i32}, [[PARAM_1_:%.+]]: !tt.ptr<f32> {tt.divisibility = 16 : i32}, [[PARAM_2_:%.+]]: i32, [[PARAM_3_:%.+]]: i32, [[PARAM_4_:%.+]]: i32) {
// CHECK-DAG:       [[CST_128_:%.+]] = arith.constant 128 : index
// CHECK-DAG:       [[VAR_0_:%.+]] = tt.get_program_id x : i32
// CHECK:           [[VAR_1_:%.+]] = arith.muli [[VAR_0_]], [[PARAM_2_]] : i32
// CHECK:           [[VAR_2_:%.+]] = arith.index_cast [[VAR_1_]] : i32 to index
// CHECK:           [[VAR_3_:%.+]] = arith.addi [[VAR_2_]], [[CST_128_]] : index
// CHECK:           [[VAR_4_:%.+]] = tts.make_tptr [[PARAM_1_]] to sizes: [128, 128], strides: [1, 1], offsets: {{.}}[[VAR_3_]], 0], shape: [0, 0], order: [] : <f32> to tensor<128x128x!tt.ptr<f32>>
// CHECK:           [[VAR_5_:%.+]] = "tts.load"([[VAR_4_]]) <{operandSegmentSizes = array<i32: 1, 0, 0>, static_mask_dims = array<i64>}> : (tensor<128x128x!tt.ptr<f32>>) -> tensor<128x128xf32>
// CHECK-DAG:       [[VAR_6_:%.+]] = math.exp [[VAR_5_]] : tensor<128x128xf32>
// CHECK-DAG:       [[VAR_7_:%.+]] = arith.muli [[VAR_0_]], [[PARAM_3_]] : i32
// CHECK:           [[VAR_8_:%.+]] = arith.index_cast [[VAR_7_]] : i32 to index
// CHECK:           [[VAR_9_:%.+]] = arith.addi [[VAR_8_]], [[CST_128_]] : index
// CHECK:           [[VAR_10_:%.+]] = tts.make_tptr [[PARAM_0_]] to sizes: [128, 128], strides: [1, 1], offsets: {{.}}[[VAR_9_]], 0], shape: [0, 0], order: [] : <f32> to tensor<128x128x!tt.ptr<f32>>
// CHECK:           "tts.store"([[VAR_10_]], [[VAR_6_]]) <{static_mask_dims = array<i64>}> : (tensor<128x128x!tt.ptr<f32>>, tensor<128x128xf32>) -> ()
// CHECK:           tt.return
// CHECK:         }
