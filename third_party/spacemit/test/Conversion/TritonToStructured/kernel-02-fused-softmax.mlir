// RUN: triton-shared-opt --triton-to-structured --canonicalize %s | FileCheck %s

module {
  tt.func public @softmax_kernel_012345(%arg0: !tt.ptr<f32>, %arg1: !tt.ptr<f32>, %arg2: i32, %arg3: i32, %arg4: i32) {
    %cst = arith.constant 0xFF800000 : f32
    %0 = tt.get_program_id x : i32
    %1 = arith.muli %0, %arg2 : i32
    %2 = tt.addptr %arg1, %1 : !tt.ptr<f32>, i32
    %3 = tt.make_range {end = 128 : i32, start = 0 : i32} : tensor<128xi32>
    %4 = tt.splat %2 : !tt.ptr<f32> -> tensor<128x!tt.ptr<f32>>
    %5 = tt.addptr %4, %3 : tensor<128x!tt.ptr<f32>>, tensor<128xi32>
    %6 = tt.splat %arg4 : i32 -> tensor<128xi32>
    %7 = arith.cmpi slt, %3, %6 : tensor<128xi32>
    %8 = tt.splat %cst : f32 -> tensor<128xf32>
    %9 = tt.load %5, %7, %8 : tensor<128x!tt.ptr<f32>>
    %10 = "tt.reduce"(%9) ({
    ^bb0(%arg5: f32, %arg6: f32):
      %21 = arith.cmpf ogt, %arg5, %arg6 : f32
      %22 = arith.select %21, %arg5, %arg6 : f32
      tt.reduce.return %22 : f32
    }) {axis = 0 : i32} : (tensor<128xf32>) -> f32
    %11 = tt.splat %10 : f32 -> tensor<128xf32>
    %12 = arith.subf %9, %11 : tensor<128xf32>
    %13 = math.exp %12 : tensor<128xf32>
    %14 = "tt.reduce"(%13) ({
    ^bb0(%arg5: f32, %arg6: f32):
      %21 = arith.addf %arg5, %arg6 : f32
      tt.reduce.return %21 : f32
    }) {axis = 0 : i32} : (tensor<128xf32>) -> f32
    %15 = tt.splat %14 : f32 -> tensor<128xf32>
    %16 = arith.divf %13, %15 : tensor<128xf32>
    %17 = arith.muli %0, %arg3 : i32
    %18 = tt.addptr %arg0, %17 : !tt.ptr<f32>, i32
    %19 = tt.splat %18 : !tt.ptr<f32> -> tensor<128x!tt.ptr<f32>>
    %20 = tt.addptr %19, %3 : tensor<128x!tt.ptr<f32>>, tensor<128xi32>
    tt.store %20, %16, %7 : tensor<128x!tt.ptr<f32>>
    tt.return
  }
}

// CHECK:         tt.func public @softmax_kernel_012345([[PARAM_0_:%.+]]: !tt.ptr<f32>, [[PARAM_1_:%.+]]: !tt.ptr<f32>, [[PARAM_2_:%.+]]: i32, [[PARAM_3_:%.+]]: i32, [[PARAM_4_:%.+]]: i32) {
// CHECK-DAG:       [[CST_0_:%.+]] = arith.constant 0xFF800000 : f32
// CHECK-DAG:       [[CST_0_1_:%.+]] = arith.constant 0 : index
// CHECK-DAG:       [[CST_128_:%.+]] = arith.constant 128 : index
// CHECK-DAG:       [[VAR_0_:%.+]] = tt.get_program_id x : i32
// CHECK:           [[VAR_1_:%.+]] = arith.muli [[VAR_0_]], [[PARAM_2_]] : i32
// CHECK:           [[VAR_2_:%.+]] = arith.index_cast [[VAR_1_]] : i32 to index
// CHECK-DAG:       [[VAR_3_:%.+]] = tts.make_tptr [[PARAM_1_]] to sizes: [128], strides: [1], offsets: {{.}}[[VAR_2_]]{{.}}, shape: [0], order: [] : <f32> to tensor<128x!tt.ptr<f32>>
// CHECK-DAG:       [[VAR_4_:%.+]] = arith.index_cast [[PARAM_4_]] : i32 to index
// CHECK:           [[VAR_5_:%.+]] = arith.minsi [[VAR_4_]], [[CST_128_]] : index
// CHECK:           [[VAR_6_:%.+]] = arith.maxsi [[VAR_5_]], [[CST_0_1_]] : index
// CHECK:           [[VAR_7_:%.+]] = "tts.load"([[VAR_3_]], [[VAR_6_]], [[CST_0_]]) <{operandSegmentSizes = array<i32: 1, 1, 1>, static_mask_dims = array<i64: -9223372036854775808>}> : (tensor<128x!tt.ptr<f32>>, index, f32) -> tensor<128xf32>
// CHECK:           [[VAR_8_:%.+]] = "tt.reduce"([[VAR_7_]]) <{axis = 0 : i32}> ({
// CHECK:           ^bb0([[IN_0_:%.+]]: f32, [[IN_1_:%.+]]: f32):
// CHECK:             [[VAR_21_:%.+]] = arith.cmpf ogt, [[IN_0_]], [[IN_1_]] : f32
// CHECK:             [[VAR_22_:%.+]] = arith.select [[VAR_21_]], [[IN_0_]], [[IN_1_]] : f32
// CHECK:             tt.reduce.return [[VAR_22_]] : f32
// CHECK:           }) : (tensor<128xf32>) -> f32
// CHECK:           [[VAR_9_:%.+]] = tt.splat [[VAR_8_]] : f32 -> tensor<128xf32>
// CHECK:           [[VAR_10_:%.+]] = arith.subf [[VAR_7_]], [[VAR_9_]] : tensor<128xf32>
// CHECK:           [[VAR_11_:%.+]] = math.exp [[VAR_10_]] : tensor<128xf32>
// CHECK:           [[VAR_12_:%.+]] = "tt.reduce"([[VAR_11_]]) <{axis = 0 : i32}> ({
// CHECK:           ^bb0([[IN_2_:%.+]]: f32, [[IN_3_:%.+]]: f32):
// CHECK:             [[VAR_21_1_:%.+]] = arith.addf [[IN_2_]], [[IN_3_]] : f32
// CHECK:             tt.reduce.return [[VAR_21_1_]] : f32
// CHECK:           }) : (tensor<128xf32>) -> f32
// CHECK:           [[VAR_13_:%.+]] = tt.splat [[VAR_12_]] : f32 -> tensor<128xf32>
// CHECK-DAG:       [[VAR_14_:%.+]] = arith.divf [[VAR_11_]], [[VAR_13_]] : tensor<128xf32>
// CHECK-DAG:       [[VAR_15_:%.+]] = arith.muli [[VAR_0_]], [[PARAM_3_]] : i32
// CHECK:           [[VAR_16_:%.+]] = arith.index_cast [[VAR_15_]] : i32 to index
// CHECK-DAG:       [[VAR_17_:%.+]] = tts.make_tptr [[PARAM_0_]] to sizes: [128], strides: [1], offsets: {{.}}[[VAR_16_]]{{.}}, shape: [0], order: [] : <f32> to tensor<128x!tt.ptr<f32>>
// CHECK-DAG:       [[VAR_18_:%.+]] = arith.index_cast [[PARAM_4_]] : i32 to index
// CHECK:           [[VAR_19_:%.+]] = arith.minsi [[VAR_18_]], [[CST_128_]] : index
// CHECK:           [[VAR_20_:%.+]] = arith.maxsi [[VAR_19_]], [[CST_0_1_]] : index
// CHECK:           "tts.store"([[VAR_17_]], [[VAR_14_]], [[VAR_20_]]) <{static_mask_dims = array<i64: -9223372036854775808>}> : (tensor<128x!tt.ptr<f32>>, tensor<128xf32>, index) -> ()
// CHECK:           tt.return
// CHECK:         }
