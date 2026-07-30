// RUN: triton-shared-opt --triton-to-structured --canonicalize --cse %s | FileCheck %s
// IR from python/examples/sign_extend.py
module {
  tt.func public @sign_extend(%arg0: !tt.ptr<i32>, %arg1: !tt.ptr<f32>, %arg2: !tt.ptr<f32>, %arg3: i32) attributes {noinline = false} {
    %cst = arith.constant dense<1.100000e+01> : tensor<4xf32>
    %0 = tt.load %arg0 : !tt.ptr<i32>
    %1 = arith.extsi %0 : i32 to i64
    %2 = tt.make_range {end = 4 : i32, start = 0 : i32} : tensor<4xi32>
    %3 = arith.extsi %2 : tensor<4xi32> to tensor<4xi64>
    %4 = tt.splat %1 : i64 -> tensor<4xi64>
    %5 = arith.addi %4, %3 : tensor<4xi64>
    %6 = arith.extsi %arg3 : i32 to i64
    %7 = tt.splat %6 : i64 -> tensor<4xi64>
    %8 = arith.cmpi slt, %5, %7 : tensor<4xi64>
    %9 = tt.splat %arg1 : !tt.ptr<f32> -> tensor<4x!tt.ptr<f32>>
    %10 = tt.addptr %9, %5 : tensor<4x!tt.ptr<f32>>, tensor<4xi64>
    %11 = tt.load %10, %8, %cst : tensor<4x!tt.ptr<f32>>
    %12 = tt.splat %arg2 : !tt.ptr<f32> -> tensor<4x!tt.ptr<f32>>
    %13 = tt.addptr %12, %2 : tensor<4x!tt.ptr<f32>>, tensor<4xi32>
    tt.store %13, %11 : tensor<4x!tt.ptr<f32>>
    tt.return
  }
}

// CHECK:         tt.func public @sign_extend([[PARAM_0_:%.+]]: !tt.ptr<i32>, [[PARAM_1_:%.+]]: !tt.ptr<f32>, [[PARAM_2_:%.+]]: !tt.ptr<f32>, [[PARAM_3_:%.+]]: i32) attributes {noinline = false} {
// CHECK-DAG:       [[CST_1_dot_100000_:%.+]] = arith.constant 1.100000e+01 : f32
// CHECK-DAG:       [[CST_4_:%.+]] = arith.constant 4 : index
// CHECK-DAG:       [[LOAD_PARAM_0_MEM_:%.+]] = tt.load [[PARAM_0_]] : !tt.ptr<i32>
// CHECK:           [[VAR_1_:%.+]] = arith.index_cast [[LOAD_PARAM_0_MEM_]] : i32 to index
// CHECK-DAG:       [[VAR_2_:%.+]] = tts.make_tptr [[PARAM_1_]] to sizes: [4], strides: [1], offsets: {{.}}[[VAR_1_]]{{.}}, shape: [0], order: [] : <f32> to tensor<4x!tt.ptr<f32>>
// CHECK-DAG:       [[VAR_3_:%.+]] = arith.addi [[VAR_1_]], [[CST_4_]] : index
// CHECK-DAG:       [[VAR_4_:%.+]] = arith.index_cast [[PARAM_3_]] : i32 to index
// CHECK:           [[VAR_5_:%.+]] = arith.minsi [[VAR_3_]], [[VAR_4_]] : index
// CHECK:           [[VAR_6_:%.+]] = arith.maxsi [[VAR_5_]], [[VAR_1_]] : index
// CHECK:           [[VAR_7_:%.+]] = arith.subi [[VAR_6_]], [[VAR_1_]] : index
// CHECK-DAG:       [[VAR_8_:%.+]] = "tts.load"([[VAR_2_]], [[VAR_7_]], [[CST_1_dot_100000_]]) <{operandSegmentSizes = array<i32: 1, 1, 1>, static_mask_dims = array<i64: -9223372036854775808>}> : (tensor<4x!tt.ptr<f32>>, index, f32) -> tensor<4xf32>
// CHECK-DAG:       [[VAR_9_:%.+]] = tts.make_tptr [[PARAM_2_]] to sizes: [4], strides: [1], offsets: [0], shape: [0], order: [] : <f32> to tensor<4x!tt.ptr<f32>>
// CHECK:           "tts.store"([[VAR_9_]], [[VAR_8_]]) <{static_mask_dims = array<i64>}> : (tensor<4x!tt.ptr<f32>>, tensor<4xf32>) -> ()
// CHECK:           tt.return
// CHECK:         }
