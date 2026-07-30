// RUN: triton-shared-opt --triton-to-structured --remove-dead-values --canonicalize --cse %s | FileCheck %s

// IR from python/examples/test_tensor_index_iterargs.py
module {
  tt.func public @addptr_with_masks(%arg0: !tt.ptr<f32>, %arg1: !tt.ptr<f32>, %arg2: i32) attributes {noinline = false} {
    %cst = arith.constant dense<-1.100000e+01> : tensor<4xf32>
    %c1_i32 = arith.constant 1 : i32
    %c4_i32 = arith.constant 4 : i32
    %c0_i32 = arith.constant 0 : i32
    %cst_0 = arith.constant dense<4> : tensor<4xi32>
    %0 = tt.make_range {end = 4 : i32, start = 0 : i32} : tensor<4xi32>
    %1 = tt.splat %arg2 : i32 -> tensor<4xi32>
    %2 = tt.splat %arg0 : !tt.ptr<f32> -> tensor<4x!tt.ptr<f32>>
    %3 = tt.splat %arg1 : !tt.ptr<f32> -> tensor<4x!tt.ptr<f32>>
    %4:2 = scf.for %arg3 = %c0_i32 to %c4_i32 step %c1_i32 iter_args(%arg4 = %0, %arg5 = %0) -> (tensor<4xi32>, tensor<4xi32>)  : i32 {
      %5 = arith.cmpi slt, %arg4, %1 : tensor<4xi32>
      %6 = tt.addptr %2, %arg4 : tensor<4x!tt.ptr<f32>>, tensor<4xi32>
      %7 = tt.load %6, %5, %cst : tensor<4x!tt.ptr<f32>>
      %8 = tt.addptr %3, %arg5 : tensor<4x!tt.ptr<f32>>, tensor<4xi32>
      tt.store %8, %7 : tensor<4x!tt.ptr<f32>>
      %9 = arith.addi %arg4, %cst_0 : tensor<4xi32>
      %10 = arith.addi %arg5, %cst_0 : tensor<4xi32>
      scf.yield %9, %10 : tensor<4xi32>, tensor<4xi32>
    }
    tt.return
  }
}

// CHECK:         tt.func public @addptr_with_masks([[PARAM_0_:%.+]]: !tt.ptr<f32>, [[PARAM_1_:%.+]]: !tt.ptr<f32>, [[PARAM_2_:%.+]]: i32) attributes {noinline = false} {
// CHECK-DAG:       [[CST_minus_1_dot_100000_:%.+]] = arith.constant -1.100000e+01 : f32
// CHECK-DAG:       [[CST_4_:%.+]] = arith.constant 4 : index
// CHECK-DAG:       [[CST_1_:%.+]] = arith.constant 1 : i32
// CHECK-DAG:       [[CST_4_1_:%.+]] = arith.constant 4 : i32
// CHECK-DAG:       [[CST_0_:%.+]] = arith.constant 0 : i32
// CHECK-DAG:       [[CST_0_1_:%.+]] = arith.constant 0 : index
// CHECK-DAG:       [[CST_1_1_:%.+]] = arith.constant 1 : index
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:       [[VAR_0_:%.+]]:2 = scf.for [[VAR_arg3_:%.+]] = [[CST_0_]] to [[CST_4_1_]] step [[CST_1_]] iter_args([[VAR_arg4_:%.+]] = [[CST_0_1_]], [[VAR_arg5_:%.+]] = [[CST_0_1_]]) -> (index, index)  : i32 {
// CHECK-DAG:         [[VAR_1_:%.+]] = tts.make_tptr [[PARAM_0_]] to sizes: [4], strides: {{.}}[[CST_1_1_]]{{.}}, offsets: {{.}}[[VAR_arg4_]]{{.}}, shape: [0], order: [] : <f32> to tensor<4x!tt.ptr<f32>>
// CHECK-DAG:         [[VAR_2_:%.+]] = arith.addi [[VAR_arg4_]], [[CST_4_]] : index
// CHECK-DAG:         [[VAR_3_:%.+]] = arith.index_cast [[PARAM_2_]] : i32 to index
// CHECK:             [[VAR_4_:%.+]] = arith.minsi [[VAR_2_]], [[VAR_3_]] : index
// CHECK:             [[VAR_5_:%.+]] = arith.maxsi [[VAR_4_]], [[VAR_arg4_]] : index
// CHECK:             [[VAR_6_:%.+]] = arith.subi [[VAR_5_]], [[VAR_arg4_]] : index
// CHECK-DAG:         [[VAR_7_:%.+]] = "tts.load"([[VAR_1_]], [[VAR_6_]], [[CST_minus_1_dot_100000_]]) <{operandSegmentSizes = array<i32: 1, 1, 1>, static_mask_dims = array<i64: -9223372036854775808>}> : (tensor<4x!tt.ptr<f32>>, index, f32) -> tensor<4xf32>
// CHECK-DAG:         [[VAR_8_:%.+]] = tts.make_tptr [[PARAM_1_]] to sizes: [4], strides: {{.}}[[CST_1_1_]]{{.}}, offsets: {{.}}[[VAR_arg5_]]{{.}}, shape: [0], order: [] : <f32> to tensor<4x!tt.ptr<f32>>
// CHECK:             "tts.store"([[VAR_8_]], [[VAR_7_]]) <{static_mask_dims = array<i64>}> : (tensor<4x!tt.ptr<f32>>, tensor<4xf32>) -> ()
// CHECK:             [[VAR_9_:%.+]] = arith.addi [[VAR_arg5_]], [[CST_4_]] : index
// CHECK:             scf.yield [[VAR_2_]], [[VAR_9_]] : index, index
// CHECK:           }
// CHECK:           tt.return
// CHECK:         }
