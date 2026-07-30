// RUN: triton-shared-opt --triton-to-structured --remove-dead-values --canonicalize --cse %s | FileCheck %s

// IR from python/examples/test_tensor_index_iterargs.py
module {
  tt.func public @tensor_indices_nested(%arg0: !tt.ptr<f32>, %arg1: !tt.ptr<f32>) attributes {noinline = false} {
    %c1_i32 = arith.constant 1 : i32
    %c0_i32 = arith.constant 0 : i32
    %c3_i32 = arith.constant 3 : i32
    %cst = arith.constant dense<4> : tensor<4xi32>
    %c2_i32 = arith.constant 2 : i32
    %0 = tt.make_range {end = 4 : i32, start = 0 : i32} : tensor<4xi32>
    %1 = tt.splat %arg0 : !tt.ptr<f32> -> tensor<4x!tt.ptr<f32>>
    %2 = tt.splat %arg1 : !tt.ptr<f32> -> tensor<4x!tt.ptr<f32>>
    %3:2 = scf.for %arg2 = %c0_i32 to %c2_i32 step %c1_i32 iter_args(%arg3 = %0, %arg4 = %0) -> (tensor<4xi32>, tensor<4xi32>)  : i32 {
      %4 = arith.muli %arg2, %c2_i32 : i32
      %5 = tt.splat %4 : i32 -> tensor<4xi32>
      %6 = arith.addi %arg3, %5 : tensor<4xi32>
      %7 = tt.addptr %1, %6 : tensor<4x!tt.ptr<f32>>, tensor<4xi32>
      %8 = tt.load %7 : tensor<4x!tt.ptr<f32>>
      %9 = tt.addptr %2, %arg4 : tensor<4x!tt.ptr<f32>>, tensor<4xi32>
      tt.store %9, %8 : tensor<4x!tt.ptr<f32>>
      %10 = arith.addi %6, %cst : tensor<4xi32>
      %11 = arith.addi %arg4, %cst : tensor<4xi32>
      %12:2 = scf.for %arg5 = %c0_i32 to %c3_i32 step %c1_i32 iter_args(%arg6 = %10, %arg7 = %11) -> (tensor<4xi32>, tensor<4xi32>)  : i32 {
        %13 = arith.muli %arg5, %c3_i32 : i32
        %14 = tt.splat %13 : i32 -> tensor<4xi32>
        %15 = arith.addi %arg6, %14 : tensor<4xi32>
        %16 = tt.addptr %1, %15 : tensor<4x!tt.ptr<f32>>, tensor<4xi32>
        %17 = tt.load %16 : tensor<4x!tt.ptr<f32>>
        %18 = tt.addptr %2, %arg7 : tensor<4x!tt.ptr<f32>>, tensor<4xi32>
        tt.store %18, %17 : tensor<4x!tt.ptr<f32>>
        %19 = arith.addi %15, %cst : tensor<4xi32>
        %20 = arith.addi %arg7, %cst : tensor<4xi32>
        scf.yield %19, %20 : tensor<4xi32>, tensor<4xi32>
      }
      scf.yield %12#0, %12#1 : tensor<4xi32>, tensor<4xi32>
    }
    tt.return
  }
}

// CHECK:         tt.func public @tensor_indices_nested([[arg0_:.+]]: !tt.ptr<f32>, [[arg1_:.+]]: !tt.ptr<f32>) attributes {noinline = false} {
// CHECK-DAG:       [[CST_4_:%.+]] = arith.constant 4 : index
// CHECK-DAG:       [[CST_1_:%.+]] = arith.constant 1 : i32
// CHECK-DAG:       [[CST_0_:%.+]] = arith.constant 0 : i32
// CHECK-DAG:       [[CST_3_:%.+]] = arith.constant 3 : i32
// CHECK-DAG:       [[CST_2_:%.+]] = arith.constant 2 : i32
// CHECK-DAG:       [[CST_0_1_:%.+]] = arith.constant 0 : index
// CHECK-DAG:       [[CST_1_1_:%.+]] = arith.constant 1 : index
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:       [[VAR_0_:%.+]]:2 = scf.for [[VAR_arg2_:%.+]] = [[CST_0_]] to [[CST_2_]] step [[CST_1_]] iter_args([[VAR_arg3_:%.+]] = [[CST_0_1_]], [[VAR_arg4_:%.+]] = [[CST_0_1_]]) -> (index, index)  : i32 {
// CHECK-DAG:         [[VAR_1_:%.+]] = arith.muli [[VAR_arg2_]], [[CST_2_]] : i32
// CHECK:             [[VAR_2_:%.+]] = arith.index_cast [[VAR_1_]] : i32 to index
// CHECK:             [[VAR_3_:%.+]] = arith.addi [[VAR_arg3_]], [[VAR_2_]] : index
// CHECK:             [[VAR_4_:%.+]] = tts.make_tptr [[arg0_]] to sizes: [4], strides: {{.}}[[CST_1_1_]]{{.}}, offsets: {{.}}[[VAR_3_]]{{.}}, shape: [0], order: [] : <f32> to tensor<4x!tt.ptr<f32>>
// CHECK-DAG:         [[VAR_5_:%.+]] = "tts.load"([[VAR_4_]]) <{operandSegmentSizes = array<i32: 1, 0, 0>, static_mask_dims = array<i64>}> : (tensor<4x!tt.ptr<f32>>) -> tensor<4xf32>
// CHECK-DAG:         [[VAR_6_:%.+]] = tts.make_tptr [[arg1_]] to sizes: [4], strides: {{.}}[[CST_1_1_]]{{.}}, offsets: {{.}}[[VAR_arg4_]]{{.}}, shape: [0], order: [] : <f32> to tensor<4x!tt.ptr<f32>>
// CHECK:             "tts.store"([[VAR_6_]], [[VAR_5_]]) <{static_mask_dims = array<i64>}> : (tensor<4x!tt.ptr<f32>>, tensor<4xf32>) -> ()
// CHECK-DAG:         [[VAR_7_:%.+]] = arith.addi [[VAR_3_]], [[CST_4_]] : index
// CHECK-DAG:         [[VAR_8_:%.+]] = arith.addi [[VAR_arg4_]], [[CST_4_]] : index
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:         [[VAR_9_:%.+]]:2 = scf.for [[VAR_arg5_:%.+]] = [[CST_0_]] to [[CST_3_]] step [[CST_1_]] iter_args([[VAR_arg6_:%.+]] = [[VAR_7_]], [[VAR_arg7_:%.+]] = [[VAR_8_]]) -> (index, index)  : i32 {
// CHECK-DAG:           [[VAR_10_:%.+]] = arith.muli [[VAR_arg5_]], [[CST_3_]] : i32
// CHECK:               [[VAR_11_:%.+]] = arith.index_cast [[VAR_10_]] : i32 to index
// CHECK:               [[VAR_12_:%.+]] = arith.addi [[VAR_arg6_]], [[VAR_11_]] : index
// CHECK:               [[VAR_13_:%.+]] = tts.make_tptr [[arg0_]] to sizes: [4], strides: {{.}}[[CST_1_1_]]{{.}}, offsets: {{.}}[[VAR_12_]]{{.}}, shape: [0], order: [] : <f32> to tensor<4x!tt.ptr<f32>>
// CHECK-DAG:           [[VAR_14_:%.+]] = "tts.load"([[VAR_13_]]) <{operandSegmentSizes = array<i32: 1, 0, 0>, static_mask_dims = array<i64>}> : (tensor<4x!tt.ptr<f32>>) -> tensor<4xf32>
// CHECK-DAG:           [[VAR_15_:%.+]] = tts.make_tptr [[arg1_]] to sizes: [4], strides: {{.}}[[CST_1_1_]]{{.}}, offsets: {{.}}[[VAR_arg7_]]{{.}}, shape: [0], order: [] : <f32> to tensor<4x!tt.ptr<f32>>
// CHECK:               "tts.store"([[VAR_15_]], [[VAR_14_]]) <{static_mask_dims = array<i64>}> : (tensor<4x!tt.ptr<f32>>, tensor<4xf32>) -> ()
// CHECK-DAG:           [[VAR_16_:%.+]] = arith.addi [[VAR_12_]], [[CST_4_]] : index
// CHECK-DAG:           [[VAR_17_:%.+]] = arith.addi [[VAR_arg7_]], [[CST_4_]] : index
// CHECK:               scf.yield [[VAR_16_]], [[VAR_17_]] : index, index
// CHECK:             }
// CHECK:             scf.yield [[VAR_9_]]#0, [[VAR_9_]]#1 : index, index
// CHECK:           }
// CHECK:           tt.return
// CHECK:         }
