// RUN: triton-shared-opt --triton-to-linalg-experimental %s | FileCheck %s

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

// CHECK-LABEL:  func.func @tensor_indices_nested
// CHECK-SAME:   ([[PARAM_0_:%.+]]: memref<*xf32>, [[PARAM_1_:%.+]]: memref<*xf32>, [[PARAM_2_:%.+]]: i32, [[PARAM_3_:%.+]]: i32, [[PARAM_4_:%.+]]: i32, [[PARAM_5_:%.+]]: i32, [[PARAM_6_:%.+]]: i32, [[PARAM_7_:%.+]]: i32) {
// CHECK-DAG:       [[CST_3_:%.+]] = arith.constant 3 : i32
// CHECK-DAG:       [[CST_1_:%.+]] = arith.constant 1 : i32
// CHECK-DAG:       [[CST_0_:%.+]] = arith.constant 0 : index
// CHECK-DAG:       [[CST_4_:%.+]] = arith.constant 4 : index
// CHECK-DAG:       [[CST_0_1_:%.+]] = arith.constant 0 : i32
// CHECK-DAG:       [[CST_2_:%.+]] = arith.constant 2 : i32
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:       [[VAR_0_:%.+]]:2 = scf.for [[VAR_arg8_:%.+]] = [[CST_0_1_]] to [[CST_2_]] step [[CST_1_]] iter_args([[VAR_arg9_:%.+]] = [[CST_0_]], [[VAR_arg10_:%.+]] = [[CST_0_]]) -> (index, index)  : i32 {
// CHECK-DAG:         [[VAR_1_:%.+]] = arith.muli [[VAR_arg8_]], [[CST_2_]] : i32
// CHECK:             [[VAR_2_:%.+]] = arith.index_cast [[VAR_1_]] : i32 to index
// CHECK:             [[VAR_3_:%.+]] = arith.addi [[VAR_arg9_]], [[VAR_2_]] : index
// CHECK-DAG:         [[VAR_reinterpret_cast_:%.+]] = memref.reinterpret_cast [[PARAM_0_]] to offset: {{.}}[[VAR_3_]]{{.}}, sizes: [4], strides: [1] : memref<*xf32> to memref<4xf32, strided<[1], offset: ?>>
// CHECK-DAG:         [[RES_:%.+]] = memref.alloc() : memref<4xf32>
// CHECK:             memref.copy [[VAR_reinterpret_cast_]], [[RES_]] : memref<4xf32, strided<[1], offset: ?>> to memref<4xf32>
// CHECK-DAG:         [[VAR_4_:%.+]] = bufferization.to_tensor [[RES_]] restrict writable : memref<4xf32> to tensor<4xf32>
// CHECK-DAG:         [[VAR_reinterpret_cast_0_:%.+]] = memref.reinterpret_cast [[PARAM_1_]] to offset: {{.}}[[VAR_arg10_]]{{.}}, sizes: [4], strides: [1] : memref<*xf32> to memref<4xf32, strided<[1], offset: ?>>
// CHECK:             bufferization.materialize_in_destination [[VAR_4_]] in writable [[VAR_reinterpret_cast_0_]] : (tensor<4xf32>, memref<4xf32, strided<[1], offset: ?>>) -> ()
// CHECK-DAG:         [[VAR_5_:%.+]] = arith.addi [[VAR_3_]], [[CST_4_]] : index
// CHECK-DAG:         [[VAR_6_:%.+]] = arith.addi [[VAR_arg10_]], [[CST_4_]] : index
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:         [[VAR_7_:%.+]]:2 = scf.for [[VAR_arg11_:%.+]] = [[CST_0_1_]] to [[CST_3_]] step [[CST_1_]] iter_args([[VAR_arg12_:%.+]] = [[VAR_5_]], [[VAR_arg13_:%.+]] = [[VAR_6_]]) -> (index, index)  : i32 {
// CHECK-DAG:           [[VAR_8_:%.+]] = arith.muli [[VAR_arg11_]], [[CST_3_]] : i32
// CHECK:               [[VAR_9_:%.+]] = arith.index_cast [[VAR_8_]] : i32 to index
// CHECK:               [[VAR_10_:%.+]] = arith.addi [[VAR_arg12_]], [[VAR_9_]] : index
// CHECK-DAG:           [[VAR_reinterpret_cast_1_:%.+]] = memref.reinterpret_cast [[PARAM_0_]] to offset: {{.}}[[VAR_10_]]{{.}}, sizes: [4], strides: [1] : memref<*xf32> to memref<4xf32, strided<[1], offset: ?>>
// CHECK-DAG:           [[RES_1_:%.+]] = memref.alloc() : memref<4xf32>
// CHECK:               memref.copy [[VAR_reinterpret_cast_1_]], [[RES_1_]] : memref<4xf32, strided<[1], offset: ?>> to memref<4xf32>
// CHECK-DAG:           [[VAR_11_:%.+]] = bufferization.to_tensor [[RES_1_]] restrict writable : memref<4xf32> to tensor<4xf32>
// CHECK-DAG:           [[VAR_reinterpret_cast_3_:%.+]] = memref.reinterpret_cast [[PARAM_1_]] to offset: {{.}}[[VAR_arg13_]]{{.}}, sizes: [4], strides: [1] : memref<*xf32> to memref<4xf32, strided<[1], offset: ?>>
// CHECK:               bufferization.materialize_in_destination [[VAR_11_]] in writable [[VAR_reinterpret_cast_3_]] : (tensor<4xf32>, memref<4xf32, strided<[1], offset: ?>>) -> ()
// CHECK-DAG:           [[VAR_12_:%.+]] = arith.addi [[VAR_10_]], [[CST_4_]] : index
// CHECK-DAG:           [[VAR_13_:%.+]] = arith.addi [[VAR_arg13_]], [[CST_4_]] : index
// CHECK:               scf.yield [[VAR_12_]], [[VAR_13_]] : index, index
// CHECK:             }
// CHECK:             scf.yield [[VAR_7_]]#0, [[VAR_7_]]#1 : index, index
// CHECK:           }
// CHECK:           return
// CHECK:         }
