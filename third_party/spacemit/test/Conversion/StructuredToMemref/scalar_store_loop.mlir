// RUN: triton-shared-opt --triton-to-linalg-experimental %s | FileCheck %s
// TODO: Fix pending https://github.com/microsoft/triton-shared/pull/215
// XFAIL: *

module {
  tt.func @reduce_kernel_2d_0d(%arg0: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %arg1: i32, %arg2: i32, %arg3: i32, %arg4: i32, %arg5: i32, %arg6: i32) {
    %c8_i32 = arith.constant 8 : i32
    %c0_i32 = arith.constant 0 : i32
    %c1_i32 = arith.constant 1 : i32
    %0 = scf.for %arg7 = %c0_i32 to %c8_i32 step %c1_i32 iter_args(%arg8 = %arg0) -> (!tt.ptr<f32>)  : i32 {
      %1 = arith.sitofp %arg7 : i32 to f32
      tt.store %arg8, %1 : !tt.ptr<f32>
      %2 = tt.addptr %arg8, %c1_i32 : !tt.ptr<f32>, i32
      scf.yield %2 : !tt.ptr<f32>
    }
    tt.return
  }
}

// CHECK-LABEL:  func.func @reduce_kernel_2d_0d
// CHECK-SAME:   ([[PARAM_0_:%.+]]: memref<*xf32> {tt.divisibility = 16 : i32}, [[PARAM_1_:%.+]]: i32, [[PARAM_2_:%.+]]: i32, [[PARAM_3_:%.+]]: i32, [[PARAM_4_:%.+]]: i32, [[PARAM_5_:%.+]]: i32, [[PARAM_6_:%.+]]: i32) {
// CHECK-DAG:       %[[CST_0_:.+]] = arith.constant 0 : index
// CHECK-DAG:       [[CST_1_:%.+]] = arith.constant 1 : index
// CHECK-DAG:       [[CST_1_1_:%.+]] = arith.constant 1 : i32
// CHECK-DAG:       [[CST_0_1_:%.+]] = arith.constant 0 : i32
// CHECK-DAG:       [[CST_8_:%.+]] = arith.constant 8 : i32
// CHECK-DAG:       [[VAR_reinterpret_cast_:%.+]] = memref.reinterpret_cast [[PARAM_0_]] to offset: [0], sizes: [1], strides: [1] : memref<*xf32> to memref<1xf32, strided<[1], offset: ?>>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:       [[VAR_0_:%.+]]:2 = scf.for [[VAR_arg7_:%.+]] = [[CST_0_1_]] to [[CST_8_]] step [[CST_1_1_]] iter_args([[VAR_arg8_:%.+]] = [[VAR_reinterpret_cast_]], [[VAR_arg9_:%.+]] = %[[CST_0_]]) -> (memref<1xf32, strided<[1], offset: ?>>, index)  : i32 {
// CHECK-DAG:         [[VAR_1_:%.+]] = arith.sitofp [[VAR_arg7_]] : i32 to f32
// CHECK:             memref.store [[VAR_1_]], [[VAR_arg8_]][%[[CST_0_]]] : memref<1xf32, strided<[1], offset: ?>>
// CHECK:             [[VAR_2_:%.+]] = arith.addi [[VAR_arg9_]], [[CST_1_]] : index
// CHECK:             [[VAR_reinterpret_cast_0_:%.+]] = memref.reinterpret_cast [[VAR_arg8_]] to offset: {{.}}[[VAR_2_]]{{.}}, sizes: [1], strides: [1] : memref<1xf32, strided<[1], offset: ?>> to memref<1xf32, strided<[1], offset: ?>>
// CHECK:             scf.yield [[VAR_reinterpret_cast_0_]], [[VAR_2_]] : memref<1xf32, strided<[1], offset: ?>>, index
// CHECK:           }
// CHECK:           return
// CHECK:         }
