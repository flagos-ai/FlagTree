// RUN: triton-shared-opt --triton-to-linalg-experimental %s | FileCheck %s

module {
  tt.func @reduce_kernel_2d_0d(%arg0: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %arg1: i32, %arg2: i32, %arg3: i32, %arg4: i32, %arg5: i32, %arg6: i32) {
    %c2_i32 = arith.constant 2 : i32
    %c8_i32 = arith.constant 8 : i32
    %c0_i32 = arith.constant 0 : i32
    %c1_i32 = arith.constant 1 : i32
    %0 = tt.addptr %arg0, %arg4 : !tt.ptr<f32>, i32
    %1 = scf.for %arg7 = %c0_i32 to %c8_i32 step %c1_i32 iter_args(%arg8 = %0) -> (!tt.ptr<f32>)  : i32 {
      %2 = scf.for %arg9 = %c0_i32 to %c2_i32 step %c1_i32 iter_args(%arg10 = %arg8) -> (!tt.ptr<f32>)  : i32 {
        %3 = arith.muli %arg7, %arg9 : i32
        %4 = arith.sitofp %3 : i32 to f32
        tt.store %arg10, %4 : !tt.ptr<f32>
        %5 = tt.addptr %arg10, %c1_i32 : !tt.ptr<f32>, i32
        scf.yield %5 : !tt.ptr<f32>
      }
      scf.yield %2 : !tt.ptr<f32>
    }
    tt.return
  }
}

// CHECK-LABEL:  func.func @reduce_kernel_2d_0d
// CHECK-SAME:   ([[PARAM_0_:%.+]]: memref<*xf32> {tt.divisibility = 16 : i32}, [[PARAM_1_:%.+]]: i32, [[PARAM_2_:%.+]]: i32, [[PARAM_3_:%.+]]: i32, [[PARAM_4_:%.+]]: i32, [[PARAM_5_:%.+]]: i32, [[PARAM_6_:%.+]]: i32, [[PARAM_7_:%.+]]: i32, [[PARAM_8_:%.+]]: i32, [[PARAM_9_:%.+]]: i32, [[PARAM_10_:%.+]]: i32, [[PARAM_11_:%.+]]: i32, [[PARAM_12_:%.+]]: i32) {
// CHECK-DAG:       [[CST_1_1_:%.+]] = arith.constant 1 : i32
// CHECK-DAG:       [[CST_8_:%.+]] = arith.constant 8 : i32
// CHECK-DAG:       [[CST_2_:%.+]] = arith.constant 2 : i32
// CHECK-DAG:       [[CST_0_:%.+]] = arith.constant 0 : i32
// CHECK-DAG:       [[VAR_1_:%.+]] = scf.for [[VAR_arg13_:%.+]] = [[CST_0_]] to [[CST_8_]] step [[CST_1_1_]] iter_args([[VAR_arg14_:%.+]] = [[PARAM_4_]]) -> (i32)  : i32 {
// CHECK-DAG:         [[VAR_2_:%.+]] = scf.for [[VAR_arg16_:%.+]] = [[CST_0_]] to [[CST_2_]] step [[CST_1_1_]] iter_args([[VAR_arg17_:%.+]] = [[VAR_arg14_]]) -> (i32)  : i32 {
// CHECK-DAG:           [[VAR_3_:%.+]] = arith.muli [[VAR_arg13_]], [[VAR_arg16_]] : i32
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:           [[VAR_4_:%.+]] = arith.sitofp [[VAR_3_]] : i32 to f32
// CHECK-DAG:           [[VAR_5_:%.+]] = arith.index_cast [[VAR_arg17_]] : i32 to index
// CHECK:               [[VAR_reinterpret_cast_:%.+]] = memref.reinterpret_cast [[PARAM_0_]] to offset: {{.}}[[VAR_5_]]{{.}}, sizes: [1], strides: [1] : memref<*xf32> to memref<1xf32, strided<[1], offset: ?>>
// CHECK:               affine.store [[VAR_4_]], [[VAR_reinterpret_cast_]][0] : memref<1xf32, strided<[1], offset: ?>>
// CHECK-DAG:           [[VAR_7_:%.+]] = arith.addi [[VAR_arg17_]], [[CST_1_1_]] : i32
// CHECK:               scf.yield [[VAR_7_]] : i32
// CHECK:             }
// CHECK:             scf.yield [[VAR_2_]] : i32
// CHECK:           }
// CHECK:           return
// CHECK:         }
