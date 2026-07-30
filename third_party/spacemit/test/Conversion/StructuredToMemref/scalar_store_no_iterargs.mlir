// RUN: triton-shared-opt --triton-to-linalg-experimental %s | FileCheck %s

module {
  tt.func @reduce_kernel_2d_0d1d2de3de(%arg0: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %arg1: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %arg2: i32 {tt.divisibility = 16 : i32, tt.max_divisibility = 16 : i32}, %arg3: i32 {tt.divisibility = 16 : i32, tt.max_divisibility = 16 : i32}, %arg4: i32, %arg5: i32, %arg6: i32, %arg7: i32, %arg8: i32, %arg9: i32) {
    %c1_i32 = arith.constant 1 : i32
    %c5_i32 = arith.constant 5 : i32
    %c0_i32 = arith.constant 0 : i32
    %0 = tt.addptr %arg1, %arg7 : !tt.ptr<f32>, i32
    %1 = arith.sitofp %arg7 : i32 to f32
    scf.for %arg10 = %c0_i32 to %c5_i32 step %c1_i32  : i32 {
      %2 = tt.addptr %0, %arg10 : !tt.ptr<f32>, i32
      tt.store %2, %1 : !tt.ptr<f32>
    }
    tt.return
  }
}

// CHECK-LABEL:  func.func @reduce_kernel_2d_0d1d2de3de
// CHECK-SAME:   ([[PARAM_0_:%.+]]: memref<*xf32> {tt.divisibility = 16 : i32}, [[PARAM_1_:%.+]]: memref<*xf32> {tt.divisibility = 16 : i32}, [[PARAM_2_:%.+]]: i32 {tt.divisibility = 16 : i32, tt.max_divisibility = 16 : i32}, [[PARAM_3_:%.+]]: i32 {tt.divisibility = 16 : i32, tt.max_divisibility = 16 : i32}, [[PARAM_4_:%.+]]: i32, [[PARAM_5_:%.+]]: i32, [[PARAM_6_:%.+]]: i32, [[PARAM_7_:%.+]]: i32, [[PARAM_8_:%.+]]: i32, [[PARAM_9_:%.+]]: i32, [[PARAM_10_:%.+]]: i32, [[PARAM_11_:%.+]]: i32, [[PARAM_12_:%.+]]: i32, [[PARAM_13_:%.+]]: i32, [[PARAM_14_:%.+]]: i32, [[PARAM_15_:%.+]]: i32) {
// CHECK-DAG:       [[CST_5_:%.+]] = arith.constant 5 : i32
// CHECK-DAG:       [[CST_1_:%.+]] = arith.constant 1 : i32
// CHECK-DAG:       [[CST_0_:%.+]] = arith.constant 0 : i32
// CHECK-DAG:       [[VAR_0_:%.+]] = arith.sitofp [[PARAM_7_]] : i32 to f32
// CHECK:           scf.for [[I_0_:%.+]] = [[CST_0_]] to [[CST_5_]] step [[CST_1_]]  : i32 {
// CHECK:             [[VAR_1_:%.+]] = arith.addi [[PARAM_7_]], [[I_0_]] : i32
// CHECK:             [[VAR_2_:%.+]] = arith.index_cast [[VAR_1_]] : i32 to index
// CHECK:             [[VAR_reinterpret_cast_:%.+]] = memref.reinterpret_cast [[PARAM_1_]] to offset: {{.}}[[VAR_2_]]{{.}}, sizes: [1], strides: [1] : memref<*xf32> to memref<1xf32, strided<[1], offset: ?>>
// CHECK:             affine.store [[VAR_0_]], [[VAR_reinterpret_cast_]][0] : memref<1xf32, strided<[1], offset: ?>>
// CHECK:           }
// CHECK:           return
// CHECK:         }
